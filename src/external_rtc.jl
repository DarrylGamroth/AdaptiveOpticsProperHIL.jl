const RTC_WIRE_MAGIC = UInt32(0x54524f41) # "AORT" on the wire
const RTC_WIRE_VERSION = UInt16(1)
const RTC_HEADER_BYTES = 32

@enum RTCMessageKind::UInt16 begin
    RTCSlopes = 1
    RTCCommands = 2
end

"""A malformed, stale, or unsafe external-RTC wire response."""
struct RTCProtocolError <: Exception
    message::String
end

Base.showerror(io::IO, err::RTCProtocolError) =
    print(io, "RTC protocol error: ", err.message)

@inline rtc_scalar_word(::Type{Float32}) = UInt32
@inline rtc_scalar_word(::Type{Float64}) = UInt64
@inline rtc_scalar_code(::Type{Float32}) = UInt16(1)
@inline rtc_scalar_code(::Type{Float64}) = UInt16(2)

struct RTCFrame{
    T<:Union{Float32,Float64},
    U<:Unsigned,
    H<:AbstractVector{UInt32},
    P<:AbstractVector{U},
}
    bytes::Vector{UInt8}
    header::H
    payload::P
end

function RTCFrame(::Type{T}, count::Integer) where {T<:Union{Float32,Float64}}
    count >= 0 || throw(ArgumentError("RTC frame element count must be >= 0"))
    count <= typemax(UInt32) ||
        throw(ArgumentError("RTC frame element count exceeds wire v1"))
    U = rtc_scalar_word(T)
    bytes = zeros(UInt8, RTC_HEADER_BYTES + sizeof(T) * Int(count))
    header = reinterpret(UInt32, @view(bytes[1:RTC_HEADER_BYTES]))
    payload = reinterpret(U, @view(bytes[(RTC_HEADER_BYTES + 1):end]))
    return RTCFrame{T,U,typeof(header),typeof(payload)}(bytes, header,
        payload)
end

@inline function _rtc_store_u64!(header, index::Int, value::UInt64)
    header[index] = htol(UInt32(value & typemax(UInt32)))
    header[index + 1] = htol(UInt32(value >> 32))
    return value
end

@inline function _rtc_load_u64(header, index::Int)
    low = UInt64(ltoh(header[index]))
    high = UInt64(ltoh(header[index + 1]))
    return low | (high << 32)
end

function encode_rtc_frame!(frame::RTCFrame{T,U}, kind::RTCMessageKind,
    sequence::UInt64, timestamp_ns::UInt64,
    values::AbstractVector) where {T,U}
    length(values) == length(frame.payload) ||
        throw(DimensionMismatch("RTC payload length does not match frame"))
    header = frame.header
    header[1] = htol(RTC_WIRE_MAGIC)
    header[2] = htol(UInt32(RTC_WIRE_VERSION) |
        (UInt32(UInt16(kind)) << 16))
    _rtc_store_u64!(header, 3, sequence)
    _rtc_store_u64!(header, 5, timestamp_ns)
    header[7] = htol(UInt32(length(values)))
    header[8] = htol(UInt32(rtc_scalar_code(T)))
    @inbounds for i in eachindex(frame.payload, values)
        frame.payload[i] = htol(reinterpret(U, T(values[i])))
    end
    return frame
end

function decode_rtc_frame!(out::AbstractVector{T}, frame::RTCFrame{T,U},
    expected_kind::RTCMessageKind, expected_sequence::UInt64) where {T,U}
    header = frame.header
    ltoh(header[1]) == RTC_WIRE_MAGIC ||
        throw(RTCProtocolError("invalid frame magic"))
    version_kind = ltoh(header[2])
    version = UInt16(version_kind & 0xffff)
    kind = UInt16(version_kind >> 16)
    version == RTC_WIRE_VERSION ||
        throw(RTCProtocolError("unsupported wire version $(version)"))
    kind == UInt16(expected_kind) ||
        throw(RTCProtocolError("unexpected message kind $(kind)"))
    sequence = _rtc_load_u64(header, 3)
    sequence == expected_sequence ||
        throw(RTCProtocolError("expected sequence $(expected_sequence), received $(sequence)"))
    count = Int(ltoh(header[7]))
    count == length(out) == length(frame.payload) ||
        throw(RTCProtocolError("payload count $(count) does not match configured count $(length(out))"))
    scalar_code = UInt16(ltoh(header[8]) & 0xffff)
    scalar_code == rtc_scalar_code(T) ||
        throw(RTCProtocolError("payload scalar code $(scalar_code) does not match $(T)"))
    @inbounds for i in eachindex(out, frame.payload)
        out[i] = reinterpret(T, ltoh(frame.payload[i]))
    end
    return _rtc_load_u64(header, 5)
end

"""
    ExternalRTCHarness(context, socket; deadline_ns=1_000_000,
                       command_limit=Inf)

Bind a prepared `HILContext` to a connected RTC TCP socket. Wire and host
buffers are allocated once. `deadline_ns == 0` disables completed-cycle
deadline accounting; it does not configure a socket timeout.
"""
mutable struct ExternalRTCHarness{
    C,
    S<:TCPSocket,
    T<:Union{Float32,Float64},
    SF<:RTCFrame{T},
    CF<:RTCFrame{T},
}
    context::C
    socket::S
    slopes_host::Vector{T}
    slopes_frame::SF
    command_frame::CF
    sequence::UInt64
    deadline_ns::UInt64
    command_limit::T
    last_cycle_ns::UInt64
    last_transport_ns::UInt64
    deadline_misses::Int
    cycles::Int
    closed::Bool
end

function ExternalRTCHarness(context::HILContext, socket::TCPSocket;
    deadline_ns::Integer=1_000_000,
    command_limit::Real=Inf)
    deadline_ns >= 0 || throw(ArgumentError("RTC deadline_ns must be >= 0"))
    T = eltype(context.command_host)
    T <: Union{Float32,Float64} ||
        throw(ArgumentError("RTC wire v1 supports Float32 and Float64 contexts"))
    limit = T(command_limit)
    limit > zero(T) ||
        throw(ArgumentError("RTC command_limit must be greater than zero"))
    slopes = AdaptiveOpticsSim.slopes(context.scenario)
    slopes_host = Vector{T}(undef, length(slopes))
    fill!(slopes_host, zero(T))
    Sockets.nagle(socket, false)
    return ExternalRTCHarness(
        context,
        socket,
        slopes_host,
        RTCFrame(T, length(slopes_host)),
        RTCFrame(T, length(context.command_host)),
        UInt64(0),
        UInt64(deadline_ns),
        limit,
        UInt64(0),
        UInt64(0),
        0,
        0,
        false,
    )
end

"""
    connect_external_rtc(context; host=ip"127.0.0.1", port,
                         deadline_ns=1_000_000, command_limit=Inf)

Connect a prepared HIL context to an RTC speaking the v1 binary contract.
"""
function connect_external_rtc(context::HILContext;
    host::IPAddr=ip"127.0.0.1",
    port::Integer,
    deadline_ns::Integer=1_000_000,
    command_limit::Real=Inf)
    0 < port <= typemax(UInt16) ||
        throw(ArgumentError("RTC TCP port must be in 1:65535"))
    socket = Sockets.connect(host, port)
    try
        return ExternalRTCHarness(context, socket;
            deadline_ns=deadline_ns,
            command_limit=command_limit)
    catch
        close(socket)
        rethrow()
    end
end

@inline function _validate_rtc_commands!(commands::AbstractVector,
    limit::Real)
    @inbounds for command in commands
        isfinite(command) ||
            throw(RTCProtocolError("received a non-finite command"))
        abs(command) <= limit ||
            throw(RTCProtocolError("received a command outside the configured limit"))
    end
    return commands
end

"""
    external_rtc_step!(harness)

Sense one AO sample, exchange slopes for same-sequence commands, validate the
response, and stage it for the next AO sample. Returns the command buffer.
"""
function external_rtc_step!(harness::ExternalRTCHarness)
    harness.closed && throw(ArgumentError("external RTC harness is closed"))
    cycle_start = time_ns()
    AdaptiveOpticsSim.sense!(harness.context.scenario)
    copyto!(harness.slopes_host,
        AdaptiveOpticsSim.slopes(harness.context.scenario))
    sequence = harness.sequence + UInt64(1)
    encode_rtc_frame!(harness.slopes_frame, RTCSlopes, sequence,
        UInt64(time_ns()), harness.slopes_host)

    transport_start = time_ns()
    written = write(harness.socket, harness.slopes_frame.bytes)
    written == length(harness.slopes_frame.bytes) ||
        throw(EOFError())
    flush(harness.socket)
    read!(harness.socket, harness.command_frame.bytes)
    transport_end = time_ns()

    decode_rtc_frame!(harness.context.command_host, harness.command_frame,
        RTCCommands, sequence)
    _validate_rtc_commands!(harness.context.command_host,
        harness.command_limit)
    copyto!(harness.context.command_buffer, harness.context.command_host)
    AdaptiveOpticsSim.set_command!(harness.context.scenario,
        harness.context.command_buffer)

    harness.sequence = sequence
    harness.cycles += 1
    harness.last_transport_ns = UInt64(transport_end - transport_start)
    harness.last_cycle_ns = UInt64(time_ns() - cycle_start)
    if !iszero(harness.deadline_ns) &&
        harness.last_cycle_ns > harness.deadline_ns
        harness.deadline_misses += 1
    end
    return harness.context.command_buffer
end

"""Run one external RTC transaction and the Proper science arm for that sample."""
function external_hil_step!(harness::ExternalRTCHarness)
    external_rtc_step!(harness)
    return science_step!(harness.context)
end

"""Return the current external-RTC sequence, latency, and deadline counters."""
@inline rtc_cycle_metrics(harness::ExternalRTCHarness) = (
    sequence=harness.sequence,
    cycles=harness.cycles,
    last_cycle_ns=harness.last_cycle_ns,
    last_transport_ns=harness.last_transport_ns,
    deadline_ns=harness.deadline_ns,
    deadline_misses=harness.deadline_misses,
)

function Base.close(harness::ExternalRTCHarness)
    if !harness.closed
        close(harness.socket)
        harness.closed = true
    end
    return nothing
end

Base.isopen(harness::ExternalRTCHarness) =
    !harness.closed && isopen(harness.socket)

function _latency_summary(values::Vector{UInt64})
    isempty(values) && throw(ArgumentError("latency sample set cannot be empty"))
    sorted = sort(values)
    return (
        min_ns=first(sorted),
        p50_ns=sorted[clamp(ceil(Int, 0.50 * length(sorted)), 1, end)],
        p95_ns=sorted[clamp(ceil(Int, 0.95 * length(sorted)), 1, end)],
        p99_ns=sorted[clamp(ceil(Int, 0.99 * length(sorted)), 1, end)],
        max_ns=last(sorted),
    )
end

"""
    benchmark_external_rtc!(harness; warmup=10, samples=1_000, trials=3)

Measure repeated warmed closed-loop request/response cycles. Raw samples are
retained, tails are reported overall and per trial, and one separate
steady-state allocation transaction is recorded.
"""
function benchmark_external_rtc!(harness::ExternalRTCHarness;
    warmup::Integer=10,
    samples::Integer=1_000,
    trials::Integer=3)
    warmup >= 0 || throw(ArgumentError("warmup must be >= 0"))
    samples >= 100 ||
        throw(ArgumentError("samples must be >= 100 to report p99"))
    trials > 0 || throw(ArgumentError("trials must be > 0"))
    for _ in 1:Int(warmup)
        external_rtc_step!(harness)
    end
    allocated_bytes = @allocated external_rtc_step!(harness)
    cycle_trials = [Vector{UInt64}(undef, Int(samples))
        for _ in 1:Int(trials)]
    transport_trials = [similar(cycle_trials[1])
        for _ in 1:Int(trials)]
    misses_before = harness.deadline_misses
    for trial in eachindex(cycle_trials)
        cycle_latencies = cycle_trials[trial]
        transport_latencies = transport_trials[trial]
        for i in eachindex(cycle_latencies)
            external_rtc_step!(harness)
            cycle_latencies[i] = harness.last_cycle_ns
            transport_latencies[i] = harness.last_transport_ns
        end
    end
    cycle_latencies = vcat(cycle_trials...)
    transport_latencies = vcat(transport_trials...)
    return (
        policy="closed-loop request/response; warmed; raw samples retained; no coordinated-omission correction",
        warmup=Int(warmup),
        samples=Int(samples),
        trials=Int(trials),
        cycle=_latency_summary(cycle_latencies),
        transport=_latency_summary(transport_latencies),
        cycle_by_trial=_latency_summary.(cycle_trials),
        transport_by_trial=_latency_summary.(transport_trials),
        deadline_ns=harness.deadline_ns,
        deadline_misses=harness.deadline_misses - misses_before,
        allocated_bytes_per_cycle=allocated_bytes,
        allocation_probe_transactions=1,
        cycle_latencies_ns=cycle_trials,
        transport_latencies_ns=transport_trials,
    )
end

"""
    serve_mock_rtc!(socket; slope_count, command_count, T=Float32,
                    gain=0.1, max_cycles=typemax(Int))

Serve deterministic same-sequence commands on a connected socket for protocol
bring-up and loopback benchmarks. This is not a physical controller model.
"""
function serve_mock_rtc!(socket::TCPSocket;
    slope_count::Integer,
    command_count::Integer,
    T::Type{<:Union{Float32,Float64}}=Float32,
    gain::Real=0.1,
    max_cycles::Integer=typemax(Int))
    slope_count > 0 || throw(ArgumentError("slope_count must be > 0"))
    command_count > 0 || throw(ArgumentError("command_count must be > 0"))
    max_cycles >= 0 || throw(ArgumentError("max_cycles must be >= 0"))
    Sockets.nagle(socket, false)
    slopes_frame = RTCFrame(T, slope_count)
    command_frame = RTCFrame(T, command_count)
    slopes = zeros(T, slope_count)
    commands = zeros(T, command_count)
    scale = T(gain)
    served = 0
    for _ in 1:Int(max_cycles)
        try
            read!(socket, slopes_frame.bytes)
        catch err
            err isa EOFError || rethrow()
            break
        end
        sequence = _rtc_load_u64(slopes_frame.header, 3)
        decode_rtc_frame!(slopes, slopes_frame, RTCSlopes, sequence)
        @inbounds for i in eachindex(commands)
            commands[i] = -scale * slopes[mod1(i, length(slopes))]
        end
        encode_rtc_frame!(command_frame, RTCCommands, sequence,
            UInt64(time_ns()), commands)
        written = write(socket, command_frame.bytes)
        written == length(command_frame.bytes) || throw(EOFError())
        flush(socket)
        served += 1
    end
    return served
end
