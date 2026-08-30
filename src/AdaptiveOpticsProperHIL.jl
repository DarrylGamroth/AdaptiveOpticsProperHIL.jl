module AdaptiveOpticsProperHIL

using AdaptiveOpticsSim
using Base64
using BenchmarkTools
using Dates
using HdrHistogram
using JSON3
using LinearAlgebra
using Proper
using Random
using Sockets
using Statistics
using TOML

export HILPayload,
    HILContext,
    deterministic_opd_cases,
    proper_field_from_payload,
    expected_field_from_payload,
    convention_metrics,
    validate_conventions,
    build_hil_context,
    set_synthetic_command!,
    ao_step!,
    science_step!,
    hil_step!,
    benchmark_hil!,
    write_result,
    DEFAULT_BENCHMARK_MATRIX,
    load_benchmark_matrix,
    benchmark_baseline_path,
    build_benchmark_case,
    benchmark_case_step!,
    benchmark_case_output,
    run_benchmark_matrix,
    write_benchmark_result,
    compare_benchmark_results

export RTCProtocolError,
    ExternalRTCHarness,
    connect_external_rtc,
    external_rtc_step!,
    external_hil_step!,
    rtc_cycle_metrics,
    benchmark_external_rtc!,
    serve_mock_rtc!

export ProperPropagationConfiguration,
    proper_propagation_node,
    proper_propagation_node_factory,
    prepare_proper_assets,
    reset_proper_assets!

export SpidersEvidenceLevel,
    SpidersDocumented,
    SpidersDeploymentConfigured,
    SpidersInferred,
    SpidersPlaceholder,
    SpidersProfileClaim,
    SpidersFocalPlaneMask,
    SpidersGalilAlignment,
    SpidersFilterSelection,
    SpidersOpticalGeometry,
    ProvisionalSpidersProfile,
    provisional_spiders_h_regular_profile,
    spiders_profile_claims,
    spiders_profile_is_provisional

struct HILPayload{O,P,T}
    opd_m::O
    pupil::P
    diameter_m::T
    focal_length_m::T
    lyot_stop_norm::T
end

mutable struct HILContext{SC,SIM,MODEL,HOT,PAYLOAD,CMD,HCMD,R1,R2,FB,WF,MB,OB,T}
    scenario::SC
    sim::SIM
    science_model::MODEL
    science_hot_call::HOT
    payload::PAYLOAD
    command_buffer::CMD
    command_host::HCMD
    tiptilt_range::R1
    dm_range::R2
    field_buffer::FB
    wavefront::WF
    mask_buffer::MB
    output_buffer::OB
    step_index::Int
    wavelength_um::T
end

function deterministic_opd_cases(::Type{T}=Float64, n::Integer=32) where {T<:AbstractFloat}
    n = Int(n)
    c = T((n + 1) / 2)
    scale = T(n / 2)
    x = [((T(i) - c) / scale) for i in 1:n, _ in 1:n]
    y = [((T(j) - c) / scale) for _ in 1:n, j in 1:n]
    asymmetric = zeros(T, n, n)
    asymmetric[clamp(n ÷ 4, 1, n), clamp(3n ÷ 5, 1, n)] = T(7e-9)
    asymmetric[clamp(3n ÷ 4, 1, n), clamp(n ÷ 3, 1, n)] = T(-3e-9)
    poke = zeros(T, n, n)
    poke[clamp(n ÷ 2 + 3, 1, n), clamp(n ÷ 2 - 5, 1, n)] = T(1e-8)
    return (
        piston=fill(T(2e-9), n, n),
        x_tilt=T(8e-9) .* x,
        y_tilt=T(-6e-9) .* y,
        focus=T(4e-9) .* (x .^ 2 .+ y .^ 2 .- T(0.5)),
        asymmetric=asymmetric,
        single_poke=poke,
    )
end

function _to_backend(A::AbstractArray, backend_array::Type{<:AbstractArray}, ::Type{T}) where {T<:AbstractFloat}
    return backend_array(T.(A))
end

function _bool_to_backend(A::AbstractArray{Bool}, backend_array::Type{<:AbstractArray})
    return backend_array(Array{Bool}(A))
end

function make_payload(opd::AbstractMatrix{T}; pupil=nothing, diameter_m::Real=8.0,
    focal_length_m::Real=80.0, lyot_stop_norm::Real=0.9) where {T<:AbstractFloat}
    p = pupil === nothing ? trues(size(opd)) : pupil
    return HILPayload(opd, p, T(diameter_m), T(focal_length_m), T(lyot_stop_norm))
end

function proper_field_from_payload(payload::HILPayload, wavelength_m::Real, n::Integer; context=nothing)
    wf = prop_begin(payload.diameter_m, wavelength_m, n; beam_diam_fraction=1.0, context=context)
    prop_multiply(wf, payload.pupil)
    prop_add_phase(wf, payload.opd_m)
    return wf.field
end

function expected_field_from_payload(payload::HILPayload, wavelength_m::Real)
    T = float(real(eltype(payload.opd_m)))
    shifted_pupil = Proper.prop_shift_center(Array(payload.pupil); inverse=true)
    shifted_opd = Proper.prop_shift_center(Array(payload.opd_m); inverse=true)
    return shifted_pupil .* cis.((T(2pi) / T(wavelength_m)) .* shifted_opd)
end

function convention_metrics(payload::HILPayload, wavelength_m::Real; context=nothing)
    field = Array(proper_field_from_payload(payload, wavelength_m, size(payload.opd_m, 1); context=context))
    expected = expected_field_from_payload(payload, wavelength_m)
    diff = field .- expected
    return (
        max_abs=maximum(abs, diff),
        rms=sqrt(mean(abs2, diff)),
        field_norm=norm(field),
        expected_norm=norm(expected),
    )
end

function validate_conventions(; T::Type{<:AbstractFloat}=Float64, n::Integer=32,
    wavelength_m::Real=1.65e-6, backend_array::Type{<:AbstractArray}=Array,
    context=nothing, atol::Real=100eps(T))
    cases = deterministic_opd_cases(T, n)
    pupil = trues(Int(n), Int(n))
    results = Dict{Symbol,NamedTuple}()
    for (name, opd_host) in pairs(cases)
        opd = backend_array === Array ? copy(opd_host) : _to_backend(opd_host, backend_array, T)
        p = backend_array === Array ? copy(pupil) : _bool_to_backend(pupil, backend_array)
        payload = make_payload(opd; pupil=p)
        metrics = convention_metrics(payload, wavelength_m; context=context)
        metrics.max_abs <= atol || throw(ArgumentError("OPD convention validation failed for $(name): max_abs=$(metrics.max_abs), atol=$(atol)"))
        results[name] = metrics
    end
    return results
end

function science_prescription(λm, n; payload::HILPayload, field=nothing, wavefront=nothing, mask=nothing, output=nothing, run_context=nothing)
    wf = if field === nothing
        prop_begin(payload.diameter_m, λm, n; beam_diam_fraction=1.0, context=run_context)
    elseif wavefront !== nothing
        prop_begin!(wavefront, payload.diameter_m, λm; beam_diam_fraction=1.0)
    else
        prop_begin!(field, payload.diameter_m, λm; beam_diam_fraction=1.0, context=run_context)
    end
    prop_multiply(wf, payload.pupil)
    prop_add_phase(wf, payload.opd_m)
    _prop_lens!(wf, payload.focal_length_m, run_context, "science arm")
    _prop_propagate!(wf, payload.focal_length_m, run_context, "focal plane mask")
    if mask === nothing
        prop_8th_order_mask(wf, 4.0; circular=true)
    else
        prop_8th_order_mask!(mask, wf, 4.0; circular=true)
    end
    _prop_propagate!(wf, payload.focal_length_m, run_context, "lyot relay")
    _prop_lens!(wf, payload.focal_length_m, run_context, "lyot relay")
    _prop_propagate!(wf, 2payload.focal_length_m, run_context, "lyot stop")
    prop_circular_aperture(wf, payload.lyot_stop_norm; NORM=true)
    _prop_propagate!(wf, payload.focal_length_m, run_context, "science camera")
    _prop_lens!(wf, payload.focal_length_m, run_context, "science camera")
    _prop_propagate!(wf, payload.focal_length_m, run_context, "final focus")
    return output === nothing ? prop_end(wf) : prop_end(wf, output)
end

@inline function _prop_lens!(wf, focal_length_m, ::Nothing, name)
    return prop_lens(wf, focal_length_m, name)
end

@inline function _prop_lens!(wf, focal_length_m, run_context::Proper.RunContext, name)
    return prop_lens(wf, focal_length_m, run_context, name)
end

@inline function _prop_propagate!(wf, distance_m, ::Nothing, name)
    return prop_propagate(wf, distance_m, name)
end

@inline function _prop_propagate!(wf, distance_m, run_context::Proper.RunContext, name)
    return prop_propagate(wf, distance_m, run_context, name)
end

function _segment_range(layout, label::Symbol)
    for seg in AdaptiveOpticsSim.command_segments(layout)
        seg.label == label && return AdaptiveOpticsSim.command_segment_range(seg)
    end
    throw(ArgumentError("missing command segment $(label)"))
end

function build_hil_context(;
    backend=AdaptiveOpticsSim.Backends.CPUBackend(),
    T::Type{<:AbstractFloat}=Float32,
    resolution::Int=128,
    n_act::Int=16,
    diameter::Real=8.0,
    science_wavelength_um::Real=1.65,
    focal_length_m::Real=80.0,
    lyot_stop_norm::Real=0.9,
    rng_seed::Integer=20260503,
)
    tel = AdaptiveOpticsSim.Telescope(
        resolution=resolution,
        diameter=diameter,
        sampling_time=T(1e-3),
        central_obstruction=T(0.1),
        T=T,
        backend=backend,
    )
    src = AdaptiveOpticsSim.Source(band=:H, magnitude=0.0)
    atm = AdaptiveOpticsSim.KolmogorovAtmosphere(tel; r0=T(0.2), L0=T(25.0))
    tiptilt = AdaptiveOpticsSim.TipTiltMirror(tel; scale=T(0.05), T=T, backend=backend, label=:tiptilt)
    dm = AdaptiveOpticsSim.DeformableMirror(tel; n_act=n_act, influence_width=T(0.3), T=T, backend=backend)
    optic = AdaptiveOpticsSim.CompositeControllableOptic(:tiptilt => tiptilt, :dm => dm)
    wfs = AdaptiveOpticsSim.ShackHartmannWFS(tel; n_lenslets=16, mode=AdaptiveOpticsSim.Diffractive(), threshold=T(0), T=T, backend=backend)
    sim = AdaptiveOpticsSim.AOSimulation(tel, src, atm, optic, wfs)
    branch = AdaptiveOpticsSim.ControlLoopBranch(:main, sim, AdaptiveOpticsSim.NullReconstructor(); rng=AdaptiveOpticsSim.runtime_rng(rng_seed))
    cfg = AdaptiveOpticsSim.SingleControlLoopConfig(
        name=:proper_hil_validation,
        branch_label=:main,
        outputs=AdaptiveOpticsSim.RuntimeOutputRequirements(slopes=true),
    )
    scenario = AdaptiveOpticsSim.build_control_loop_scenario(cfg, branch)
    AdaptiveOpticsSim.prepare!(scenario)

    proper_ctx = Proper.RunContext(typeof(sim.tel.state.opd))
    science_model = Proper.prepare_model(
        :proper_hil_validation_science,
        science_prescription,
        science_wavelength_um,
        resolution;
        context=proper_ctx,
        precision=T,
        pool_size=1,
    )
    payload = HILPayload(
        AdaptiveOpticsSim.opd_map(sim.tel),
        AdaptiveOpticsSim.pupil_mask(sim.tel),
        T(sim.tel.params.diameter),
        T(focal_length_m),
        T(lyot_stop_norm),
    )
    command_buffer = similar(AdaptiveOpticsSim.command(scenario))
    copyto!(command_buffer, AdaptiveOpticsSim.command(scenario))
    command_host = Vector{T}(undef, length(command_buffer))
    copyto!(command_host, Array(command_buffer))
    field_buffer = similar(AdaptiveOpticsSim.opd_map(sim.tel), Complex{T}, resolution, resolution)
    run_context = Proper.prepared_context(science_model, 1)
    wavefront = Proper.prop_begin!(
        field_buffer,
        payload.diameter_m,
        T(science_wavelength_um) * T(1e-6);
        beam_diam_fraction=1.0,
        context=run_context,
    )
    mask_buffer = similar(AdaptiveOpticsSim.opd_map(sim.tel), T, resolution, resolution)
    output_buffer = similar(AdaptiveOpticsSim.opd_map(sim.tel), T, resolution, resolution)
    science_hot_call = Proper.prepare_run(
        science_model;
        slot=1,
        activate_context=false,
        payload=payload,
        field=field_buffer,
        wavefront=wavefront,
        mask=mask_buffer,
        output=output_buffer,
        run_context=run_context,
    )
    layout = AdaptiveOpticsSim.command_layout(scenario)
    return HILContext(
        scenario,
        sim,
        science_model,
        science_hot_call,
        payload,
        command_buffer,
        command_host,
        _segment_range(layout, :tiptilt),
        _segment_range(layout, :dm),
        field_buffer,
        wavefront,
        mask_buffer,
        output_buffer,
        0,
        T(science_wavelength_um),
    )
end

function set_synthetic_command!(ctx::HILContext)
    ctx.step_index += 1
    T = eltype(ctx.command_host)
    tstep = T(ctx.step_index)
    tiptilt_start = first(ctx.tiptilt_range)
    ctx.command_host[tiptilt_start] = T(5e-3) * sin(T(0.1) * tstep)
    ctx.command_host[tiptilt_start + 1] = -T(5e-3) * cos(T(0.1) * tstep)
    dm_value = T(5e-9) * tstep
    @inbounds for i in ctx.dm_range
        ctx.command_host[i] = dm_value
    end
    copyto!(ctx.command_buffer, ctx.command_host)
    AdaptiveOpticsSim.set_command!(ctx.scenario, ctx.command_buffer)
    return ctx.command_buffer
end

function ao_step!(ctx::HILContext)
    set_synthetic_command!(ctx)
    AdaptiveOpticsSim.sense!(ctx.scenario)
    return AdaptiveOpticsSim.slopes(ctx.scenario)
end

science_step!(ctx::HILContext) = Proper.prop_run(ctx.science_hot_call)

function hil_step!(ctx::HILContext)
    ao_step!(ctx)
    return science_step!(ctx)
end

function _trial_stats(trial::BenchmarkTools.Trial)
    return (
        median_ns=Float64(median(trial.times)),
        mean_ns=Float64(mean(trial.times)),
        min_ns=Float64(minimum(trial.times)),
        max_ns=Float64(maximum(trial.times)),
        samples=length(trial.times),
        median_allocs=Float64(median(trial.allocs)),
        median_bytes=Float64(median(trial.memory)),
    )
end

function benchmark_hil!(ctx::HILContext; samples::Integer=20, seconds::Real=5.0, warmup::Integer=3)
    for _ in 1:warmup
        hil_step!(ctx)
    end
    trial = @benchmark hil_step!($ctx) samples=Int(samples) seconds=Float64(seconds) evals=1
    psf, sampling = science_step!(ctx)
    return (
        policy="combined AO command-to-Proper pixels HIL benchmark; BenchmarkTools warmed steady-state; TTFx excluded",
        stats=_trial_stats(trial),
        output=(psf_size=size(psf), sampling=sampling, step_index=ctx.step_index),
    )
end

function write_result(path::AbstractString, result; meta=(;))
    mkpath(dirname(path))
    payload = (; meta=merge((timestamp_utc=Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sss"),), meta), result=result)
    open(path, "w") do io
        JSON3.pretty(io, payload)
        write(io, '\n')
    end
    return path
end

include("benchmark_matrix.jl")
include("latency_histograms.jl")
include("external_rtc.jl")
include("proper_algorithm.jl")
include("spiders_provisional.jl")

end
