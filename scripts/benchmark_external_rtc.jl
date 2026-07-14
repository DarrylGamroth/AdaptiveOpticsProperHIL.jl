using AdaptiveOpticsProperHIL
using JSON3
using Sockets

host = parse(IPAddr, get(ENV, "AOS_RTC_HOST", "127.0.0.1"))
port = parse(Int, get(ENV, "AOS_RTC_PORT", "50000"))
resolution = parse(Int, get(ENV, "AOS_RTC_RESOLUTION", "64"))
n_act = parse(Int, get(ENV, "AOS_RTC_N_ACT", "8"))
warmup = parse(Int, get(ENV, "AOS_RTC_WARMUP", "10"))
samples = parse(Int, get(ENV, "AOS_RTC_SAMPLES", "1000"))
trials = parse(Int, get(ENV, "AOS_RTC_TRIALS", "3"))
deadline_ns = parse(Int, get(ENV, "AOS_RTC_DEADLINE_NS", "1000000"))
command_limit = parse(Float64, get(ENV, "AOS_RTC_COMMAND_LIMIT", "1.0"))
retain_raw_samples = lowercase(get(ENV, "AOS_RTC_RETAIN_RAW_SAMPLES", "true")) in
    ("1", "true", "yes", "on")
histogram_highest_trackable_ns = parse(Int,
    get(ENV, "AOS_RTC_HISTOGRAM_MAX_NS", "3600000000000"))
histogram_significant_figures = parse(Int,
    get(ENV, "AOS_RTC_HISTOGRAM_SIGFIGS", "3"))

ctx = build_hil_context(
    resolution=resolution,
    n_act=n_act,
    T=Float32,
)
harness = connect_external_rtc(ctx;
    host=host,
    port=port,
    deadline_ns=deadline_ns,
    command_limit=command_limit,
)
try
    result = benchmark_external_rtc!(harness;
        warmup=warmup,
        samples=samples,
        trials=trials,
        retain_raw_samples=retain_raw_samples,
        histogram_highest_trackable_ns=histogram_highest_trackable_ns,
        histogram_significant_figures=histogram_significant_figures,
    )
    result_path = get(ENV, "AOS_RTC_RESULT", "")
    if isempty(result_path)
        JSON3.pretty(stdout, result)
        write(stdout, '\n')
    else
        write_result(result_path, result; meta=(
            backend="cpu",
            transport="tcp_loopback",
            arrival_model="closed_loop_request_response",
            julia_version=string(VERSION),
            cpu=Sys.CPU_NAME,
            cpu_threads=Sys.CPU_THREADS,
            julia_threads=Threads.nthreads(),
            source_revision=get(ENV, "AOS_SOURCE_REVISION", "worktree"),
            resolution=resolution,
            n_act=n_act,
            precision="Float32",
        ))
        println("wrote ", result_path)
        println("cycle_p50_ns=", result.cycle.p50_ns)
        println("cycle_p99_ns=", result.cycle.p99_ns)
        println("transport_p50_ns=", result.transport.p50_ns)
        println("transport_p99_ns=", result.transport.p99_ns)
        if !isnothing(result.cycle.p99_9_ns)
            println("cycle_p99_9_ns=", result.cycle.p99_9_ns)
            println("transport_p99_9_ns=", result.transport.p99_9_ns)
        end
        println("allocated_bytes_per_cycle=",
            result.allocated_bytes_per_cycle)
        println("deadline_misses=", result.deadline_misses)
    end
finally
    close(harness)
end
