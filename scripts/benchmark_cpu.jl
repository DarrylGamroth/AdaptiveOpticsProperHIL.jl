using AdaptiveOpticsProperHIL

samples = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
seconds = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 5.0
resolution = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 128

ctx = build_hil_context(resolution=resolution, n_act=16, T=Float32)
result = benchmark_hil!(ctx; samples=samples, seconds=seconds)
path = joinpath(@__DIR__, "..", "results", "cpu_hil_benchmark.json")
write_result(path, result; meta=(backend="cpu", resolution=resolution, precision="Float32"))
println("wrote ", path)
println("median_ns=", result.stats.median_ns)
println("median_allocs=", result.stats.median_allocs)
println("median_bytes=", result.stats.median_bytes)
