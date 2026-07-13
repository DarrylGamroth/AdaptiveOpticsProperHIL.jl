backend = length(ARGS) >= 1 ? lowercase(ARGS[1]) : "cpu"
mode = length(ARGS) >= 2 ? lowercase(ARGS[2]) : "run"
samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing
seconds = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : nothing
warmup = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing

if backend == "amdgpu"
    import AMDGPU
    AMDGPU.allowscalar(false)
end

using AdaptiveOpticsProperHIL
using TOML

mode in ("run", "record", "check") ||
    throw(ArgumentError("mode must be run, record, or check"))

matrix_path = DEFAULT_BENCHMARK_MATRIX
matrix = load_benchmark_matrix(matrix_path)
result = run_benchmark_matrix(
    backend;
    matrix_path=matrix_path,
    samples=samples,
    seconds=seconds,
    warmup=warmup,
)

for case in result["cases"]
    stats = case["stats"]
    println(
        rpad(case["id"], 28),
        " median_ms=", round(stats["median_ns"] / 1e6; digits=3),
        " p95_ms=", round(stats["p95_ns"] / 1e6; digits=3),
        " allocs=", Int(stats["median_allocs"]),
        " bytes=", Int(stats["median_bytes"]),
    )
end

if mode == "record"
    path = benchmark_baseline_path(matrix, backend; matrix_path=matrix_path)
    write_benchmark_result(path, result)
    println("wrote baseline ", path)
elseif mode == "check"
    path = benchmark_baseline_path(matrix, backend; matrix_path=matrix_path)
    isfile(path) || throw(ArgumentError("baseline does not exist: $(path)"))
    comparison = compare_benchmark_results(result, TOML.parsefile(path), matrix)
    for case in comparison["comparisons"]
        case["passed"] && continue
        println("regression ", case["id"], ": ", join(case["failures"], ", "))
    end
    comparison["passed"] || error("benchmark regression thresholds exceeded")
    println("all benchmark regression thresholds passed")
end
