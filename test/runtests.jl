using AdaptiveOpticsProperHIL
using AdaptiveOpticsSim
using Test

@testset "AdaptiveOpticsProperHIL" begin
    @testset "OPD convention cases" begin
        results = validate_conventions(T=Float64, n=32, atol=1e-12)
        @test keys(results) == Set([:piston, :x_tilt, :y_tilt, :focus, :asymmetric, :single_poke])
        @test all(r -> r.max_abs <= 1e-12, values(results))
    end

    @testset "CPU HIL seam smoke" begin
        ctx = build_hil_context(resolution=64, n_act=8, T=Float32)
        psf, sampling = hil_step!(ctx)
        @test size(psf) == (64, 64)
        @test sampling > 0
        @test length(ctx.command_buffer) == 66
        for _ in 1:3
            hil_step!(ctx)
        end
        @test (@allocated hil_step!(ctx)) == 0
    end

    @testset "Versioned benchmark matrix" begin
        matrix = load_benchmark_matrix()
        @test matrix["suite_version"] == "1.0.0"
        @test Set(matrix["axes"]["backends"]) == Set(["cpu", "amdgpu"])
        @test Set(matrix["axes"]["detectors"]) == Set(["ccd", "cmos", "emccd", "saphira"])
        @test Set(matrix["axes"]["wfs"]) == Set(["shack_hartmann", "pyramid"])
        @test length(matrix["cases"]) == 9
        @test length(unique(case["id"] for case in matrix["cases"])) == 9

        defaults = matrix["defaults"]
        for case in matrix["cases"]
            case["kind"] == "wfs_detector" || continue
            context = build_benchmark_case(case, CPUBackend(); defaults=defaults)
            output = benchmark_case_step!(context)
            @test ndims(output) == 2
            @test all(isfinite, output)
            @test sum(output) > 0
        end

        stats = Dict{String,Any}(
            "median_ns" => 100.0,
            "p95_ns" => 120.0,
            "median_allocs" => 0.0,
            "median_bytes" => 0.0,
        )
        case = Dict{String,Any}("id" => "one", "stats" => stats)
        baseline = Dict{String,Any}(
            "suite_id" => matrix["suite_id"],
            "suite_version" => matrix["suite_version"],
            "backend" => "cpu",
            "cases" => [case],
        )
        equal_candidate = deepcopy(baseline)
        @test compare_benchmark_results(equal_candidate, baseline, matrix)["passed"]
        slow_candidate = deepcopy(baseline)
        slow_candidate["cases"][1]["stats"]["median_ns"] = 200.0
        comparison = compare_benchmark_results(slow_candidate, baseline, matrix)
        @test !comparison["passed"]
        @test "median_ns" in comparison["comparisons"][1]["failures"]
    end
end
