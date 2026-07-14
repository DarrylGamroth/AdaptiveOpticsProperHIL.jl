using AdaptiveOpticsProperHIL
using AdaptiveOpticsSim
using Sockets
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

    @testset "External RTC loopback contract" begin
        ctx = build_hil_context(resolution=32, n_act=4, T=Float32)
        server = listen(ip"127.0.0.1", 0)
        port = Int(last(getsockname(server)))
        slope_count = length(AdaptiveOpticsSim.slopes(ctx.scenario))
        command_count = length(ctx.command_host)
        transaction_count = 2 + 1 + 100
        server_task = @async begin
            client = accept(server)
            try
                serve_mock_rtc!(client;
                    slope_count=slope_count,
                    command_count=command_count,
                    T=Float32,
                    max_cycles=transaction_count,
                )
            finally
                close(client)
            end
        end
        harness = connect_external_rtc(ctx;
            port=port,
            deadline_ns=100_000_000,
            command_limit=1.0,
        )
        try
            external_rtc_step!(harness)
            @test harness.sequence == 1
            @test harness.cycles == 1
            @test all(isfinite, ctx.command_host)
            @test ctx.command_host ≈ [
                -0.1f0 * harness.slopes_host[mod1(i,
                    length(harness.slopes_host))]
                for i in eachindex(ctx.command_host)
            ]
            result = benchmark_external_rtc!(harness;
                warmup=1,
                samples=100,
                trials=1,
            )
            @test result.samples == 100
            @test result.trials == 1
            @test result.cycle.p50_ns <= result.cycle.p99_ns <=
                result.cycle.max_ns
            @test result.transport.p50_ns <= result.transport.p99_ns <=
                result.transport.max_ns
            @test result.deadline_misses == 0
            @test result.allocated_bytes_per_cycle >= 0
            @test rtc_cycle_metrics(harness).cycles == transaction_count
        finally
            close(harness)
            wait(server_task)
            close(server)
        end

        for T in (Float32, Float64)
            frame = AdaptiveOpticsProperHIL.RTCFrame(T, 2)
            input = T[1, 2]
            AdaptiveOpticsProperHIL.encode_rtc_frame!(frame,
                AdaptiveOpticsProperHIL.RTCCommands, UInt64(3), UInt64(4),
                input)
            output = zeros(T, 2)
            @test AdaptiveOpticsProperHIL.decode_rtc_frame!(output, frame,
                AdaptiveOpticsProperHIL.RTCCommands, UInt64(3)) == UInt64(4)
            @test output == input
            @test_throws RTCProtocolError begin
                AdaptiveOpticsProperHIL.decode_rtc_frame!(output, frame,
                    AdaptiveOpticsProperHIL.RTCCommands, UInt64(2))
            end
        end
        @test_throws RTCProtocolError begin
            AdaptiveOpticsProperHIL._validate_rtc_commands!(
                Float32[Inf], 1.0f0)
        end
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
