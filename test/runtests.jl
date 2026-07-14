using AdaptiveOpticsProperHIL
using AdaptiveOpticsSim
using Base64
using HdrHistogram
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
        transaction_count = (2 + 1 + 1_000) + (1 + 1_000)
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
                samples=1_000,
                trials=1,
            )
            @test result.samples == 1_000
            @test result.trials == 1
            @test result.cycle.samples == 1_000
            @test result.transport.samples == 1_000
            @test result.cycle.p50_ns <= result.cycle.p99_ns <=
                result.cycle.max_ns
            @test result.transport.p50_ns <= result.transport.p99_ns <=
                result.transport.max_ns
            @test result.cycle.p50_ns <= result.cycle.p90_ns <=
                result.cycle.p95_ns <= result.cycle.p99_ns
            @test isnothing(result.cycle.p99_9_ns)
            @test result.deadline_misses == 0
            @test result.allocated_bytes_per_cycle >= 0
            @test result.retain_raw_samples
            @test length(only(result.cycle_latencies_ns)) == 1_000
            @test length(result.cycle_histograms) == 1
            artifact = only(result.cycle_histograms)
            @test artifact.samples == 1_000
            @test artifact.source_revision ==
                "cd42987e0739c06d29d20b4341f088b85e7a620c"
            decoded = HdrHistogram.decode_from_compressed_byte_buffer(
                Base64.base64decode(artifact.payload_base64))
            @test HdrHistogram.total_count(decoded) == 1_000
            @test HdrHistogram.value_at_percentile(decoded, 99.0) ==
                result.cycle.p99_ns

            compact_result = benchmark_external_rtc!(harness;
                warmup=0,
                samples=1_000,
                trials=1,
                retain_raw_samples=false,
            )
            @test !compact_result.retain_raw_samples
            @test isnothing(compact_result.cycle_latencies_ns)
            @test isnothing(compact_result.transport_latencies_ns)
            @test only(compact_result.cycle_histograms).samples == 1_000
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

    @testset "Latency histogram contract" begin
        params = AdaptiveOpticsProperHIL.LatencyHistogramParams(
            highest_trackable_ns=1_000_000,
            significant_figures=3,
        )
        state = AdaptiveOpticsProperHIL.LatencyHistogramState(params)
        AdaptiveOpticsProperHIL.record_latency!(state, 10)
        @test (@allocated AdaptiveOpticsProperHIL.record_latency!(state, 11)) == 0
        for value in 12:1_009
            AdaptiveOpticsProperHIL.record_latency!(state, value)
        end
        summary = AdaptiveOpticsProperHIL.latency_summary(state)
        @test summary.samples == 1_000
        @test summary.p50_ns <= summary.p95_ns <= summary.p99_ns <= summary.max_ns
        @test isnothing(summary.p99_9_ns)

        other = AdaptiveOpticsProperHIL.LatencyHistogramState(params)
        for value in 1:1_000
            AdaptiveOpticsProperHIL.record_latency!(other, value)
        end
        AdaptiveOpticsProperHIL.merge_latency!(state, other)
        @test HdrHistogram.total_count(state.histogram) == 2_000
        @test_throws ArgumentError AdaptiveOpticsProperHIL.record_latency!(state, 1_000_001)
        @test_throws ArgumentError AdaptiveOpticsProperHIL.latency_summary(
            AdaptiveOpticsProperHIL.LatencyHistogramState(params))
        @test_throws ArgumentError AdaptiveOpticsProperHIL.LatencyHistogramParams(
            highest_trackable_ns=1)

        long_state = AdaptiveOpticsProperHIL.LatencyHistogramState(params)
        for value in 1:10_000
            AdaptiveOpticsProperHIL.record_latency!(long_state, value)
        end
        @test !isnothing(AdaptiveOpticsProperHIL.latency_summary(long_state).p99_9_ns)
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
