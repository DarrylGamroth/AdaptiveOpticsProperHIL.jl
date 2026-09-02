using AdaptiveOpticsProperHIL
using CUDA
using LinearAlgebra: norm
using Proper
using SPIDERSSim.ProperOptics
using Statistics

CUDA.functional() || error("CUDA is not functional on this host")
CUDA.allowscalar(false)

percentile_ms(values_ns, percentile) =
    round(quantile(values_ns, percentile) / 1e6; digits=3)

to_device(::Nothing) = nothing
to_device(array::AbstractArray) = CuArray(array)

function to_device(entrance::SPIDERSSim.ProperOptics.StaticPropagationEntrance)
    return SPIDERSSim.ProperOptics.StaticPropagationEntrance(
        to_device(entrance.field),
    )
end

function to_device(entrance::SPIDERSSim.ProperOptics.ResidualPropagationEntrance)
    return SPIDERSSim.ProperOptics.ResidualPropagationEntrance(
        to_device(entrance.field),
        to_device(entrance.residual_nm),
        entrance.sampling_m,
    )
end

function to_device(entrance::SPIDERSSim.ProperOptics.ExternalPropagationEntrance)
    return SPIDERSSim.ProperOptics.ExternalPropagationEntrance(
        to_device(entrance.field),
        to_device(entrance.pupil_amplitude),
        to_device(entrance.opd_m),
        entrance.pupil_sampling_m,
        entrance.opd_sampling_m,
    )
end

function to_device(values::NamedTuple)
    pairs = (name => to_device(getfield(values, name)) for name in keys(values))
    return (; pairs...)
end

function to_device(prepared::PreparedLLOWFSPropagation)
    entrance = to_device(prepared.entrance)
    field = similar(entrance.field)
    context = Proper.RunContext(typeof(field))
    wavefront = prop_begin!(
        field,
        prepared.configuration.telescope_diameter_m,
        prop_get_wavelength(prepared.wavefront);
        beam_diam_fraction=prepared.configuration.beam_diameter_fraction,
        context,
    )
    device_prepared = PreparedLLOWFSPropagation(
        prepared.configuration,
        prepared.relay_configuration,
        entrance,
        context,
        wavefront,
        to_device(prepared.apertures),
        to_device(prepared.apodizer_internal),
        to_device(prepared.fpm_opd_internal),
        to_device(prepared.lyot_reflection_internal),
        to_device(prepared.resample_scratch),
        to_device(prepared.output),
        prepared.refractive_indices,
    )
    propagate_llowfs!(device_prepared)
    CUDA.synchronize()
    return device_prepared
end

function measure(model, samples::Int)
    latencies_ns = Vector{UInt64}(undef, samples)
    for index in eachindex(latencies_ns)
        started_ns = time_ns()
        propagate_optics!(model)
        CUDA.synchronize()
        latencies_ns[index] = time_ns() - started_ns
    end
    return latencies_ns
end

function summarize(name, values_ns)
    println(name, ':')
    println("  samples: ", length(values_ns))
    println("  min_ms: ", round(minimum(values_ns) / 1e6; digits=3))
    println("  median_ms: ", percentile_ms(values_ns, 0.50))
    println("  p90_ms: ", percentile_ms(values_ns, 0.90))
    println("  p95_ms: ", percentile_ms(values_ns, 0.95))
    length(values_ns) >= 100 &&
        println("  p99_ms: ", percentile_ms(values_ns, 0.99))
    println("  max_ms: ", round(maximum(values_ns) / 1e6; digits=3))
    println("  mean_frames_per_s: ", round(1e9 / mean(values_ns); digits=3))
    return nothing
end

function relative_l2(device_output, cpu_output)
    gpu_output = Array(device_output)
    return norm(gpu_output .- cpu_output) / norm(cpu_output)
end

function check_changed_input!(cpu_prepared, device_prepared, captured_model)
    fill!(cpu_prepared.entrance.opd_m, 0f0)
    flat_reference = copy(propagate_llowfs!(cpu_prepared))
    fill!(device_prepared.entrance.opd_m, 0f0)
    propagate_optics!(captured_model)
    CUDA.synchronize()
    flat_relative_l2 = relative_l2(device_prepared.output, flat_reference)

    fill!(cpu_prepared.entrance.opd_m, 10f-9)
    changed_reference = copy(propagate_llowfs!(cpu_prepared))
    fill!(device_prepared.entrance.opd_m, 10f-9)
    propagate_optics!(captured_model)
    CUDA.synchronize()
    changed_relative_l2 = relative_l2(device_prepared.output, changed_reference)

    fill!(cpu_prepared.entrance.opd_m, 0f0)
    fill!(device_prepared.entrance.opd_m, 0f0)
    return (; flat_relative_l2, changed_relative_l2)
end

function main(args)
    gridsize = length(args) >= 1 ? parse(Int, args[1]) : 1024
    samples = length(args) >= 2 ? parse(Int, args[2]) : 100
    beam_diameter_fraction = length(args) >= 3 ?
        parse(Float64, args[3]) : 0.1141323837216534

    configuration = SCCPropagationConfiguration(
        reference_pinhole=true,
        beam_diameter_fraction=beam_diameter_fraction,
    )
    cpu_prepared = prepare_llowfs_propagation(
        1.55e-6,
        gridsize;
        configuration,
        pupil_amplitude=ones(Float32, 128, 128),
        pupil_sampling_m=7.92 / 128,
        opd_m=zeros(Float32, 128, 128),
        T=Float32,
    )
    device_prepared = to_device(cpu_prepared)
    sampling_m = prop_get_sampling(device_prepared.wavefront)
    direct_model = PreparedOpticalIntensityModel(
        device_prepared,
        propagate_llowfs!,
        llowfs_intensity,
        sampling_m,
    )
    captured_model = capture_cuda_optical_model(direct_model; warmup=10)
    correctness = check_changed_input!(
        cpu_prepared,
        device_prepared,
        captured_model,
    )

    for _ in 1:10
        propagate_optics!(direct_model)
    end
    CUDA.synchronize()
    direct_before = measure(direct_model, samples)
    graph_latencies = measure(captured_model, samples)
    direct_after = measure(direct_model, samples)

    direct_host_bytes = @allocated begin
        propagate_optics!(direct_model)
        CUDA.synchronize()
    end
    graph_launch_host_bytes = @allocated propagate_optics!(captured_model)
    CUDA.synchronize()
    graph_synchronized_host_bytes = @allocated begin
        propagate_optics!(captured_model)
        CUDA.synchronize()
    end

    println("HIL SPIDERS LLOWFS CUDA Graph benchmark")
    println("policy: preparation, compilation, capture, instantiation, and warmup are outside samples")
    println("boundary: one synchronized PROPER propagation; detector, camera, DM, and transfers are excluded")
    println("load_model: closed-loop single propagation")
    println("environment:")
    println("  julia_version: ", VERSION)
    println("  julia_threads: ", Threads.nthreads())
    println("  cuda_runtime: ", CUDA.runtime_version())
    println("  cuda_driver: ", CUDA.driver_version())
    println("  device: ", CUDA.name(CUDA.device()))
    println("parameters:")
    println("  wavelength_m: 1.55e-6")
    println("  gridsize: ", gridsize)
    println("  beam_diameter_fraction: ", beam_diameter_fraction)
    println("  input_mode: external_pupil_and_opd")
    println("  precision: Float32")
    println("  samples_per_phase: ", samples)
    println("correctness:")
    println("  flat_relative_l2_vs_cpu: ", correctness.flat_relative_l2)
    println("  changed_input_relative_l2_vs_cpu: ", correctness.changed_relative_l2)
    println("allocations:")
    println("  direct_synchronized_host_bytes: ", direct_host_bytes)
    println("  graph_launch_host_bytes: ", graph_launch_host_bytes)
    println("  graph_synchronized_host_bytes: ", graph_synchronized_host_bytes)
    summarize("direct_before", direct_before)
    summarize("graph", graph_latencies)
    summarize("direct_after", direct_after)
    println("raw_direct_before_ns: ", join(direct_before, ','))
    println("raw_graph_ns: ", join(graph_latencies, ','))
    println("raw_direct_after_ns: ", join(direct_after, ','))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
