using AdaptiveOpticsProperHIL
using AdaptiveOpticsSim
using Proper
using Test

using AdaptiveOpticsSim.AlgorithmGraphs

include("algorithm_graph_fixture.jl")

function run_proper_graph_gpu_tests(BackendArray, synchronize_gpu)
    n = 16
    host_target = AdaptiveOpticsSim.Backends.HostComputeDevice()
    accelerator_probe = BackendArray(zeros(Float32, 1))
    accelerator_target = AdaptiveOpticsSim.Backends.compute_device(
        accelerator_probe,
    )

    pupil_opd_host = reshape(Float32.(0:(n * n - 1)), n, n) .* 1.0f-9
    pupil_amplitude_host =
        reshape(Float32.(1:(n * n)), n, n) ./ Float32(n * n)
    configuration = ProperPropagationConfiguration(
        pupil_intensity_prescription;
        resolution=n,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
    )

    host_graph = prepare_algorithm_graph(
        proper_test_graph(
            configuration,
            pupil_opd_host,
            pupil_amplitude_host,
        );
        target=host_target,
    )
    step_graph!(host_graph)
    host_reference = copy(graph_output(host_graph, :intensity))

    pupil_opd = BackendArray(pupil_opd_host)
    pupil_amplitude = BackendArray(pupil_amplitude_host)
    graph = prepare_algorithm_graph(
        proper_test_graph(configuration, pupil_opd, pupil_amplitude);
        target=accelerator_target,
    )
    owner = AlgorithmGraphs.prepared_graph_node(graph, :proper)
    @test owner.pupil_opd === pupil_opd
    @test owner.pupil_amplitude === pupil_amplitude
    @test AdaptiveOpticsSim.Backends.compute_device(graph) == accelerator_target
    @test AdaptiveOpticsSim.Backends.compute_device(owner.workspace.field) ==
        accelerator_target
    @test step_graph!(graph) === graph

    accelerator_output = graph_output(graph, :intensity)
    synchronize_gpu()
    @test owner.output === accelerator_output
    @test accelerator_output isa BackendArray
    @test AdaptiveOpticsSim.Backends.compute_device(accelerator_output) ==
        accelerator_target
    @test Array(accelerator_output) ≈ host_reference rtol=3.0f-4 atol=3.0f-6

    host_publication = Matrix{Float32}(undef, n, n)
    copyto!(host_publication, accelerator_output)
    synchronize_gpu()
    @test host_publication ≈ host_reference rtol=3.0f-4 atol=3.0f-6
    @test AdaptiveOpticsSim.Backends.compute_device(accelerator_output) ==
        accelerator_target
    @test reset_graph!(graph) === graph

    host_input_definition = proper_test_graph(
        configuration,
        pupil_opd_host,
        pupil_amplitude,
    )
    @test_throws AlgorithmGraphError prepare_algorithm_graph(
        host_input_definition;
        target=accelerator_target,
    )
    return nothing
end
