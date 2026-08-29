using AdaptiveOpticsProperHIL
using AdaptiveOpticsSim
using CalculonAlgorithms
using Proper
using Test

using AdaptiveOpticsSim.AlgorithmGraphs

include("calculon_fixture.jl")

function run_calculon_proper_gpu_tests(BackendArray, synchronize_gpu)
    n = 16
    host_target = AdaptiveOpticsSim.Backends.HostComputeDevice()
    accelerator_probe = BackendArray(zeros(Float32, 1))
    accelerator_target =
        AdaptiveOpticsSim.Backends.compute_device(accelerator_probe)

    pupil_opd_host = reshape(Float32.(0:(n * n - 1)), n, n) .* 1.0f-9
    pupil_amplitude_host =
        reshape(Float32.(1:(n * n)), n, n) ./ Float32(n * n)

    host_configuration = ProperPropagationConfiguration(
        pupil_intensity_prescription;
        target=host_target,
        resolution=n,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
    )
    host_declaration = proper_propagation_declaration(host_configuration)
    host_algorithm = prepare_algorithm(host_declaration, host_configuration)
    host_reference = zeros(Float32, n, n)
    CalculonAlgorithms.process!(
        host_reference,
        host_algorithm,
        pupil_opd_host,
        pupil_amplitude_host,
    )

    configuration = ProperPropagationConfiguration(
        pupil_intensity_prescription;
        target=accelerator_target,
        resolution=n,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
    )
    declaration = proper_propagation_declaration(configuration)
    pupil_opd = BackendArray(pupil_opd_host)
    pupil_amplitude = BackendArray(pupil_amplitude_host)
    direct_output = AdaptiveOpticsSim.Backends.allocate_device_array(
        accelerator_target,
        Float32,
        n,
        n,
    )

    direct = prepare_algorithm(declaration, configuration)
    direct_workspace = direct.workspace.workspace
    @test AdaptiveOpticsSim.Backends.compute_device(
        direct_workspace.invocation.field,
    ) == accelerator_target
    @test AdaptiveOpticsSim.Backends.compute_device(
        direct_workspace.pupil_opd,
    ) == accelerator_target
    @test AdaptiveOpticsSim.Backends.compute_device(
        direct_workspace.pupil_amplitude,
    ) == accelerator_target
    @test AdaptiveOpticsSim.Backends.compute_device(
        direct_workspace.output,
    ) == accelerator_target
    @test CalculonAlgorithms.process!(
        direct_output,
        direct,
        pupil_opd,
        pupil_amplitude,
    ) === direct_output
    synchronize_gpu()
    @test AdaptiveOpticsSim.Backends.compute_device(direct_output) ==
        accelerator_target
    @test Array(direct_output) ≈ host_reference rtol=3.0f-4 atol=3.0f-6

    definition = algorithm_graph(
        (algorithm_node(:proper, declaration, configuration),);
        inputs=(
            graph_input(:pupil_opd, :proper => :pupil_opd, pupil_opd),
            graph_input(
                :pupil_amplitude,
                :proper => :pupil_amplitude,
                pupil_amplitude,
            ),
        ),
        outputs=(graph_output(:intensity, :proper => :output),),
    )
    graph = prepare_algorithm_graph(definition; target=accelerator_target)
    bound = AlgorithmGraphs.prepared_algorithm(graph, Val(:proper))
    @test bound.pupil_opd === pupil_opd
    @test bound.pupil_amplitude === pupil_amplitude
    @test AdaptiveOpticsSim.Backends.compute_device(graph) ==
        accelerator_target
    @test step_graph!(graph) === graph

    accelerator_output = graph_output(graph, Val(:intensity))
    synchronize_gpu()
    @test bound.output === accelerator_output
    @test AdaptiveOpticsSim.Backends.compute_device(bound.invocation.field) ==
        accelerator_target
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

    host_input_definition = algorithm_graph(
        (algorithm_node(:proper, declaration, configuration),);
        inputs=(
            graph_input(
                :pupil_opd,
                :proper => :pupil_opd,
                pupil_opd_host,
            ),
            graph_input(
                :pupil_amplitude,
                :proper => :pupil_amplitude,
                pupil_amplitude,
            ),
        ),
    )
    @test_throws AlgorithmGraphError prepare_algorithm_graph(
        host_input_definition;
        target=accelerator_target,
    )

    return nothing
end
