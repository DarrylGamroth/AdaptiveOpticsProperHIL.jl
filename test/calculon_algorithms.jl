using AdaptiveOpticsProperHIL
using AdaptiveOpticsSim
using CalculonAlgorithms
using Proper
using Test

using AdaptiveOpticsSim.AlgorithmGraphs

function pupil_intensity_prescription(
    wavelength_m,
    resolution;
    pupil_opd,
    pupil_amplitude,
    diameter_m,
    field,
    wavefront,
    output,
    run_context,
)
    size(field) == (resolution, resolution) || error("field binding changed")
    Proper.prop_begin!(
        wavefront,
        diameter_m,
        wavelength_m;
        beam_diam_fraction=one(wavelength_m),
    )
    Proper.prop_multiply(wavefront, pupil_amplitude)
    Proper.prop_add_phase(wavefront, pupil_opd)
    return Proper.prop_end(wavefront, output)
end

@testset "Calculon Proper propagation" begin
    target = AdaptiveOpticsSim.Backends.HostComputeDevice()
    configuration = ProperPropagationConfiguration(
        pupil_intensity_prescription;
        target=target,
        resolution=8,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
    )
    declaration = proper_propagation_declaration(configuration)

    @test algorithm_label(declaration) == "proper-propagation-f32"
    @test algorithm_property_policy(declaration) isa
        CalculonAlgorithms.NoDeclaredProperties
    @test length(algorithm_configuration(declaration)) == 9
    @test length(algorithm_ports(declaration)) == 3

    pupil_opd = reshape(Float32.(0:63), 8, 8) .* 1.0f-9
    pupil_amplitude = reshape(Float32.(1:64), 8, 8) ./ 64.0f0
    direct_output = zeros(Float32, 8, 8)
    direct = prepare_algorithm(declaration, configuration)
    @test direct.workspace.state.rng ===
        direct.workspace.workspace.invocation.context.rng
    @test CalculonAlgorithms.process!(
        direct_output,
        direct,
        pupil_opd,
        pupil_amplitude,
    ) === direct_output
    @test sum(direct_output) ≈ sum(abs2, pupil_amplitude)
    CalculonAlgorithms.process!(
        direct_output,
        direct,
        pupil_opd,
        pupil_amplitude,
    )
    @test @allocated(CalculonAlgorithms.process!(
        direct_output,
        direct,
        pupil_opd,
        pupil_amplitude,
    )) == 0
    @test CalculonAlgorithms.reset!(direct) === direct
    @test @allocated(CalculonAlgorithms.reset!(direct)) == 0

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
    graph = prepare_algorithm_graph(definition; target=target)
    @test AdaptiveOpticsSim.Backends.compute_device(graph) == target
    @test AlgorithmGraphs.prepared_algorithm(graph, Val(:proper)).pupil_opd ===
        pupil_opd
    @test AlgorithmGraphs.prepared_algorithm(
        graph,
        Val(:proper),
    ).pupil_amplitude ===
        pupil_amplitude
    @test step_graph!(graph) === graph
    @test graph_output(graph, Val(:intensity)) == direct_output
    @test @allocated(step_graph!(graph)) == 0
    @test @inferred(step_graph!(graph)) === graph
    @test reset_graph!(graph) === graph
    @test @allocated(reset_graph!(graph)) == 0

    wrapped_opd = @view pupil_opd[:, :]
    bad_definition = algorithm_graph(
        (algorithm_node(:proper, declaration, configuration),);
        inputs=(
            graph_input(:pupil_opd, :proper => :pupil_opd, wrapped_opd),
            graph_input(
                :pupil_amplitude,
                :proper => :pupil_amplitude,
                pupil_amplitude,
            ),
        ),
    )
    @test_throws AlgorithmGraphError prepare_algorithm_graph(
        bad_definition;
        target=target,
    )

    @test_throws ArgumentError ProperPropagationConfiguration(
        pupil_intensity_prescription;
        resolution=0,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
    )
    @test_throws ArgumentError ProperPropagationConfiguration(
        :pupil_intensity_prescription;
        resolution=8,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
    )
end
