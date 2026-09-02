using AdaptiveOpticsProperHIL
using AdaptiveOpticsSim
using Proper
using Test

using AdaptiveOpticsSim.AlgorithmGraphs

include("algorithm_graph_fixture.jl")

@testset "AOS-native Proper propagation node" begin
    target = AdaptiveOpticsSim.Backends.HostComputeDevice()
    configuration = ProperPropagationConfiguration(
        pupil_intensity_prescription;
        resolution=8,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
    )
    pupil_opd = reshape(Float32.(0:63), 8, 8) .* 1.0f-9
    pupil_amplitude = reshape(Float32.(1:64), 8, 8) ./ 64.0f0

    graph = prepare_algorithm_graph(
        proper_test_graph(configuration, pupil_opd, pupil_amplitude);
        target=target,
    )
    owner = AlgorithmGraphs.prepared_graph_node(graph, :proper)
    @test AlgorithmGraphs.graph_node_capture_capability(owner) isa
        AlgorithmGraphs.GraphNodeCaptureSafe
    @test graph_name(graph) === :proper_test
    @test AdaptiveOpticsSim.Backends.compute_device(graph) == target
    @test owner.pupil_opd === pupil_opd
    @test owner.pupil_amplitude === pupil_amplitude
    @test owner.state.rng === owner.workspace.context.rng
    @test step_graph!(graph) === graph
    output = graph_output(graph, :intensity)
    @test owner.output === output
    @test sum(output) ≈ sum(abs2, pupil_amplitude)
    @test @allocated(step_graph!(graph)) == 0
    @test @inferred(step_graph!(graph)) === graph
    @test reset_graph!(graph) === graph
    @test @allocated(reset_graph!(graph)) == 0

    node_types = merge(
        builtin_graph_node_types(),
        (
            proper_propagation_f32=proper_propagation_node_factory(
                pupil_intensity_prescription;
                T=Float32,
            ),
        ),
    )
    from_file = prepare_algorithm_graph(
        load_algorithm_graph(
            joinpath(@__DIR__, "graph_files", "proper_propagation.toml");
            node_types=node_types,
            bindings=(
                pupil_opd=pupil_opd,
                pupil_amplitude=pupil_amplitude,
            ),
        );
        target=target,
    )
    @test graph_name(from_file) === :proper_propagation
    @test step_graph!(from_file) === from_file
    @test graph_output(from_file, :intensity) == output

    rectangular_configuration = ProperPropagationConfiguration(
        rectangular_pupil_intensity_prescription;
        resolution=8,
        output_rows=6,
        output_columns=5,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
        output_schema="test.rectangular-intensity/1",
    )
    rectangular_graph = prepare_algorithm_graph(
        proper_test_graph(
            rectangular_configuration,
            pupil_opd,
            pupil_amplitude,
        );
        target=target,
    )
    @test step_graph!(rectangular_graph) === rectangular_graph
    rectangular_output = graph_output(rectangular_graph, :intensity)
    @test size(rectangular_output) == (6, 5)
    rectangular_expected = similar(rectangular_output)
    Proper.prop_magnify!(
        rectangular_expected,
        output,
        1.0f0,
        Proper.RunContext(typeof(output)),
    )
    @test rectangular_output ≈ rectangular_expected
    @test @allocated(step_graph!(rectangular_graph)) == 0
    @test @inferred(step_graph!(rectangular_graph)) === rectangular_graph

    padded_configuration = ProperPropagationConfiguration(
        padded_pupil_intensity_prescription;
        resolution=8,
        pupil_resolution=4,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
    )
    padded_opd = reshape(Float32.(0:15), 4, 4) .* 1.0f-9
    padded_amplitude = ones(Float32, 4, 4)
    padded_graph = prepare_algorithm_graph(
        proper_test_graph(
            padded_configuration,
            padded_opd,
            padded_amplitude,
        );
        target=target,
    )
    @test padded_configuration.pupil_resolution == 4
    @test step_graph!(padded_graph) === padded_graph
    @test size(graph_output(padded_graph, :intensity)) == (8, 8)
    @test all(isfinite, graph_output(padded_graph, :intensity))
    @test @allocated(step_graph!(padded_graph)) == 0
    @test @inferred(step_graph!(padded_graph)) === padded_graph

    wrapped_opd = @view pupil_opd[:, :]
    bad_definition = proper_test_graph(
        configuration,
        wrapped_opd,
        pupil_amplitude,
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
        pupil_intensity_prescription;
        resolution=8,
        pupil_resolution=0,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
    )
    @test_throws ArgumentError ProperPropagationConfiguration(
        pupil_intensity_prescription;
        resolution=8,
        output_rows=0,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
    )
    @test_throws ArgumentError ProperPropagationConfiguration(
        :pupil_intensity_prescription;
        resolution=8,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
    )
    @test_throws AlgorithmGraphError node_types.proper_propagation_f32(
        :proper,
        (diameter_m=8.0, resolution=8, wavelength_um=1.65, extra=true),
        NamedTuple(),
    )
end
