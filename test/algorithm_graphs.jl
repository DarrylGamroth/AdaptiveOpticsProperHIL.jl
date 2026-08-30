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

    llowfs_configuration = ProperPropagationConfiguration(
        rectangular_pupil_intensity_prescription;
        resolution=8,
        output_rows=6,
        output_columns=5,
        diameter_m=8.0f0,
        wavelength_um=1.55f0,
        output_schema="test.spiders.llowfs-observation/1",
    )
    scc_configuration = ProperPropagationConfiguration(
        rectangular_pupil_intensity_prescription;
        resolution=8,
        output_rows=4,
        output_columns=7,
        diameter_m=8.0f0,
        wavelength_um=1.55f0,
        output_schema="test.spiders.scc-observation/1",
    )
    spiders_node_types = merge(
        builtin_graph_node_types(),
        (
            spiders_llowfs_proper=
                proper_propagation_node_factory(llowfs_configuration),
            spiders_scc_proper=
                proper_propagation_node_factory(scc_configuration),
        ),
    )
    spiders_bindings = (
        pupil_opd=pupil_opd,
        pupil_amplitude=pupil_amplitude,
    )
    llowfs_graph = prepare_algorithm_graph(
        load_algorithm_graph(
            joinpath(
                dirname(@__DIR__),
                "examples",
                "graphs",
                "spiders_llowfs_hil.toml",
            );
            node_types=spiders_node_types,
            bindings=spiders_bindings,
        );
        target=target,
    )
    scc_graph = prepare_algorithm_graph(
        load_algorithm_graph(
            joinpath(
                dirname(@__DIR__),
                "examples",
                "graphs",
                "spiders_scc_hil.toml",
            );
            node_types=spiders_node_types,
            bindings=spiders_bindings,
        );
        target=target,
    )
    @test graph_name(llowfs_graph) === :spiders_llowfs_hil
    @test graph_name(scc_graph) === :spiders_scc_hil
    @test step_graph!(llowfs_graph) === llowfs_graph
    @test step_graph!(scc_graph) === scc_graph
    @test size(graph_output(llowfs_graph, :llowfs_observation)) == (6, 5)
    @test size(graph_output(scc_graph, :scc_observation)) == (4, 7)
    @test @allocated(step_graph!(llowfs_graph)) == 0
    @test @allocated(step_graph!(scc_graph)) == 0
    @test_throws AlgorithmGraphError begin
        spiders_node_types.spiders_llowfs_proper(
            :llowfs,
            (resolution=8,),
            NamedTuple(),
        )
    end

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
