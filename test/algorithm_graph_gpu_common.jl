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

    rectangular_configuration = ProperPropagationConfiguration(
        rectangular_pupil_intensity_prescription;
        resolution=n,
        output_rows=n - 2,
        output_columns=n - 3,
        diameter_m=8.0f0,
        wavelength_um=1.65f0,
    )
    host_rectangular_graph = prepare_algorithm_graph(
        proper_test_graph(
            rectangular_configuration,
            pupil_opd_host,
            pupil_amplitude_host,
        );
        target=host_target,
    )
    step_graph!(host_rectangular_graph)
    rectangular_reference =
        copy(graph_output(host_rectangular_graph, :intensity))
    rectangular_graph = prepare_algorithm_graph(
        proper_test_graph(
            rectangular_configuration,
            pupil_opd,
            pupil_amplitude,
        );
        target=accelerator_target,
    )
    @test step_graph!(rectangular_graph) === rectangular_graph
    rectangular_output = graph_output(rectangular_graph, :intensity)
    synchronize_gpu()
    @test size(rectangular_output) == (n - 2, n - 3)
    @test rectangular_output isa BackendArray
    @test AdaptiveOpticsSim.Backends.compute_device(rectangular_output) ==
        accelerator_target
    @test Array(rectangular_output) ≈ rectangular_reference rtol=3.0f-4 atol=3.0f-6

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

    spiders_profile = provisional_spiders_h_regular_profile(Float32)
    spiders_configurations = (
        provisional_spiders_llowfs_configuration(spiders_profile),
        provisional_spiders_scc_configuration(spiders_profile),
    )
    for spiders_configuration in spiders_configurations
        pupil_n = spiders_configuration.pupil_resolution
        spiders_opd_host = zeros(Float32, pupil_n, pupil_n)
        center = Float32(pupil_n + 1) / 2f0
        @inbounds for column in axes(spiders_opd_host, 2)
            value = 2f-7 * (Float32(column) - center) / Float32(pupil_n)
            for row in axes(spiders_opd_host, 1)
                spiders_opd_host[row, column] = value
            end
        end
        spiders_amplitude_host = ones(Float32, pupil_n, pupil_n)
        spiders_host_graph = prepare_algorithm_graph(
            proper_test_graph(
                spiders_configuration,
                spiders_opd_host,
                spiders_amplitude_host,
            );
            target=host_target,
        )
        step_graph!(spiders_host_graph)
        spiders_reference = copy(graph_output(
            spiders_host_graph,
            :intensity,
        ))

        spiders_opd = BackendArray(spiders_opd_host)
        spiders_amplitude = BackendArray(spiders_amplitude_host)
        spiders_graph = prepare_algorithm_graph(
            proper_test_graph(
                spiders_configuration,
                spiders_opd,
                spiders_amplitude,
            );
            target=accelerator_target,
        )
        @test step_graph!(spiders_graph) === spiders_graph
        spiders_output = graph_output(spiders_graph, :intensity)
        synchronize_gpu()
        @test spiders_output isa BackendArray
        @test AdaptiveOpticsSim.Backends.compute_device(spiders_output) ==
            accelerator_target
        @test size(spiders_output) == size(spiders_reference)
        @test all(isfinite, Array(spiders_output))
        @test Array(spiders_output) ≈ spiders_reference rtol=1f-3 atol=3f-6

        spiders_owner = AlgorithmGraphs.prepared_graph_node(
            spiders_graph,
            :proper,
        )
        for values in (
            spiders_owner.workspace.assets.apodizer_mask,
            spiders_owner.workspace.assets.focal_plane_mask_internal,
            spiders_owner.workspace.assets.lyot_mask,
            spiders_owner.workspace.assets.padded_pupil_opd,
            spiders_owner.workspace.assets.padded_pupil_amplitude,
            spiders_owner.workspace.assets.square_intensity,
            spiders_owner.workspace.assets.rotated_intensity,
            spiders_owner.workspace.assets.centered_field,
        )
            @test AdaptiveOpticsSim.Backends.compute_device(values) ==
                accelerator_target
        end
    end
    return nothing
end
