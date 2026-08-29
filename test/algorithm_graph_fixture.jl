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

function proper_test_graph(configuration, pupil_opd, pupil_amplitude)
    return algorithm_graph(
        (proper_propagation_node(:proper, configuration),);
        name=:proper_test,
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
end
