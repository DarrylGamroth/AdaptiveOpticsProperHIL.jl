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

function rectangular_pupil_intensity_prescription(
    wavelength_m,
    resolution;
    pupil_opd,
    pupil_amplitude,
    diameter_m,
    field,
    wavefront,
    output,
    run_context,
    square_intensity,
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
    _, sampling_m = Proper.prop_end(wavefront, square_intensity)
    Proper.prop_magnify!(
        output,
        square_intensity,
        one(wavelength_m),
        run_context,
    )
    return output, sampling_m
end

function padded_pupil_intensity_prescription(
    wavelength_m,
    resolution;
    pupil_opd,
    pupil_amplitude,
    diameter_m,
    field,
    wavefront,
    output,
    run_context,
    padded_opd,
    padded_amplitude,
)
    size(field) == (resolution, resolution) || error("field binding changed")
    Proper.prop_magnify!(
        padded_opd,
        pupil_opd,
        one(wavelength_m),
        run_context,
    )
    Proper.prop_magnify!(
        padded_amplitude,
        pupil_amplitude,
        one(wavelength_m),
        run_context,
    )
    Proper.prop_begin!(
        wavefront,
        diameter_m,
        wavelength_m;
        beam_diam_fraction=size(pupil_opd, 1) / resolution,
    )
    Proper.prop_multiply(wavefront, padded_amplitude)
    Proper.prop_add_phase(wavefront, padded_opd)
    return Proper.prop_end(wavefront, output)
end

function AdaptiveOpticsProperHIL.prepare_proper_assets(
    ::typeof(padded_pupil_intensity_prescription),
    target,
    ::Type{T},
    resolution::Int,
    output_shape::Tuple{Int,Int},
) where {T}
    output_shape == (resolution, resolution) || error(
        "padded test output must match the propagation grid",
    )
    padded_opd = AdaptiveOpticsSim.Backends.allocate_device_array(
        target,
        T,
        resolution,
        resolution,
    )
    padded_amplitude = similar(padded_opd)
    fill!(padded_opd, zero(T))
    fill!(padded_amplitude, zero(T))
    return (; padded_opd, padded_amplitude)
end

function AdaptiveOpticsProperHIL.prepare_proper_assets(
    ::typeof(rectangular_pupil_intensity_prescription),
    target,
    ::Type{T},
    resolution::Int,
    output_shape::Tuple{Int,Int},
) where {T}
    (
        output_shape[1] <= resolution &&
        output_shape[2] <= resolution
    ) || error("rectangular test output must fit inside the propagation grid")
    square_intensity =
        AdaptiveOpticsSim.Backends.allocate_device_array(
            target,
            T,
            resolution,
            resolution,
        )
    fill!(square_intensity, zero(T))
    return (; square_intensity)
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
