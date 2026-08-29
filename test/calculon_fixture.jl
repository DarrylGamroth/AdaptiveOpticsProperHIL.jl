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
