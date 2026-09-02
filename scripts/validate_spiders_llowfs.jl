using AdaptiveOpticsProperHIL
using AdaptiveOpticsSim
using Proper
using SPIDERSSim.ProperOptics

const FULL_SENSOR_X = 320
const FULL_SENSOR_Y = 256
const ROI_X = 34
const ROI_Y = 47
const PIXEL_PITCH_M = 15e-6
const NATIVE_SAMPLING_BEAM_PRODUCT = 116.87156093097308

centered_range(parent::Int, child::Int) = begin
    start = fld(parent - child, 2) + 1
    start:(start + child - 1)
end

function roi_range(start::Int, length::Int, parent::Int, label::AbstractString)
    range = start:(start + length - 1)
    first(range) >= 1 && last(range) <= parent || error(
        "$label $range lies outside 1:$parent",
    )
    return range
end

function source_range(full_source::UnitRange{Int}, roi::UnitRange{Int})
    start = first(full_source) + first(roi) - 1
    return start:(start + length(roi) - 1)
end

function build_spiders_llowfs_hil(
    gridsize::Int,
    roi_mode::Symbol,
    roi_x::UnitRange{Int},
    roi_y::UnitRange{Int},
)
    gridsize >= FULL_SENSOR_X || error(
        "gridsize must be at least the 320-pixel GoldEye x dimension",
    )
    beam_diameter_fraction = NATIVE_SAMPLING_BEAM_PRODUCT / gridsize
    residual_nm = zeros(Float32, 128, 128)
    propagation_configuration = SCCPropagationConfiguration(
        reference_pinhole=true,
        beam_diameter_fraction=beam_diameter_fraction,
    )
    llowfs = prepare_llowfs_propagation(
        1.55e-6,
        gridsize;
        configuration=propagation_configuration,
        ao_residual_nm=residual_nm,
        T=Float32,
    )
    sampling_m = prop_get_sampling(llowfs.wavefront)
    isapprox(sampling_m, PIXEL_PITCH_M; rtol=2e-4) || error(
        "PROPER output sampling $sampling_m m is not the 15 um GoldEye pitch",
    )
    optical_model = PreparedOpticalIntensityModel(
        llowfs,
        propagate_llowfs!,
        llowfs_intensity,
        sampling_m,
    )

    full_source_y = centered_range(gridsize, FULL_SENSOR_Y)
    full_source_x = centered_range(gridsize, FULL_SENSOR_X)
    frame_shape, source_y, source_x, readout_window = if roi_mode === :software
        (
            (FULL_SENSOR_X, FULL_SENSOR_Y),
            full_source_y,
            full_source_x,
            AdaptiveOpticsSim.Detectors.FrameWindow(roi_x, roi_y),
        )
    elseif roi_mode === :hardware
        (
            (ROI_X, ROI_Y),
            source_range(full_source_y, roi_y),
            source_range(full_source_x, roi_x),
            nothing,
        )
    else
        error("ROI mode must be software or hardware")
    end

    detector = AdaptiveOpticsSim.Detectors.Detector(
        exposure_duration=1f-3,
        qe=1f0,
        noise=AdaptiveOpticsSim.Detectors.NoiseNone(),
        sensor=AdaptiveOpticsSim.Detectors.InGaAsSensor(T=Float32),
        psf_sampling=1,
        binning=1,
        readout_window=readout_window,
        output_type=Float32,
        T=Float32,
    )
    hil = prepare_detector_hil(
        optical_model,
        detector;
        frame_shape=frame_shape,
        wavelength_m=1.55f-6,
        source_rows=source_y,
        source_cols=source_x,
        source_axes=(:y, :x),
        normalized_to_photon_rate=1f6,
        rng=AdaptiveOpticsSim.runtime_rng(20260831),
    )
    return (; hil, llowfs, sampling_m, beam_diameter_fraction)
end

function main(args)
    gridsize = length(args) >= 1 ? parse(Int, args[1]) : 512
    roi_mode = length(args) >= 2 ? Symbol(lowercase(args[2])) : :software
    default_x = first(centered_range(FULL_SENSOR_X, ROI_X))
    default_y = first(centered_range(FULL_SENSOR_Y, ROI_Y))
    roi_x_start = length(args) >= 3 ? parse(Int, args[3]) : default_x
    roi_y_start = length(args) >= 4 ? parse(Int, args[4]) : default_y
    roi_x = roi_range(roi_x_start, ROI_X, FULL_SENSOR_X, "ROI x")
    roi_y = roi_range(roi_y_start, ROI_Y, FULL_SENSOR_Y, "ROI y")

    context = build_spiders_llowfs_hil(gridsize, roi_mode, roi_x, roi_y)
    frame = detector_hil_step!(context.hil)
    detector_hil_step!(context.hil)
    optical_allocations = @allocated propagate_llowfs!(context.llowfs)
    combined_allocations = @allocated detector_hil_step!(context.hil)

    println("SPIDERS Zemax LLOWFS prepared HIL validation")
    println("  grid: ", gridsize)
    println("  beam_diameter_fraction: ", context.beam_diameter_fraction)
    println("  output_sampling_m: ", context.sampling_m)
    println("  detector_storage_convention: (SizeX, SizeY)")
    println("  full_sensor_shape: ", (FULL_SENSOR_X, FULL_SENSOR_Y))
    println("  roi_mode: ", roi_mode)
    println("  provisional_roi_x: ", roi_x)
    println("  provisional_roi_y: ", roi_y)
    println("  output_shape: ", size(frame))
    println("  output_checksum: ", sum(frame))
    println("  proper_allocated_bytes_per_call: ", optical_allocations)
    println("  combined_allocated_bytes_per_call: ", combined_allocations)
    println("  warning: replace the centered ROI origin with live VENOMS OffsetX/OffsetY")
    return nothing
end

main(ARGS)
