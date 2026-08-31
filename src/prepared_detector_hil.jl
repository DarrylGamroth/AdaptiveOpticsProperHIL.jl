const _AOOptics = AdaptiveOpticsSim.Optics
const _AODetectors = AdaptiveOpticsSim.Detectors

"""
    PreparedOpticalIntensityModel(prepared, propagate!, intensity, sampling_m)

Bind an allocation-stable prepared optical model to the operations required by
the detector HIL seam. `propagate!(prepared)` must update and return the same
real intensity matrix returned by `intensity(prepared)`. `sampling_m` is the
physical sample pitch in both matrix dimensions.
"""
struct PreparedOpticalIntensityModel{P,F,G,T<:AbstractFloat}
    prepared::P
    propagate!::F
    intensity::G
    sampling_m::T
end

function PreparedOpticalIntensityModel(
    prepared,
    propagate!::F,
    intensity::G,
    sampling_m::Real,
) where {F,G}
    source = intensity(prepared)
    source isa AbstractMatrix{<:AbstractFloat} || throw(ArgumentError(
        "prepared optical intensity must be a real matrix",
    ))
    T = float(real(eltype(source)))
    pitch = T(sampling_m)
    isfinite(pitch) && pitch > zero(T) || throw(ArgumentError(
        "optical sampling_m must be finite and positive",
    ))
    return PreparedOpticalIntensityModel{
        typeof(prepared),F,G,T,
    }(prepared, propagate!, intensity, pitch)
end

@inline propagate_optics!(model::PreparedOpticalIntensityModel) =
    model.propagate!(model.prepared)
@inline optical_intensity(model::PreparedOpticalIntensityModel) =
    model.intensity(model.prepared)
@inline optical_sampling_m(model::PreparedOpticalIntensityModel) =
    model.sampling_m

"""
    PreparedDetectorHIL

Single-writer owner for a prepared optical model, a direct native-pixel window
of its intensity, and an AdaptiveOpticsSim prepared detector acquisition.
Preparation, array ownership, ROI selection, and detector allocation remain
outside the repeated call to [`detector_hil_step!`](@ref).
"""
struct PreparedDetectorHIL{M,A,I,D,R,RY,RX,Mapping}
    model::M
    detector_values::A
    intensity_map::I
    acquisition::D
    rng::R
    source_rows::RY
    source_cols::RX
    axis_mapping::Mapping
end

@inline function _copy_optical_window!(
    destination::Matrix,
    source::Matrix,
    source_rows::UnitRange{Int},
    source_cols::UnitRange{Int},
    ::Val{false},
)
    return _copy_optical_window!(
        destination, source, source_rows, source_cols,
    )
end


@inline function _copy_optical_window!(
    destination::Matrix,
    source::Matrix,
    source_rows::UnitRange{Int},
    source_cols::UnitRange{Int},
    ::Val{true},
)
    first_row = first(source_rows) - 1
    first_col = first(source_cols) - 1
    @inbounds for y in axes(destination, 2)
        source_row = first_row + y
        for x in axes(destination, 1)
            destination[x, y] = source[source_row, first_col + x]
        end
    end
    return destination
end

@inline function _centered_range(parent_length::Int, child_length::Int)
    0 < child_length <= parent_length || throw(DimensionMismatch(
        "detector frame length $child_length does not fit optical length $parent_length",
    ))
    first_index = fld(parent_length - child_length, 2) + 1
    return first_index:(first_index + child_length - 1)
end

@inline function _copy_optical_window!(
    destination::AbstractMatrix,
    source::AbstractMatrix,
    source_rows::UnitRange{Int},
    source_cols::UnitRange{Int},
    ::Val{false},
)
    return _copy_optical_window!(
        destination, source, source_rows, source_cols,
    )
end


@inline function _copy_optical_window!(
    destination::AbstractMatrix,
    source::AbstractMatrix,
    source_rows::UnitRange{Int},
    source_cols::UnitRange{Int},
    ::Val{true},
)
    permutedims!(
        destination,
        view(source, source_rows, source_cols),
        (2, 1),
    )
    return destination
end

function _resolve_source_range(
    selected::Union{Nothing,UnitRange{Int}},
    parent_length::Int,
    child_length::Int,
    label::AbstractString,
)
    range = isnothing(selected) ?
            _centered_range(parent_length, child_length) : selected
    length(range) == child_length || throw(DimensionMismatch(
        "$label length $(length(range)) does not match detector frame length $child_length",
    ))
    first(range) >= 1 && last(range) <= parent_length || throw(BoundsError(
        Base.OneTo(parent_length), range,
    ))
    return range
end

@inline function _copy_optical_window!(
    destination::Matrix,
    source::Matrix,
    source_rows::UnitRange{Int},
    source_cols::UnitRange{Int},
)
    first_row = first(source_rows) - 1
    first_col = first(source_cols) - 1
    @inbounds for column in axes(destination, 2)
        source_column = first_col + column
        for row in axes(destination, 1)
            destination[row, column] = source[first_row + row, source_column]
        end
    end
    return destination
end

@inline function _copy_optical_window!(
    destination::AbstractMatrix,
    source::AbstractMatrix,
    source_rows::UnitRange{Int},
    source_cols::UnitRange{Int},
)
    copyto!(destination, view(source, source_rows, source_cols))
    return destination
end

"""
    prepare_detector_hil(model, detector; frame_shape, wavelength_m,
        source_rows=nothing, source_cols=nothing, source_axes=(:x, :y),
        normalized_to_photon_rate, rng)

Prepare a direct, one-PROPER-sample-per-detector-pixel handoff. The detector
must use `psf_sampling=1` and `binning=1`; this seam deliberately performs no
binning or resampling. `source_rows` and `source_cols` select the physical
sensor footprint from the optical grid and default to a centered footprint.
Set `source_axes=(:y, :x)` for conventional row-column optical arrays that
must be written into the SPIDERS/VENOMS `(SizeX, SizeY)` detector convention.

An AdaptiveOpticsSim `FrameWindow` on `detector` represents a software crop of
the completed frame. To represent hardware ROI readout, set `frame_shape` to
the ROI dimensions and leave the detector readout window unset.
"""
function prepare_detector_hil(
    model::PreparedOpticalIntensityModel,
    detector;
    frame_shape::Tuple{Int,Int},
    wavelength_m::Real,
    source_rows::Union{Nothing,UnitRange{Int}}=nothing,
    source_cols::Union{Nothing,UnitRange{Int}}=nothing,
    source_axes::NTuple{2,Symbol}=(:x, :y),
    normalized_to_photon_rate::Real,
    rng=AdaptiveOpticsSim.runtime_rng(20260831),
)
    detector.params.psf_sampling == 1 || throw(ArgumentError(
        "native-pixel detector HIL requires psf_sampling=1",
    ))
    detector.params.binning == 1 || throw(ArgumentError(
        "native-pixel detector HIL requires binning=1",
    ))
    all(>(0), frame_shape) || throw(ArgumentError(
        "detector frame dimensions must be positive",
    ))

    source_axes in ((:x, :y), (:y, :x)) || throw(ArgumentError(
        "source_axes must be (:x, :y) or (:y, :x)",
    ))
    transpose_source = source_axes === (:y, :x)
    source_frame_shape = transpose_source ? reverse(frame_shape) : frame_shape
    source = optical_intensity(model)
    rows = _resolve_source_range(
        source_rows, size(source, 1), source_frame_shape[1], "source_rows",
    )
    cols = _resolve_source_range(
        source_cols, size(source, 2), source_frame_shape[2], "source_cols",
    )
    detector_values = similar(source, eltype(source), frame_shape)
    axis_mapping = Val(transpose_source)
    _copy_optical_window!(
        detector_values, source, rows, cols, axis_mapping,
    )

    T = eltype(detector_values)
    wavelength = T(wavelength_m)
    isfinite(wavelength) && wavelength > zero(T) || throw(ArgumentError(
        "wavelength_m must be finite and positive",
    ))
    sampling = (T(optical_sampling_m(model)), T(optical_sampling_m(model)))
    metadata = _AOOptics.OpticalPlaneMetadata(
        _AOOptics.DetectorPlane(),
        detector_values;
        coordinate_domain=_AOOptics.MetricCoordinates(),
        sampling,
        orientation=_AOOptics.PlaneAxisOrientation((:x, :y), (1, 1)),
        spectral=_AOOptics.MonochromaticChannel(wavelength),
        normalization=_AOOptics.DimensionlessNormalization(),
        spatial_measure=_AOOptics.CellIntegratedMeasure(),
        coherence=_AOOptics.IncoherentIntensityAddition(),
    )
    intensity_map = _AOOptics.IntensityMap(metadata, detector_values)
    acquisition = _AODetectors.prepare_detector_acquisition(
        detector,
        intensity_map;
        normalized_to_photon_rate,
    )
    return PreparedDetectorHIL(
        model,
        detector_values,
        intensity_map,
        acquisition,
        rng,
        rows,
        cols,
        axis_mapping,
    )
end

"""
    detector_hil_step!(prepared)

Propagate the bound optical model, copy its selected native pixels into the
bound detector intensity, and run one prepared detector acquisition.
"""
function detector_hil_step!(prepared::PreparedDetectorHIL)
    source = propagate_optics!(prepared.model)
    _copy_optical_window!(
        prepared.detector_values,
        source,
        prepared.source_rows,
        prepared.source_cols,
        prepared.axis_mapping,
    )
    return _AODetectors.capture!(prepared.acquisition, prepared.rng)
end

@inline detector_hil_frame(prepared::PreparedDetectorHIL) =
    _AODetectors.output_frame(prepared.acquisition.detector)
