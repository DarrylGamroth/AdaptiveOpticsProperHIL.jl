const DEFAULT_PUPIL_OPD_SCHEMA = "org.adaptiveopticssim.pupil-opd/1"
const DEFAULT_PUPIL_AMPLITUDE_SCHEMA =
    "org.adaptiveopticssim.pupil-amplitude/1"
const DEFAULT_NORMALIZED_INTENSITY_SCHEMA =
    "org.adaptiveopticssim.normalized-intensity/1"

"""
Typed construction configuration for one Calculon-compatible Proper
prescription.

The prescription receives `(wavelength_m, resolution)` and the prepared
keywords `pupil_opd`, `pupil_amplitude`, `diameter_m`, `field`, `wavefront`,
`output`, and `run_context`. It must write `output` and return the usual Proper
`(output, sampling_m)` result. Every ndarray uses one element type and one exact
compute target.
"""
struct ProperPropagationConfiguration{
    T<:Union{Float32,Float64},
    Prescription,
    Target<:AdaptiveOpticsSim.Backends.AbstractComputeDevice,
}
    prescription::Prescription
    target::Target
    resolution::Int
    diameter_m::T
    wavelength_um::T
    rng_seed::UInt64
    pupil_opd_schema::String
    pupil_amplitude_schema::String
    output_schema::String
end

function ProperPropagationConfiguration(
    prescription;
    target::AdaptiveOpticsSim.Backends.AbstractComputeDevice=
        AdaptiveOpticsSim.Backends.HostComputeDevice(),
    resolution::Integer,
    diameter_m::Real,
    wavelength_um::Real,
    rng_seed::Integer=0,
    pupil_opd_schema::AbstractString=DEFAULT_PUPIL_OPD_SCHEMA,
    pupil_amplitude_schema::AbstractString=DEFAULT_PUPIL_AMPLITUDE_SCHEMA,
    output_schema::AbstractString=DEFAULT_NORMALIZED_INTENSITY_SCHEMA,
)
    prescription isa Function || throw(ArgumentError(
        "prescription must be a Julia function accepted by Proper.prepare_run",
    ))
    n = Int(resolution)
    n > 0 || throw(ArgumentError("resolution must be positive"))
    diameter, wavelength = promote(float(diameter_m), float(wavelength_um))
    T = typeof(diameter)
    T in (Float32, Float64) || throw(ArgumentError(
        "diameter and wavelength must promote to Float32 or Float64",
    ))
    isfinite(diameter) && diameter > zero(T) || throw(ArgumentError(
        "diameter_m must be finite and positive",
    ))
    isfinite(wavelength) && wavelength > zero(T) || throw(ArgumentError(
        "wavelength_um must be finite and positive",
    ))
    0 <= rng_seed <= typemax(UInt64) || throw(ArgumentError(
        "rng_seed must fit UInt64",
    ))
    schemas = (
        String(pupil_opd_schema),
        String(pupil_amplitude_schema),
        String(output_schema),
    )
    all(!isempty, schemas) || throw(ArgumentError(
        "Proper propagation schemas must not be empty",
    ))
    return ProperPropagationConfiguration{
        T,
        typeof(prescription),
        typeof(target),
    }(
        prescription,
        target,
        n,
        diameter,
        wavelength,
        UInt64(rng_seed),
        schemas...,
    )
end

"""
    prepare_proper_assets(prescription, target, T, resolution)

Prepare a named tuple of prescription-specific keyword assets. The default is
empty. A prescription may specialize this method to allocate masks, plans, or
replaceable scratch on `target`; the returned assets belong to the prepared
workspace, not the immutable algorithm plan.
"""
prepare_proper_assets(prescription, target, ::Type, resolution::Int) = (;)

"""Reset prescription-specific prepared assets at a serialized graph boundary."""
reset_proper_assets!(prescription, assets::NamedTuple) = assets

"""
Return the Calculon declaration type for `configuration`.

Load `CalculonAlgorithms` before calling this function. Proper remains an
optional companion of AdaptiveOpticsSim rather than a core dependency.
"""
function proper_propagation_declaration(
    configuration::ProperPropagationConfiguration,
)
    extension = Base.get_extension(
        @__MODULE__,
        :AdaptiveOpticsProperHILCalculonAlgorithmsExt,
    )
    isnothing(extension) && throw(ArgumentError(
        "load CalculonAlgorithms to enable the Proper algorithm declaration",
    ))
    return extension._proper_propagation_declaration(configuration)
end
