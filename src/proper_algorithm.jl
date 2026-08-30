const DEFAULT_PUPIL_OPD_SCHEMA = "org.adaptiveopticssim.pupil-opd/1"
const DEFAULT_PUPIL_AMPLITUDE_SCHEMA =
    "org.adaptiveopticssim.pupil-amplitude/1"
const DEFAULT_NORMALIZED_INTENSITY_SCHEMA =
    "org.adaptiveopticssim.normalized-intensity/1"

const _AOG = AdaptiveOpticsSim.AlgorithmGraphs
const _Backends = AdaptiveOpticsSim.Backends

"""
Typed construction configuration for one AOS-native Proper propagation node.

The prescription receives `(wavelength_m, resolution)` and the prepared
keywords `pupil_opd`, `pupil_amplitude`, `diameter_m`, `field`, `wavefront`,
`output`, and `run_context`. It must write `output` and return the usual Proper
`(output, sampling_m)` result. `resolution` fixes the square Proper wavefront
grid. `pupil_resolution` independently fixes both square pupil inputs and
defaults to `resolution`; a padded prescription must explicitly embed or
resample those inputs into its prepared propagation-grid scratch. `output_rows`
and `output_columns` independently declare the caller-visible result and
default to `resolution`. A prescription that uses a different camera shape
must perform its prepared crop, resampling, or detector mapping into `output`
itself. The containing algorithm graph selects the exact CPU or GPU target;
every ndarray used by this node has one element type and that same target.
"""
struct ProperPropagationConfiguration{
    T<:Union{Float32,Float64},
    Prescription<:Function,
}
    prescription::Prescription
    resolution::Int
    pupil_resolution::Int
    output_rows::Int
    output_columns::Int
    diameter_m::T
    wavelength_um::T
    rng_seed::UInt64
    pupil_opd_schema::String
    pupil_amplitude_schema::String
    output_schema::String
end

function ProperPropagationConfiguration(
    prescription::Function;
    resolution::Integer,
    pupil_resolution::Union{Nothing,Integer}=nothing,
    output_rows::Union{Nothing,Integer}=nothing,
    output_columns::Union{Nothing,Integer}=nothing,
    diameter_m::Real,
    wavelength_um::Real,
    rng_seed::Integer=0,
    pupil_opd_schema::AbstractString=DEFAULT_PUPIL_OPD_SCHEMA,
    pupil_amplitude_schema::AbstractString=DEFAULT_PUPIL_AMPLITUDE_SCHEMA,
    output_schema::AbstractString=DEFAULT_NORMALIZED_INTENSITY_SCHEMA,
)
    n = Int(resolution)
    n > 0 || throw(ArgumentError("resolution must be positive"))
    pupil_n = isnothing(pupil_resolution) ? n : Int(pupil_resolution)
    pupil_n > 0 || throw(ArgumentError(
        "pupil_resolution must be positive",
    ))
    rows = isnothing(output_rows) ? n : Int(output_rows)
    columns = isnothing(output_columns) ? n : Int(output_columns)
    rows > 0 || throw(ArgumentError("output_rows must be positive"))
    columns > 0 || throw(ArgumentError("output_columns must be positive"))
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
    all(schema -> !isempty(schema), schemas) || throw(ArgumentError(
        "Proper propagation schemas must not be empty",
    ))
    return ProperPropagationConfiguration{
        T,
        typeof(prescription),
    }(
        prescription,
        n,
        pupil_n,
        rows,
        columns,
        diameter,
        wavelength,
        UInt64(rng_seed),
        schemas...,
    )
end

function ProperPropagationConfiguration(prescription; kwargs...)
    throw(ArgumentError(
        "prescription must be a Julia function accepted by Proper.prepare_prescription",
    ))
end

"""
    prepare_proper_state(prescription, T)

Prepare prescription-specific persistent scientific state. The default returns
`nothing`. State is owned by the prepared graph node and is distinct from the
immutable propagation plan and the replaceable scratch returned by
[`prepare_proper_assets`](@ref).
"""
prepare_proper_state(prescription, ::Type) = nothing

"""Reset prescription-specific persistent state at a serialized boundary."""
reset_proper_state!(prescription, state) = state

_proper_state_keywords(prescription, state) = NamedTuple()

"""
    prepare_proper_assets(prescription, target, T, resolution, output_shape)

Prepare a named tuple of prescription-specific keyword assets. The default is
empty. A prescription may specialize this method to allocate masks, plans, or
replaceable scratch on `target`; `output_shape` is the declared
`(output_rows, output_columns)` tuple. The returned assets belong to the
prepared workspace, not the immutable propagation plan.
"""
prepare_proper_assets(
    prescription,
    target,
    ::Type,
    resolution::Int,
    output_shape::Tuple{Int,Int},
) = (;)

"""Reset prescription-specific prepared assets at a serialized graph boundary."""
reset_proper_assets!(prescription, assets::NamedTuple) = assets

struct _ProperPropagationNode{
    T<:Union{Float32,Float64},
    Prescription<:Function,
} end

struct _ProperPropagationPlan{
    T<:Union{Float32,Float64},
    Prescription<:Function,
    Target<:_Backends.AbstractComputeDevice,
}
    prescription::Prescription
    target::Target
    resolution::Int
    pupil_resolution::Int
    output_rows::Int
    output_columns::Int
    diameter_m::T
    wavelength_um::T
    rng_seed::UInt64
end

mutable struct _ProperPropagationState{RNG,PrescriptionState}
    rng::RNG
    initial_rng::RNG
    prescription::PrescriptionState
end

struct _ProperPropagationWorkspace{Context,Assets,Field,Wavefront,Run}
    context::Context
    assets::Assets
    field::Field
    wavefront::Wavefront
    run::Run
end

struct _ProperPropagationOwner{Plan,State,Workspace,OPD,Amplitude,Output}
    plan::Plan
    state::State
    workspace::Workspace
    pupil_opd::OPD
    pupil_amplitude::Amplitude
    output::Output
end

"""
    proper_propagation_node(name, configuration)

Declare one AOS-native graph node around a prepared Proper prescription. This
is a Julia graph adapter, not a second scientific algorithm API: the
prescription remains an ordinary Proper Julia function, while
`AdaptiveOpticsSim.AlgorithmGraphs` owns topology, exact buffers, and stepping.
"""
function proper_propagation_node(
    name::Symbol,
    configuration::ProperPropagationConfiguration{T,Prescription},
) where {T,Prescription}
    return _AOG.algorithm_node(
        name,
        _ProperPropagationNode{T,Prescription},
        configuration;
        props=NamedTuple(),
    )
end

function _proper_plan(
    configuration::ProperPropagationConfiguration,
    target::_Backends.AbstractComputeDevice,
)
    return _ProperPropagationPlan(
        configuration.prescription,
        target,
        configuration.resolution,
        configuration.pupil_resolution,
        configuration.output_rows,
        configuration.output_columns,
        configuration.diameter_m,
        configuration.wavelength_um,
        configuration.rng_seed,
    )
end

function _prepare_proper_state(prescription, ::Type{T}, seed::UInt64) where {T}
    rng = Random.Xoshiro(seed)
    return _ProperPropagationState(
        rng,
        copy(rng),
        prepare_proper_state(prescription, T),
    )
end

@inline function _reset_proper_state!(
    prescription,
    state::_ProperPropagationState,
)
    copy!(state.rng, state.initial_rng)
    reset_proper_state!(prescription, state.prescription)
    return state
end

function _validate_proper_buffer(
    plan::_ProperPropagationPlan{T},
    values,
    role::AbstractString,
    expected::Tuple{Int,Int},
) where {T}
    eltype(values) === T || throw(ArgumentError(
        "$role element type $(eltype(values)) does not match $T",
    ))
    ndims(values) == 2 || throw(DimensionMismatch("$role must be a matrix"))
    size(values) == expected || throw(DimensionMismatch(
        "$role shape $(size(values)) does not match $expected",
    ))
    actual = _Backends.compute_device(values)
    actual == plan.target || throw(ArgumentError(
        "$role target $actual does not match $(plan.target)",
    ))
    return values
end

function _allocate_proper_buffer(plan::_ProperPropagationPlan, ::Type{T}) where {T}
    values = _Backends.allocate_device_array(
        plan.target,
        T,
        plan.resolution,
        plan.resolution,
    )
    fill!(values, zero(T))
    return values
end

const _STANDARD_PROPER_ASSET_KEYS = (
    :pupil_opd,
    :pupil_amplitude,
    :diameter_m,
    :field,
    :wavefront,
    :output,
    :prescription_state,
    :run_context,
)

function _validate_proper_asset_target(
    value::AbstractArray,
    target,
    name::Symbol,
)
    actual = _Backends.compute_device(value)
    actual == target || throw(ArgumentError(
        "prepared Proper asset $name target $actual does not match $target",
    ))
    return value
end

function _validate_proper_asset_target(values::NamedTuple, target, name::Symbol)
    for (nested_name, value) in pairs(values)
        _validate_proper_asset_target(value, target, nested_name)
    end
    return values
end

function _validate_proper_asset_target(values::Tuple, target, name::Symbol)
    for value in values
        _validate_proper_asset_target(value, target, name)
    end
    return values
end

_validate_proper_asset_target(value, target, name::Symbol) = value

function _validate_proper_assets(assets, target)
    assets isa NamedTuple || throw(ArgumentError(
        "prepare_proper_assets must return a NamedTuple",
    ))
    for name in keys(assets)
        name in _STANDARD_PROPER_ASSET_KEYS && throw(ArgumentError(
            "prepared Proper asset $name conflicts with an adapter-owned keyword",
        ))
    end
    _validate_proper_asset_target(assets, target, :assets)
    return assets
end

function _prepare_proper_workspace(
    plan::_ProperPropagationPlan{T},
    state::_ProperPropagationState,
    pupil_opd,
    pupil_amplitude,
    output,
) where {T}
    pupil_shape = (plan.pupil_resolution, plan.pupil_resolution)
    output_shape = (plan.output_rows, plan.output_columns)
    _validate_proper_buffer(plan, pupil_opd, "pupil OPD", pupil_shape)
    _validate_proper_buffer(
        plan,
        pupil_amplitude,
        "pupil amplitude",
        pupil_shape,
    )
    _validate_proper_buffer(plan, output, "Proper output", output_shape)

    field = _allocate_proper_buffer(plan, Complex{T})
    run_context = Proper.RunContext(typeof(field); rng=state.rng)
    wavefront = Proper.prop_begin!(
        field,
        plan.diameter_m,
        plan.wavelength_um * T(1e-6);
        beam_diam_fraction=one(T),
        context=run_context,
    )
    assets = _validate_proper_assets(
        prepare_proper_assets(
            plan.prescription,
            plan.target,
            T,
            plan.resolution,
            output_shape,
        ),
        plan.target,
    )
    prescription = Proper.prepare_prescription(
        plan.prescription,
        plan.wavelength_um,
        plan.resolution;
        context=run_context,
        precision=T,
    )
    standard = (
        pupil_opd=pupil_opd,
        pupil_amplitude=pupil_amplitude,
        diameter_m=plan.diameter_m,
        field=field,
        wavefront=wavefront,
        output=output,
        run_context=run_context,
    )
    state_keywords = _proper_state_keywords(
        plan.prescription,
        state.prescription,
    )
    run = Proper.prepare_run(
        prescription;
        context=run_context,
        activate_context=false,
        merge(assets, standard, state_keywords)...,
    )
    return _ProperPropagationWorkspace(
        run_context,
        assets,
        field,
        wavefront,
        run,
    )
end

@inline function _run_proper_workspace!(
    workspace::_ProperPropagationWorkspace,
    output,
)
    result = Proper.prop_run(workspace.run)
    first(result) === output || throw(ArgumentError(
        "the Proper prescription must return its graph-owned output buffer",
    ))
    return output
end

function _AOG.graph_node_ports(
    ::Type{_ProperPropagationNode{T,Prescription}},
    configuration::ProperPropagationConfiguration{T,Prescription},
) where {T,Prescription}
    pupil_shape = (
        configuration.pupil_resolution,
        configuration.pupil_resolution,
    )
    output_shape = (
        configuration.output_rows,
        configuration.output_columns,
    )
    return (
        _AOG.graph_port_contract(
            :pupil_opd,
            :input,
            :data,
            T,
            pupil_shape,
            configuration.pupil_opd_schema,
            :column_major,
        ),
        _AOG.graph_port_contract(
            :pupil_amplitude,
            :input,
            :data,
            T,
            pupil_shape,
            configuration.pupil_amplitude_schema,
            :column_major,
        ),
        _AOG.graph_port_contract(
            :output,
            :output,
            :data,
            T,
            output_shape,
            configuration.output_schema,
            :column_major,
        ),
    )
end

function _AOG.prepare_graph_node(
    ::Type{_ProperPropagationNode{T,Prescription}},
    configuration::ProperPropagationConfiguration{T,Prescription},
    ::NamedTuple{()},
    inputs::NamedTuple{(:pupil_opd,:pupil_amplitude)},
    outputs::NamedTuple{(:output,)},
    ::NamedTuple{()},
    target::_Backends.AbstractComputeDevice,
) where {T,Prescription}
    plan = _proper_plan(configuration, target)
    state = _prepare_proper_state(plan.prescription, T, plan.rng_seed)
    workspace = _prepare_proper_workspace(
        plan,
        state,
        inputs.pupil_opd,
        inputs.pupil_amplitude,
        outputs.output,
    )
    return _ProperPropagationOwner(
        plan,
        state,
        workspace,
        inputs.pupil_opd,
        inputs.pupil_amplitude,
        outputs.output,
    )
end

@inline function _AOG.step_graph_node!(owner::_ProperPropagationOwner)
    _run_proper_workspace!(owner.workspace, owner.output)
    return nothing
end

function _AOG.reset_graph_node!(owner::_ProperPropagationOwner)
    _reset_proper_state!(owner.plan.prescription, owner.state)
    reset_proper_assets!(owner.plan.prescription, owner.workspace.assets)
    return nothing
end

struct ProperPropagationNodeFactory{
    T<:Union{Float32,Float64},
    Prescription<:Function,
}
    prescription::Prescription
end

"""
    proper_propagation_node_factory(prescription; T=Float32)

Return a cold node factory suitable for an explicit `node_types` entry passed
to `AlgorithmGraphs.load_algorithm_graph`. The application supplies the Julia
prescription; the TOML file can select only the name under which the
application admitted it and cannot evaluate or resolve Julia code.
"""
function proper_propagation_node_factory(
    prescription::Function;
    T::Type{<:Union{Float32,Float64}}=Float32,
)
    return ProperPropagationNodeFactory{T,typeof(prescription)}(prescription)
end

struct ConfiguredProperPropagationNodeFactory{
    Configuration<:ProperPropagationConfiguration,
}
    configuration::Configuration
end

"""
    proper_propagation_node_factory(configuration)

Return a cold TOML node factory with a complete trusted Proper configuration
captured by the Julia application. Nodes using this factory must contain an
empty `config` table and no props. This form is intended for instrument graphs
whose physical prescription and configuration come from an application-owned
profile rather than from scalar literals in the topology file.
"""
function proper_propagation_node_factory(
    configuration::ProperPropagationConfiguration,
)
    return ConfiguredProperPropagationNodeFactory(configuration)
end

function (factory::ConfiguredProperPropagationNodeFactory)(
    name::Symbol,
    config::NamedTuple,
    props::NamedTuple,
)
    _require_proper_file_fields(config, (), (), "node '$name' config")
    _require_proper_file_fields(props, (), (), "node '$name' props")
    return proper_propagation_node(name, factory.configuration)
end

function _require_proper_file_fields(
    values::NamedTuple,
    allowed::Tuple,
    required::Tuple,
    context::AbstractString,
)
    for name in keys(values)
        name in allowed || throw(_AOG.AlgorithmGraphError(
            "$context has unknown field '$name'",
        ))
    end
    for name in required
        hasproperty(values, name) || throw(_AOG.AlgorithmGraphError(
            "$context is missing field '$name'",
        ))
    end
    return nothing
end

function _proper_file_integer(value, context::AbstractString)
    value isa Integer && !(value isa Bool) || throw(_AOG.AlgorithmGraphError(
        "$context must be an integer",
    ))
    return Int(value)
end

function _proper_file_real(value, context::AbstractString)
    value isa Real && !(value isa Bool) || throw(_AOG.AlgorithmGraphError(
        "$context must be a real number",
    ))
    isfinite(value) || throw(_AOG.AlgorithmGraphError(
        "$context must be finite",
    ))
    return value
end

function _proper_file_string(value, context::AbstractString)
    value isa AbstractString || throw(_AOG.AlgorithmGraphError(
        "$context must be a string",
    ))
    return String(value)
end

function (factory::ProperPropagationNodeFactory{T})(
    name::Symbol,
    config::NamedTuple,
    props::NamedTuple,
) where {T}
    allowed = (
        :diameter_m,
        :output_columns,
        :output_rows,
        :output_schema,
        :pupil_resolution,
        :pupil_amplitude_schema,
        :pupil_opd_schema,
        :resolution,
        :rng_seed,
        :wavelength_um,
    )
    required = (:diameter_m, :resolution, :wavelength_um)
    _require_proper_file_fields(config, allowed, required, "node '$name' config")
    _require_proper_file_fields(props, (), (), "node '$name' props")
    configuration = try
        ProperPropagationConfiguration(
            factory.prescription;
            resolution=_proper_file_integer(
                config.resolution,
                "node '$name' config.resolution",
            ),
            pupil_resolution=hasproperty(config, :pupil_resolution) ?
                _proper_file_integer(
                    config.pupil_resolution,
                    "node '$name' config.pupil_resolution",
                ) : nothing,
            output_rows=hasproperty(config, :output_rows) ?
                _proper_file_integer(
                    config.output_rows,
                    "node '$name' config.output_rows",
                ) : nothing,
            output_columns=hasproperty(config, :output_columns) ?
                _proper_file_integer(
                    config.output_columns,
                    "node '$name' config.output_columns",
                ) : nothing,
            diameter_m=T(_proper_file_real(
                config.diameter_m,
                "node '$name' config.diameter_m",
            )),
            wavelength_um=T(_proper_file_real(
                config.wavelength_um,
                "node '$name' config.wavelength_um",
            )),
            rng_seed=hasproperty(config, :rng_seed) ? _proper_file_integer(
                config.rng_seed,
                "node '$name' config.rng_seed",
            ) : 0,
            pupil_opd_schema=hasproperty(config, :pupil_opd_schema) ?
                _proper_file_string(
                    config.pupil_opd_schema,
                    "node '$name' config.pupil_opd_schema",
                ) : DEFAULT_PUPIL_OPD_SCHEMA,
            pupil_amplitude_schema=hasproperty(
                config,
                :pupil_amplitude_schema,
            ) ? _proper_file_string(
                config.pupil_amplitude_schema,
                "node '$name' config.pupil_amplitude_schema",
            ) : DEFAULT_PUPIL_AMPLITUDE_SCHEMA,
            output_schema=hasproperty(config, :output_schema) ?
                _proper_file_string(
                    config.output_schema,
                    "node '$name' config.output_schema",
                ) : DEFAULT_NORMALIZED_INTENSITY_SCHEMA,
        )
    catch error
        error isa InterruptException && rethrow()
        error isa _AOG.AlgorithmGraphError && rethrow()
        throw(_AOG.AlgorithmGraphError(
            "node '$name' config is invalid: $(sprint(showerror, error))",
        ))
    end
    return proper_propagation_node(name, configuration)
end
