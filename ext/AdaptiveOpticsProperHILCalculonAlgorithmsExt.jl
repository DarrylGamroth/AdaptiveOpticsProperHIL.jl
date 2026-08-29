module AdaptiveOpticsProperHILCalculonAlgorithmsExt

import AdaptiveOpticsProperHIL
import AdaptiveOpticsSim
import CalculonAlgorithms
import Proper
import Random

const Parent = AdaptiveOpticsProperHIL
const AOG = AdaptiveOpticsSim.AlgorithmGraphs
const Backends = AdaptiveOpticsSim.Backends
const CA = CalculonAlgorithms

struct ProperPropagation{
    T<:Union{Float32,Float64},
    Prescription,
    Target<:Backends.AbstractComputeDevice,
} <: CA.AbstractAlgorithmDeclaration end

struct ProperPropagationPlan{
    T<:Union{Float32,Float64},
    Prescription,
    Target<:Backends.AbstractComputeDevice,
} <: CA.AbstractAlgorithmPlan
    prescription::Prescription
    target::Target
    resolution::Int
    diameter_m::T
    wavelength_um::T
    rng_seed::UInt64
end

struct _ProperPropagationState{RNG}
    rng::RNG
    initial_rng::RNG
end

function _prepare_state(seed::UInt64)
    rng = Random.Xoshiro(seed)
    return _ProperPropagationState(rng, copy(rng))
end

@inline function _reset_state!(state::_ProperPropagationState)
    copy!(state.rng, state.initial_rng)
    return state
end

struct _ProperInvocation{Context,Assets,Field,Wavefront,Run,Output}
    context::Context
    assets::Assets
    field::Field
    wavefront::Wavefront
    run::Run
    output::Output
end

struct _DirectProperWorkspace{Context,Invocation,OPD,Amplitude,Output}
    context::Context
    invocation::Invocation
    pupil_opd::OPD
    pupil_amplitude::Amplitude
    output::Output
end

struct _DirectProperExecution{State,Workspace}
    state::State
    workspace::Workspace
end

struct _ProperGraphPrototype{Declaration,Plan,Formats}
    plan::Plan
    formats::Formats
end

struct _GraphBoundProperPropagation{
    Plan,
    State,
    Invocation,
    OPD,
    Amplitude,
    Output,
}
    plan::Plan
    state::State
    invocation::Invocation
    pupil_opd::OPD
    pupil_amplitude::Amplitude
    output::Output
end

function _proper_propagation_declaration(
    ::Parent.ProperPropagationConfiguration{T,Prescription,Target},
) where {T,Prescription,Target}
    return ProperPropagation{T,Prescription,Target}
end

_precision_label(::Type{Float32}) = "f32"
_precision_label(::Type{Float64}) = "f64"

function CA.algorithm_label(
    ::Type{ProperPropagation{T,Prescription,Target}},
) where {T,Prescription,Target}
    return "proper-propagation-$(_precision_label(T))"
end

function CA.algorithm_configuration(
    ::Type{ProperPropagation{T,Prescription,Target}},
) where {T,Prescription,Target}
    construction = CA.ConstructionConfiguration()
    rebuild = CA.GraphRebuildConfiguration()
    return (
        CA.AlgorithmConfiguration(:prescription, Prescription, construction),
        CA.AlgorithmConfiguration(:target, Target, rebuild),
        CA.AlgorithmConfiguration(:resolution, Int, rebuild),
        CA.AlgorithmConfiguration(:diameter_m, T, construction),
        CA.AlgorithmConfiguration(:wavelength_um, T, construction),
        CA.AlgorithmConfiguration(:rng_seed, UInt64, construction),
        CA.AlgorithmConfiguration(:pupil_opd_schema, String, rebuild),
        CA.AlgorithmConfiguration(:pupil_amplitude_schema, String, rebuild),
        CA.AlgorithmConfiguration(:output_schema, String, rebuild),
    )
end

function CA.algorithm_ports(
    ::Type{ProperPropagation{T,Prescription,Target}},
) where {T,Prescription,Target}
    return (
        CA.AlgorithmPort(
            :pupil_opd,
            CA.InputPort(),
            CA.FrameData(),
            T,
            nothing,
        ),
        CA.AlgorithmPort(
            :pupil_amplitude,
            CA.InputPort(),
            CA.FrameData(),
            T,
            nothing,
        ),
        CA.AlgorithmPort(
            :output,
            CA.OutputPort(),
            CA.FrameData(),
            T,
            :pupil_opd,
        ),
    )
end

function CA.algorithm_property_policy(
    ::Type{<:ProperPropagation},
)
    return CA.NoDeclaredProperties()
end

function _formats(configuration::Parent.ProperPropagationConfiguration{T}) where {T}
    shape = (configuration.resolution, configuration.resolution)
    return (
        CA.AlgorithmFormat(T, shape, configuration.pupil_opd_schema),
        CA.AlgorithmFormat(T, shape, configuration.pupil_amplitude_schema),
        CA.AlgorithmFormat(T, shape, configuration.output_schema),
    )
end

function _plan(configuration::Parent.ProperPropagationConfiguration)
    return ProperPropagationPlan(
        configuration.prescription,
        configuration.target,
        configuration.resolution,
        configuration.diameter_m,
        configuration.wavelength_um,
        configuration.rng_seed,
    )
end

function _validate_buffer(
    plan::ProperPropagationPlan{T},
    values,
    role::AbstractString,
) where {T}
    storage_type = Backends.array_backend_type(
        Backends.compute_device_backend(plan.target),
    )
    values isa storage_type || throw(ArgumentError(
        "$role uses $(typeof(values)); expected native $storage_type storage",
    ))
    eltype(values) === T || throw(ArgumentError(
        "$role element type $(eltype(values)) does not match $T",
    ))
    ndims(values) == 2 || throw(DimensionMismatch("$role must be a matrix"))
    expected = (plan.resolution, plan.resolution)
    size(values) == expected || throw(DimensionMismatch(
        "$role shape $(size(values)) does not match $expected",
    ))
    actual = Backends.compute_device(values)
    actual == plan.target || throw(ArgumentError(
        "$role target $actual does not match $(plan.target)",
    ))
    return values
end

function _allocate_buffer(plan::ProperPropagationPlan, ::Type{T}) where {T}
    values = Backends.allocate_device_array(
        plan.target,
        T,
        plan.resolution,
        plan.resolution,
    )
    fill!(values, zero(T))
    return values
end

const _STANDARD_ASSET_KEYS = (
    :pupil_opd,
    :pupil_amplitude,
    :diameter_m,
    :field,
    :wavefront,
    :output,
    :run_context,
)

function _validate_asset_target(value::AbstractArray, target, name::Symbol)
    storage_type = Backends.array_backend_type(
        Backends.compute_device_backend(target),
    )
    value isa storage_type || throw(ArgumentError(
        "prepared Proper asset $name uses $(typeof(value)); expected native " *
        "$storage_type storage",
    ))
    actual = Backends.compute_device(value)
    actual == target || throw(ArgumentError(
        "prepared Proper asset $name target $actual does not match $target",
    ))
    return value
end

function _validate_asset_target(values::NamedTuple, target, name::Symbol)
    for (nested_name, value) in pairs(values)
        _validate_asset_target(value, target, nested_name)
    end
    return values
end

function _validate_asset_target(values::Tuple, target, name::Symbol)
    for value in values
        _validate_asset_target(value, target, name)
    end
    return values
end

_validate_asset_target(value, target, name::Symbol) = value

function _validate_assets(assets, target)
    assets isa NamedTuple || throw(ArgumentError(
        "prepare_proper_assets must return a NamedTuple",
    ))
    for name in keys(assets)
        name in _STANDARD_ASSET_KEYS && throw(ArgumentError(
            "prepared Proper asset $name conflicts with an adapter-owned keyword",
        ))
    end
    _validate_asset_target(assets, target, :assets)
    return assets
end

function _prepare_invocation(
    plan::ProperPropagationPlan{T},
    state::_ProperPropagationState,
    pupil_opd,
    pupil_amplitude,
    output,
) where {T}
    _validate_buffer(plan, pupil_opd, "pupil OPD")
    _validate_buffer(plan, pupil_amplitude, "pupil amplitude")
    _validate_buffer(plan, output, "Proper output")

    field = _allocate_buffer(plan, Complex{T})
    run_context = Proper.RunContext(typeof(field); rng=state.rng)
    wavefront = Proper.prop_begin!(
        field,
        plan.diameter_m,
        plan.wavelength_um * T(1e-6);
        beam_diam_fraction=one(T),
        context=run_context,
    )
    assets = _validate_assets(
        Parent.prepare_proper_assets(
            plan.prescription,
            plan.target,
            T,
            plan.resolution,
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
    run = Proper.prepare_run(
        prescription;
        context=run_context,
        activate_context=false,
        merge(assets, standard)...,
    )
    return _ProperInvocation(
        run_context,
        assets,
        field,
        wavefront,
        run,
        output,
    )
end

@inline function _run_invocation!(invocation::_ProperInvocation)
    result = Proper.prop_run(invocation.run)
    first(result) === invocation.output || throw(ArgumentError(
        "the Proper prescription must return its adapter-owned output buffer",
    ))
    return invocation.output
end

function _prepare_direct_workspace(plan::ProperPropagationPlan{T}) where {T}
    context = Backends._prepare_device_execution_context(plan.target)
    return Backends._with_prepared_device_execution_context(context) do
        try
            state = _prepare_state(plan.rng_seed)
            pupil_opd = _allocate_buffer(plan, T)
            pupil_amplitude = _allocate_buffer(plan, T)
            output = _allocate_buffer(plan, T)
            invocation = _prepare_invocation(
                plan,
                state,
                pupil_opd,
                pupil_amplitude,
                output,
            )
            return _DirectProperExecution(
                state,
                _DirectProperWorkspace(
                    context,
                    invocation,
                    pupil_opd,
                    pupil_amplitude,
                    output,
                ),
            )
        finally
            Backends._synchronize_prepared_device_execution_context!(context)
        end
    end
end

function CA.prepare_algorithm(
    ::Type{Declaration},
    configuration::Parent.ProperPropagationConfiguration{
        T,
        Prescription,
        Target,
    },
) where {
    T,
    Prescription,
    Target,
    Declaration<:ProperPropagation{T,Prescription,Target},
}
    plan = _plan(configuration)
    workspace = _prepare_direct_workspace(plan)
    return CA.PreparedAlgorithm(Declaration, plan, workspace, _formats(configuration))
end

function CA.process!(
    output,
    prepared::CA.PreparedAlgorithm{<:ProperPropagation},
    pupil_opd,
    pupil_amplitude,
)
    plan = prepared.plan
    workspace = prepared.workspace.workspace
    _validate_buffer(plan, pupil_opd, "pupil OPD")
    _validate_buffer(plan, pupil_amplitude, "pupil amplitude")
    _validate_buffer(plan, output, "Proper output")
    Backends._with_prepared_device_execution_context(workspace.context) do
        try
            copyto!(workspace.pupil_opd, pupil_opd)
            copyto!(workspace.pupil_amplitude, pupil_amplitude)
            _run_invocation!(workspace.invocation)
            copyto!(output, workspace.output)
        finally
            Backends._synchronize_prepared_device_execution_context!(
                workspace.context,
            )
        end
    end
    return output
end

function CA.reset!(prepared::CA.PreparedAlgorithm{<:ProperPropagation})
    execution = prepared.workspace
    workspace = execution.workspace
    Backends._with_prepared_device_execution_context(workspace.context) do
        try
            _reset_state!(execution.state)
            Parent.reset_proper_assets!(
                prepared.plan.prescription,
                workspace.invocation.assets,
            )
        finally
            Backends._synchronize_prepared_device_execution_context!(
                workspace.context,
            )
        end
    end
    return prepared
end


function AOG._prepare_algorithm_instance(
    ::Type{Declaration},
    configuration::Parent.ProperPropagationConfiguration{
        T,
        Prescription,
        Target,
    },
) where {
    T,
    Prescription,
    Target,
    Declaration<:ProperPropagation{T,Prescription,Target},
}
    plan = _plan(configuration)
    formats = _formats(configuration)
    return _ProperGraphPrototype{Declaration,typeof(plan),typeof(formats)}(
        plan,
        formats,
    )
end

function AOG._algorithm_port_contracts(
    ::Type{Declaration},
    prototype::_ProperGraphPrototype{Declaration},
) where {T,Prescription,Target,Declaration<:ProperPropagation{T,Prescription,Target}}
    formats = prototype.formats
    return (
        AOG._graph_port_contract(
            :pupil_opd,
            :input,
            :data,
            T,
            formats[1].shape,
            formats[1].schema,
            :column_major,
        ),
        AOG._graph_port_contract(
            :pupil_amplitude,
            :input,
            :data,
            T,
            formats[2].shape,
            formats[2].schema,
            :column_major,
        ),
        AOG._graph_port_contract(
            :output,
            :output,
            :data,
            T,
            formats[3].shape,
            formats[3].schema,
            :column_major,
        ),
    )
end

function AOG._bind_algorithm_instance(
    ::Type{Declaration},
    prototype::_ProperGraphPrototype{Declaration},
    inputs::NamedTuple,
    outputs::NamedTuple,
) where {Declaration<:ProperPropagation}
    state = _prepare_state(prototype.plan.rng_seed)
    invocation = _prepare_invocation(
        prototype.plan,
        state,
        inputs.pupil_opd,
        inputs.pupil_amplitude,
        outputs.output,
    )
    return _GraphBoundProperPropagation(
        prototype.plan,
        state,
        invocation,
        inputs.pupil_opd,
        inputs.pupil_amplitude,
        outputs.output,
    )
end

@inline function AOG._process_algorithm!(
    owner::_GraphBoundProperPropagation,
    outputs::NamedTuple,
    inputs::NamedTuple,
)
    inputs.pupil_opd === owner.pupil_opd || error(
        "prepared Proper pupil OPD binding changed",
    )
    inputs.pupil_amplitude === owner.pupil_amplitude || error(
        "prepared Proper pupil-amplitude binding changed",
    )
    outputs.output === owner.output || error(
        "prepared Proper output binding changed",
    )
    _run_invocation!(owner.invocation)
    return nothing
end

function AOG._reset_algorithm!(owner::_GraphBoundProperPropagation)
    _reset_state!(owner.state)
    Parent.reset_proper_assets!(owner.plan.prescription, owner.invocation.assets)
    return nothing
end

end # module AdaptiveOpticsProperHILCalculonAlgorithmsExt
