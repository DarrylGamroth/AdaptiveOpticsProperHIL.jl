const DEFAULT_BENCHMARK_MATRIX = normpath(joinpath(
    @__DIR__, "..", "contracts", "hil-benchmark-matrix-v1.toml"))

abstract type AbstractBenchmarkOutputAssembler end

struct DirectBenchmarkOutput <: AbstractBenchmarkOutputAssembler end

struct ShackHartmannBenchmarkOutput{A} <: AbstractBenchmarkOutputAssembler
    image::A
    n_lenslets::Int
end

mutable struct WFSDetectorBenchmarkContext{S,H,C,A}
    scenario::S
    command_host::H
    command_buffer::C
    assembler::A
    step_index::Int
end

struct ProperScienceBenchmarkContext{C}
    hil::C
end

function load_benchmark_matrix(path::AbstractString=DEFAULT_BENCHMARK_MATRIX)
    matrix = TOML.parsefile(path)
    get(matrix, "schema_version", 0) == 1 ||
        throw(ArgumentError("unsupported benchmark matrix schema in $(path)"))
    axes = matrix["axes"]
    cases = matrix["cases"]
    isempty(cases) && throw(ArgumentError("benchmark matrix must define at least one case"))
    ids = String[String(case["id"]) for case in cases]
    allunique(ids) || throw(ArgumentError("benchmark matrix case ids must be unique"))
    for axis in ("backends", "detectors", "wfs", "science_arms")
        isempty(axes[axis]) && throw(ArgumentError("benchmark matrix axis $(axis) must not be empty"))
    end
    return matrix
end

function benchmark_baseline_path(matrix::AbstractDict, backend::AbstractString;
    matrix_path::AbstractString=DEFAULT_BENCHMARK_MATRIX)
    relative_path = matrix["baselines"][lowercase(backend)]
    return normpath(joinpath(dirname(matrix_path), "..", relative_path))
end

@inline _benchmark_sensor(::Val{:ccd}, ::Type{T}, backend) where {T} =
    AdaptiveOpticsSim.CCDSensor(T=T)
@inline _benchmark_sensor(::Val{:cmos}, ::Type{T}, backend) where {T} =
    AdaptiveOpticsSim.CMOSSensor(T=T, backend=backend)
@inline _benchmark_sensor(::Val{:emccd}, ::Type{T}, backend) where {T} =
    AdaptiveOpticsSim.EMCCDSensor(T=T)
@inline _benchmark_sensor(::Val{:saphira}, ::Type{T}, backend) where {T} =
    AdaptiveOpticsSim.HgCdTeAvalancheArraySensor(
        T=T, sampling_mode=AdaptiveOpticsSim.SingleRead())

function _benchmark_sensor(name::AbstractString, ::Type{T}, backend) where {T}
    symbol = Symbol(lowercase(name))
    symbol in (:ccd, :cmos, :emccd, :saphira) ||
        throw(ArgumentError("unsupported detector matrix value $(name)"))
    return _benchmark_sensor(Val(symbol), T, backend)
end

function _benchmark_wfs(::Val{:shack_hartmann}, tel, backend, ::Type{T}, defaults) where {T}
    return AdaptiveOpticsSim.ShackHartmannWFS(
        tel;
        n_lenslets=Int(defaults["n_lenslets"]),
        n_pix_subap=div(Int(defaults["resolution"]), Int(defaults["n_lenslets"])),
        diffraction_padding=2,
        threshold=zero(T),
        mode=AdaptiveOpticsSim.Diffractive(),
        T=T,
        backend=backend,
    )
end

function _benchmark_wfs(::Val{:pyramid}, tel, backend, ::Type{T}, defaults) where {T}
    return AdaptiveOpticsSim.PyramidWFS(
        tel;
        pupil_samples=Int(defaults["pupil_samples"]),
        modulation=one(T),
        modulation_points=4,
        diffraction_padding=2,
        threshold=zero(T),
        mode=AdaptiveOpticsSim.Diffractive(),
        T=T,
        backend=backend,
    )
end

function _benchmark_wfs(name::AbstractString, tel, backend, ::Type{T}, defaults) where {T}
    symbol = Symbol(lowercase(name))
    symbol in (:shack_hartmann, :pyramid) ||
        throw(ArgumentError("unsupported WFS matrix value $(name)"))
    return _benchmark_wfs(Val(symbol), tel, backend, T, defaults)
end

function _benchmark_output_assembler(wfs, scenario)
    hasproperty(wfs, :params) &&
        hasproperty(getproperty(wfs, :params), :n_lenslets) ||
        return DirectBenchmarkOutput()
    cube = AdaptiveOpticsSim.wfs_frame(scenario)
    n_lenslets = wfs.params.n_lenslets
    image = similar(cube, eltype(cube),
        n_lenslets * size(cube, 2), n_lenslets * size(cube, 3))
    fill!(image, zero(eltype(image)))
    return ShackHartmannBenchmarkOutput(image, n_lenslets)
end

function build_benchmark_case(case_spec::AbstractDict, backend;
    defaults::AbstractDict, T::Type{<:AbstractFloat}=Float32)
    kind = case_spec["kind"]
    kind == "proper_science" && return ProperScienceBenchmarkContext(build_hil_context(
        backend=backend,
        T=T,
        resolution=Int(defaults["resolution"]),
        n_act=Int(defaults["n_act"]),
        rng_seed=Int(defaults["rng_seed"]),
    ))
    kind == "wfs_detector" || throw(ArgumentError("unsupported benchmark case kind $(kind)"))

    resolution = Int(defaults["resolution"])
    tel = AdaptiveOpticsSim.Telescope(
        resolution=resolution,
        diameter=T(8),
        sampling_time=T(1e-3),
        central_obstruction=T(0.1),
        T=T,
        backend=backend,
    )
    src = AdaptiveOpticsSim.Source(band=:I, magnitude=0.0, T=T)
    atm = AdaptiveOpticsSim.KolmogorovAtmosphere(
        tel; r0=T(0.2), L0=T(25), T=T, backend=backend)
    dm = AdaptiveOpticsSim.DeformableMirror(
        tel;
        n_act=Int(defaults["n_act"]),
        influence_width=T(0.3),
        T=T,
        backend=backend,
    )
    wfs = _benchmark_wfs(case_spec["wfs"], tel, backend, T, defaults)
    sensor = _benchmark_sensor(case_spec["detector"], T, backend)
    detector = AdaptiveOpticsSim.Detector(
        noise=AdaptiveOpticsSim.NoiseNone(),
        integration_time=one(T),
        qe=one(T),
        binning=1,
        sensor=sensor,
        T=T,
        backend=backend,
    )
    sim = AdaptiveOpticsSim.AOSimulation(tel, src, atm, dm, wfs)
    branch = AdaptiveOpticsSim.ControlLoopBranch(
        :main,
        sim,
        AdaptiveOpticsSim.NullReconstructor();
        wfs_detector=detector,
        rng=AdaptiveOpticsSim.runtime_rng(Int(defaults["rng_seed"])),
    )
    config = AdaptiveOpticsSim.SingleControlLoopConfig(
        name=Symbol(case_spec["id"]),
        branch_label=:main,
        outputs=AdaptiveOpticsSim.RuntimeOutputRequirements(
            slopes=true, wfs_pixels=true, science_pixels=false),
    )
    scenario = AdaptiveOpticsSim.build_control_loop_scenario(config, branch)
    AdaptiveOpticsSim.prepare!(scenario)
    command_buffer = similar(AdaptiveOpticsSim.command(scenario))
    command_host = Vector{T}(undef, length(command_buffer))
    fill!(command_host, zero(T))
    copyto!(command_buffer, command_host)
    assembler = _benchmark_output_assembler(wfs, scenario)
    return WFSDetectorBenchmarkContext(
        scenario, command_host, command_buffer, assembler, 0)
end

@inline function _assemble_benchmark_output!(
    assembler::ShackHartmannBenchmarkOutput, scenario)
    return AdaptiveOpticsSim.shack_hartmann_detector_image!(
        assembler.image,
        AdaptiveOpticsSim.wfs_frame(scenario),
        assembler.n_lenslets,
    )
end

@inline _assemble_benchmark_output!(::DirectBenchmarkOutput, scenario) =
    AdaptiveOpticsSim.wfs_frame(scenario)

function benchmark_case_step!(ctx::WFSDetectorBenchmarkContext)
    ctx.step_index += 1
    T = eltype(ctx.command_host)
    phase = T(0.013) * T(ctx.step_index)
    @inbounds for i in eachindex(ctx.command_host)
        ctx.command_host[i] = T(5e-9) * sin(phase + T(0.007) * T(i))
    end
    copyto!(ctx.command_buffer, ctx.command_host)
    AdaptiveOpticsSim.set_command!(ctx.scenario, ctx.command_buffer)
    AdaptiveOpticsSim.sense!(ctx.scenario)
    output = _assemble_benchmark_output!(ctx.assembler, ctx.scenario)
    AdaptiveOpticsSim.synchronize_backend!(AdaptiveOpticsSim.execution_style(output))
    return output
end

function benchmark_case_step!(ctx::ProperScienceBenchmarkContext)
    output, _ = hil_step!(ctx.hil)
    AdaptiveOpticsSim.synchronize_backend!(AdaptiveOpticsSim.execution_style(output))
    return output
end

@inline benchmark_case_output(ctx::WFSDetectorBenchmarkContext) =
    _assemble_benchmark_output!(ctx.assembler, ctx.scenario)
@inline benchmark_case_output(ctx::ProperScienceBenchmarkContext) = ctx.hil.output_buffer

function _loaded_module(package_name::AbstractString)
    for (pkgid, mod) in Base.loaded_modules
        pkgid.name == package_name && return mod
    end
    return nothing
end

function _benchmark_backend(name::AbstractString)
    normalized = lowercase(name)
    normalized == "cpu" && return AdaptiveOpticsSim.CPUBackend()
    normalized == "amdgpu" || throw(ArgumentError("unsupported benchmark backend $(name)"))
    amdgpu = _loaded_module("AMDGPU")
    amdgpu === nothing && throw(ArgumentError(
        "AMDGPU must be loaded before running the AMDGPU benchmark matrix"))
    functional = Base.invokelatest(getfield(amdgpu, :functional))
    functional || throw(ArgumentError("AMDGPU is not functional on this host"))
    return AdaptiveOpticsSim.AMDGPUBackend()
end

function _benchmark_device(backend::AbstractString)
    lowercase(backend) == "cpu" && return isempty(Sys.cpu_info()) ? "unknown" : Sys.cpu_info()[1].model
    amdgpu = _loaded_module("AMDGPU")
    amdgpu === nothing && return "unknown"
    return string(Base.invokelatest(getfield(amdgpu, :device)))
end

function _benchmark_stats(trial::BenchmarkTools.Trial)
    times = Float64.(trial.times)
    return Dict{String,Any}(
        "median_ns" => median(times),
        "mean_ns" => mean(times),
        "min_ns" => minimum(times),
        "max_ns" => maximum(times),
        "p95_ns" => quantile(times, 0.95),
        "samples" => length(times),
        "median_allocs" => Float64(median(trial.allocs)),
        "median_bytes" => Float64(median(trial.memory)),
    )
end

function _benchmark_case_result(case_spec, backend, defaults, samples, seconds, warmup)
    context = build_benchmark_case(case_spec, backend; defaults=defaults)
    for _ in 1:warmup
        benchmark_case_step!(context)
    end
    trial = @benchmark benchmark_case_step!($context) samples=Int(samples) seconds=Float64(seconds) evals=1
    output = benchmark_case_step!(context)
    host_output = Array(output)
    return Dict{String,Any}(
        "id" => case_spec["id"],
        "kind" => case_spec["kind"],
        "wfs" => get(case_spec, "wfs", "none"),
        "detector" => get(case_spec, "detector", "none"),
        "stats" => _benchmark_stats(trial),
        "output" => Dict{String,Any}(
            "size" => collect(size(output)),
            "checksum" => sum(Float64, host_output),
        ),
    )
end

function run_benchmark_matrix(backend_name::AbstractString;
    matrix_path::AbstractString=DEFAULT_BENCHMARK_MATRIX,
    samples::Union{Nothing,Integer}=nothing,
    seconds::Union{Nothing,Real}=nothing,
    warmup::Union{Nothing,Integer}=nothing,
    case_ids::Union{Nothing,AbstractVector{<:AbstractString}}=nothing)
    matrix = load_benchmark_matrix(matrix_path)
    normalized_backend = lowercase(backend_name)
    normalized_backend in matrix["axes"]["backends"] ||
        throw(ArgumentError("backend $(backend_name) is not part of this matrix"))
    backend = _benchmark_backend(normalized_backend)
    defaults = matrix["defaults"]
    resolved_samples = isnothing(samples) ? Int(defaults["samples"]) : Int(samples)
    resolved_seconds = isnothing(seconds) ? Float64(defaults["seconds"]) : Float64(seconds)
    resolved_warmup = isnothing(warmup) ? Int(defaults["warmup"]) : Int(warmup)
    resolved_samples > 0 || throw(ArgumentError("samples must be positive"))
    resolved_seconds > 0 || throw(ArgumentError("seconds must be positive"))
    resolved_warmup >= 0 || throw(ArgumentError("warmup must be non-negative"))
    selected_ids = isnothing(case_ids) ? nothing : Set(String.(case_ids))
    cases = filter(matrix["cases"]) do case_spec
        normalized_backend in case_spec["backends"] &&
            (isnothing(selected_ids) || case_spec["id"] in selected_ids)
    end
    isempty(cases) && throw(ArgumentError("no benchmark cases matched the requested selection"))
    results = map(cases) do case_spec
        _benchmark_case_result(
            case_spec, backend, defaults, resolved_samples, resolved_seconds, resolved_warmup)
    end
    return Dict{String,Any}(
        "schema_version" => matrix["schema_version"],
        "suite_id" => matrix["suite_id"],
        "suite_version" => matrix["suite_version"],
        "backend" => normalized_backend,
        "timestamp_utc" => Dates.format(
            Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sss"),
        "policy" => "warmed steady-state command-to-pixels; evals=1; GPU synchronized; compilation excluded",
        "environment" => Dict{String,Any}(
            "julia_version" => string(VERSION),
            "julia_threads" => Threads.nthreads(),
            "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
            "device" => _benchmark_device(normalized_backend),
            "adaptive_optics_sim_version" => string(Base.pkgversion(AdaptiveOpticsSim)),
            "proper_version" => string(Base.pkgversion(Proper)),
        ),
        "parameters" => Dict{String,Any}(
            "samples" => resolved_samples,
            "seconds" => resolved_seconds,
            "warmup" => resolved_warmup,
            "resolution" => Int(defaults["resolution"]),
        ),
        "cases" => results,
    )
end

function write_benchmark_result(path::AbstractString, result::AbstractDict)
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, result)
    end
    return path
end

function _result_cases(result::AbstractDict)
    return Dict(String(case["id"]) => case for case in result["cases"])
end

function compare_benchmark_results(candidate::AbstractDict, baseline::AbstractDict,
    matrix::AbstractDict)
    candidate["suite_id"] == baseline["suite_id"] == matrix["suite_id"] ||
        throw(ArgumentError("benchmark suite ids do not match"))
    candidate["suite_version"] == baseline["suite_version"] == matrix["suite_version"] ||
        throw(ArgumentError("benchmark suite versions do not match"))
    candidate["backend"] == baseline["backend"] ||
        throw(ArgumentError("candidate and baseline backends do not match"))
    backend = String(candidate["backend"])
    thresholds = matrix["thresholds"][backend]
    candidate_cases = _result_cases(candidate)
    baseline_cases = _result_cases(baseline)
    keys(candidate_cases) == keys(baseline_cases) ||
        throw(ArgumentError("candidate and baseline case sets do not match"))

    passed = true
    comparisons = Dict{String,Any}[]
    for id in sort!(collect(keys(candidate_cases)))
        candidate_stats = candidate_cases[id]["stats"]
        baseline_stats = baseline_cases[id]["stats"]
        median_limit = baseline_stats["median_ns"] * thresholds["median_ratio"]
        p95_limit = baseline_stats["p95_ns"] * thresholds["p95_ratio"]
        alloc_limit = ceil(
            baseline_stats["median_allocs"] * thresholds["allocation_ratio"] +
                thresholds["max_extra_allocs"])
        byte_limit = ceil(
            baseline_stats["median_bytes"] * thresholds["byte_ratio"] +
                thresholds["max_extra_bytes"])
        failures = String[]
        candidate_stats["median_ns"] <= median_limit || push!(failures, "median_ns")
        candidate_stats["p95_ns"] <= p95_limit || push!(failures, "p95_ns")
        candidate_stats["median_allocs"] <= alloc_limit || push!(failures, "median_allocs")
        candidate_stats["median_bytes"] <= byte_limit || push!(failures, "median_bytes")
        case_passed = isempty(failures)
        passed &= case_passed
        push!(comparisons, Dict{String,Any}(
            "id" => id,
            "passed" => case_passed,
            "failures" => failures,
            "median_ratio" => candidate_stats["median_ns"] / baseline_stats["median_ns"],
            "p95_ratio" => candidate_stats["p95_ns"] / baseline_stats["p95_ns"],
            "median_allocs" => candidate_stats["median_allocs"],
            "allocation_limit" => alloc_limit,
            "median_bytes" => candidate_stats["median_bytes"],
            "byte_limit" => byte_limit,
        ))
    end
    return Dict{String,Any}(
        "passed" => passed,
        "backend" => backend,
        "suite_version" => matrix["suite_version"],
        "comparisons" => comparisons,
    )
end
