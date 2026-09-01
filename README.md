# AdaptiveOpticsProperHIL.jl

Cross-package HIL validation and benchmark harness for `AdaptiveOpticsSim.jl`
and `Proper.jl`.

This project intentionally keeps integration tests, GPU hardware scripts,
combined command-to-pixels benchmarks, and result artifacts out of the base
packages.

## Quick Start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=. scripts/validate_cpu.jl
julia --project=. scripts/benchmark_cpu.jl 5 1.0 64
julia --project=. scripts/benchmark_matrix.jl cpu check
```

## What This Validates

- OPD units, sign, centering, and orientation at the package boundary
- total-OPD handoff through `prop_add_phase`
- combined command-to-pixels HIL runtime shape
- a versioned lockstep TCP boundary for an external real-time controller
- CPU by default, optional AMDGPU and CUDA scripts when hardware is available

## Preferred Handoff

The primary HIL seam is total sampled residual OPD plus pupil:

```julia
psf, sampling = prop_run(science_model; payload=payload)
```

Use `PASSVALUE` only for upstream PROPER compatibility or parity harnesses.

For a repeated RTC/HIL loop, bind the Proper call once and reuse the same
payload and output buffers:

```julia
run_context = Proper.prepared_context(science_model, 1)
wavefront = Proper.prop_begin!(field_buffer, payload.diameter_m, 1.65f-6;
    beam_diam_fraction=1.0,
    context=run_context,
)
run = Proper.prepare_run(
    science_model;
    slot=1,
    activate_context=false,
    payload=payload,
    field=field_buffer,
    wavefront=wavefront,
    mask=mask_buffer,
    output=output_buffer,
    run_context=run_context,
)

psf, sampling = Proper.prop_run(run)
```

This bypasses per-frame keyword merging and model asset lookup. It requires a
native Julia keyword prescription; upstream-style `PASSVALUE` remains a
compatibility path on `prop_run`.

## AOS Algorithm Graphs

An ordinary Julia-native Proper prescription can be one optional node in an
`AdaptiveOpticsSim.AlgorithmGraphs` complete-frame graph. Construct its typed
configuration and node directly:

```julia
configuration = ProperPropagationConfiguration(
    science_prescription;
    resolution=128,
    pupil_resolution=128,
    output_rows=128,
    output_columns=128,
    diameter_m=8.0f0,
    wavelength_um=1.65f0,
)
node = proper_propagation_node(:science_camera, configuration)
```

The adapter prepares the prescription once and supplies `pupil_opd`,
`pupil_amplitude`, `diameter_m`, `field`, `wavefront`, `output`, and
`run_context` keyword bindings. `prepare_proper_assets` may be specialized for
a prescription that needs prepared masks, plans, or scratch; it receives both
the square propagation resolution and the declared rectangular output shape.
Those assets stay in the workspace owner rather than the immutable propagation
plan. `pupil_resolution` defaults to `resolution`; when it differs, the
prescription owns a prepared embedding or resampling from the square pupil
inputs into the square propagation grid. `output_rows` and `output_columns`
also default to `resolution`. When they differ, the prescription owns the
prepared crop, resampling, detector mapping, and scientific units of the
resulting observation.

The graph's `prepare_algorithm_graph(...; target=...)` call selects the exact
CPU, AMDGPU, or CUDA device. The scientific configuration therefore remains
portable across targets. Julia code remains the unrestricted composition path
for generated or conditional topology.

TOML graphs can use a Proper node without naming or evaluating Julia code from
the file. The application explicitly admits a prescription under one node-type
name:

```julia
node_types = merge(
    builtin_graph_node_types(),
    (spiders_proper_f32=proper_propagation_node_factory(
        spiders_prescription;
        T=Float32,
    ),),
)
definition = load_algorithm_graph(
    "spiders.toml";
    node_types=node_types,
    bindings=(pupil_opd=opd, pupil_amplitude=pupil),
)
```

The TOML node then uses `type = "spiders_proper_f32"`. This keeps the file
declarative and reviewable while the Julia application remains responsible for
the trusted prescription registry and large ndarray bindings.

The OPD, amplitude, field, output, Proper workspace, and graph target use one
element type, backend, and exact device. The graph never inserts a host/device
copy. A CPU RTC therefore consumes a GPU-generated product only through an
explicit application-owned copy or prepared handoff after the graph output
boundary.

Focused conformance environments live under `test/algorithm_graph`,
`test/algorithm_graph_amdgpu`, and `test/algorithm_graph_cuda`. The GPU fixtures
execute the same node with scalar indexing disabled and compare the explicitly
published host result with the CPU result.

For a real instrument loop, pass total residual OPD when the AO package owns
atmosphere, telescope, and DM composition. Pass a DM-only sampled surface only
when the Proper prescription intentionally owns the remaining optical error
budget and applies the DM surface itself.

## SPIDERS HIL Topology

SPIDERS is one instrument scenario in this package, not the package boundary.
The reusable boundary remains any prepared PROPER intensity model connected to
AdaptiveOpticsSim or to an AOS algorithm graph. `SpidersProper.jl` owns the
private SPIDERS optical prescription; this repository owns its cross-package
validation, detector handoff, RTC topology, and benchmark evidence.

[`spiders_llowfs_hil.toml`](examples/graphs/spiders_llowfs_hil.toml) and
[`spiders_scc_hil.toml`](examples/graphs/spiders_scc_hil.toml) record the two
external-RTC sensor topologies without embedding an RTC implementation:

```text
LLOWFS pupil OPD + amplitude -> trusted Proper prescription -> LLOWFS observation
SCC pupil OPD + amplitude    -> trusted Proper prescription -> SCC observation
```

The application captures each qualified or explicitly provisional prescription
and its complete physical configuration before loading the file:

```julia
node_types = merge(
    builtin_graph_node_types(),
    (
        spiders_llowfs_proper=
            proper_propagation_node_factory(llowfs_configuration),
        spiders_scc_proper=
            proper_propagation_node_factory(scc_configuration),
    ),
)

llowfs_definition = load_algorithm_graph(
    "examples/graphs/spiders_llowfs_hil.toml";
    node_types,
    bindings=(
        pupil_opd=llowfs_pupil_opd,
        pupil_amplitude=llowfs_pupil_amplitude,
    ),
)
scc_definition = load_algorithm_graph(
    "examples/graphs/spiders_scc_hil.toml";
    node_types,
    bindings=(
        pupil_opd=scc_pupil_opd,
        pupil_amplitude=scc_pupil_amplitude,
    ),
)
```

The graphs are intentionally separate because LLOWFS and SCC have different
acquisition rates. The application drives each prepared graph at its own
complete-frame model-time boundary; combining them in one simple graph would
incorrectly execute both prescriptions on every step.

The currently admitted legacy RTC products establish a 34-by-47 LLOWFS
observation and a 416-by-380 SCC observation. They do not establish a qualified
SPIDERS Proper prescription, propagation scale, flux normalization, or detector
response. Those values stay in the trusted Julia configurations instead of
appearing as anonymous guesses in TOML.

The Zemax-derived reflected-Lyot model can be validated with the optional
private-package environment:

```bash
julia --project=env/spiders -e 'using Pkg; Pkg.instantiate()'
julia --project=env/spiders scripts/validate_spiders_llowfs.jl 512 software
```

The direct handoff uses the deployed VENOMS array convention `(SizeX, SizeY)`:
the GoldEye sensor is represented as `(320, 256)` with a `(34, 47)` output ROI,
15 micrometre pixels, `psf_sampling=1`, and `binning=1`. A software ROI uses an
AdaptiveOpticsSim `FrameWindow`; a hardware ROI uses a physically cropped
detector input. Both paths perform a direct pixel copy from the PROPER grid and
no image resampling. The default centered ROI origin is only a validation
placeholder: use live VENOMS `OffsetX` and `OffsetY` for instrument matching.

For model development before the as-built optical prescription is available,
`provisional_spiders_h_regular_profile()` records a reviewable regular-H
starting point. It includes the deployed SpiderMan FPM-stage and Galil presets,
the science-wheel serial numbers and slot tables, and the dimensions reported
by the SPIDERS optical design. It is explicitly not a qualified instrument
configuration:

```julia
profile = provisional_spiders_h_regular_profile(Float32)

llowfs_configuration = provisional_spiders_llowfs_configuration(profile)
scc_configuration = provisional_spiders_chopped_scc_configuration(profile)

for claim in spiders_configuration_claims(scc_configuration)
    @info "SPIDERS assumption" name=claim.name evidence=claim.evidence current_choice=claim.current_choice question=claim.qualification_question
end
```

The current wiring assumption assigns wheel 2 to LLOWFS and wheel 3 to SCC.
SpiderMan contains no checked-in optical-arm label for either serial. The two
wheels have identical slot tables, so reversing that ownership does not change
the provisional monochromatic regular-H model; it changes only the hardware
identity. The normal filter is provisionally slot 5 (`1550-25nm`), inferred
from those tables and the local `LOWFS/filt5` calibration family. Reverse
the assumed wiring without changing package code:

```julia
profile = provisional_spiders_h_regular_profile(
    Float32;
    llowfs_wheel=3,
    scc_wheel=2,
)
```

`spiders_profile_claims(profile)` is the qualification checklist. In addition
to wheel ownership and the normal filter slot, it asks for the LLOWFS selector
state, SCC acquisition state, installed reference pinhole and position angle,
as-built tilt-Gaussian-vortex and apodizer maps, and qualified detector
mapping. The defaults use a 203 micrometre reference hole, the focal-plane and
zoomed-in LLOWFS selectors, an unblocked/fringed SCC path, and a zero-degree
reference-hole position angle. These are named placeholders, not observations
of current hardware state.

The profile's paper-backed values come from the regular-H optical design in
*Optical design of SPIDERS, the Subaru Pathfinder Instrument for Detecting
Exoplanets and Retrieving Spectra* (Proc. SPIE 12185,
DOI `10.1117/12.2629644`). The local calibration arrays establish product
shapes and provide future validation targets; a reference image or interaction
matrix is not an optical transmission map and is not silently substituted for
one.

The three configuration builders provide executable first-order Proper
prescriptions for the existing TOML graphs. By default they embed a 128-by-128
AOS pupil in a 512-by-512 propagation grid, leaving enough padded field for the
off-axis SCC reference hole. They then apply:

- a binary annular surrogate for the documented apodizer geometry;
- a unit-amplitude charge-4 vortex with explicit placeholder tilt/Gaussian
  central phase parameters inside the documented 5.7-lambda/D regular-H mask;
- the documented FPM-to-L2 distance and L2 focal length;
- a binary Lyot main aperture, enlarged orthogonal spider surrogate, and the
  selected off-axis reference hole;
- the complementary rejected-light path for LLOWFS or transmitted main-plus-
  reference path for SCC; and
- prepared camera rotation/resampling into the legacy product shape.

`provisional_spiders_scc_configuration` freezes the reference path open or
closed and is useful for optical debugging. The HIL path should normally use
`provisional_spiders_chopped_scc_configuration`. Its persistent chopper state
alternates one complete reference-open fringed exposure with one complete
reference-closed unfringed exposure. The immutable plan owns the initial phase;
the prepared state owns only the successfully completed frame sequence. Reset
restores the initial phase.

The pupil embedding, propagation, mask maps, and detector resampling use
preallocated target-native buffers. A warmed CPU graph step allocates no Julia
heap memory. The same prescription uses Proper's CPU/GPU-native operations.
The local AMDGPU result agrees with the CPU result to less than `1.8e-6`
maximum absolute intensity error for the static paths with scalar indexing
disabled. Both chopped phases also pass the shared CPU/GPU numerical tolerance;
CUDA execution of these provisional prescriptions remains to be revalidated.

This is a topology-preserving simulation surrogate, not an as-built optical
model. It is monochromatic and emits relative intensity. It does not yet model
throughput, exposure, detector gain/noise/nonlinearity, bad pixels, saturation,
distortion, measured aberrations, exact fold signs, or calibrated pixel
registration. `spiders_configuration_claims(configuration)` combines all eight
instrument-profile questions with four prescription-specific numerical
assumptions. A chopped configuration adds the frame-synchronization assumption,
so none of these defaults can disappear into implementation code.

Register the two prepared configurations with the existing graph files:

```julia
node_types = merge(
    builtin_graph_node_types(),
    (
        spiders_llowfs_proper=
            proper_propagation_node_factory(llowfs_configuration),
        spiders_scc_proper=
            proper_propagation_node_factory(scc_configuration),
    ),
)
```

Drive the SCC graph once per complete camera exposure. After a successful step,
query the exact frame identity before publishing it:

```julia
timestamp = step_graph_at!(scc_graph, scc_model_time_driver)
status = spiders_chopper_frame_status(scc_graph, Val(:scc_optics))
frame = graph_output(scc_graph, :scc_observation)
sequence = spiders_chopper_sequence(status)
phase = spiders_chopper_phase(status)
```

The application or PipeWire adapter publishes `frame` together with
`timestamp`, `sequence`, and `phase`. It owns the transport metadata and any
legacy CRED2 tag-pixel convention; the optical prescription does not overwrite
a science pixel. The deployed sorter may then classify and difference the raw
alternating frames. An in-process reference path can instead retain the latest
frame of each phase and form `fringed - unfringed` in a separate node.

The current model assumes the blade transition occurs outside each exposure.
A future rolling-shutter strategy can retain the same atomic output boundary
while integrating each row over its own exposure interval, including the
time-varying chopper aperture, turbulence, and applied DM state. No partially
read detector frame should become graph-visible.

The primary closed-loop HIL path is therefore:

```text
AOS -> complete LLOWFS/SCC observations -> PipeWireAO RTC
    <- complete 468-coordinate PDM command <-
```

The surrounding AOS simulation applies that returned command through the
measured DM model and renders the next path OPDs. The SPIDERS file contains no
pixel estimator, reconstructor, temporal controller, command mixer, or DM
policy. An in-process reference RTC may be composed separately for regression
testing, but it is not part of this HIL topology.

## GPU Validation

AMDGPU on the local machine:

```bash
julia --project=env/amdgpu -e 'using Pkg; Pkg.instantiate()'
julia --project=env/amdgpu scripts/validate_amdgpu.jl
julia --project=env/amdgpu scripts/benchmark_amdgpu.jl 5 1.0 64
```

CUDA on `spiders`:

```bash
./scripts/run_cuda_spiders.sh 5 1.0 64
```

`run_cuda_spiders.sh` stages a disposable sibling workspace by default so it
does not modify a dirty `AdaptiveOpticsSim.jl` checkout on the remote host.
Proper is resolved from its pinned GitHub source revision.

### CUDA Graph prepared propagation

For a device-resident prepared optical model, load CUDA and capture the warmed
propagation once during HIL setup:

```julia
using AdaptiveOpticsProperHIL
using CUDA

graph_model = capture_cuda_optical_model(optical_model; warmup=10)

# Update the existing input arrays in place, then enqueue one propagation.
propagate_optics!(graph_model)
```

The CUDA extension is optional; CUDA is not a hard dependency of this package.
The HIL wrapper delegates capture and launch to Proper 0.3's general CUDA Graph
executor, so the same prepared propagation also works without AdaptiveOpticsSim.
`propagate_optics!` on the captured model launches asynchronously so later
same-stream device work can follow without an intermediate host barrier. Call
`CUDA.synchronize()` only where the application needs completion on the host.
The prepared arrays, FFT plans, grid, and control flow must retain their
captured identities. Recapture after replacing buffers or changing propagation
topology.

The private SPIDERS-only experiment compares ordinary and graph-backed PROPER
propagation while excluding the detector, camera, DM, and host/device transfer:

```bash
julia --project=env/spiders_cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=env/spiders_cuda scripts/benchmark_spiders_cuda_graph.jl 1024 100
```

On 2026-08-31, one 100-sample bracketed run on an NVIDIA GeForce RTX
3050 Ti Laptop GPU (Julia 1.12.6, CUDA.jl 6.3.1) produced:

| Synchronized propagation | Median | p95 | Mean rate | Host bytes/call |
|---|---:|---:|---:|---:|
| Direct, before graph | 13.850 ms | 14.344 ms | 71.651 Hz | 139,152 |
| CUDA Graph | 13.744 ms | 13.910 ms | 72.800 Hz | 48 |
| Direct, after graph | 14.002 ms | 14.132 ms | 71.398 Hz | 139,152 |

The graph launch alone allocated zero host bytes; the remaining 48 bytes came
from the explicit synchronization used by the benchmark. Relative L2 error
against CPU was `2.9819694e-6` for the flat input and `2.963033e-6` after an
in-place 10 nm OPD change. This supports using a graph to remove per-frame host
launch construction and modestly tighten latency. It does not change the main
1024-grid limit: the optical FFT workload still holds this GPU near 73 Hz. The
run is host-specific integration evidence, not a portable performance claim.

## Current Reference Results

The checked-in `results/*.json` files record short warmed steady-state
BenchmarkTools runs at resolution 64. They are not performance claims for a
final HIL system; they are regression artifacts for this integration seam.

- CPU: `results/cpu_hil_benchmark.json`
- AMDGPU: `results/amdgpu_hil_benchmark.json`
- CUDA: `results/cuda_hil_benchmark.json`

## Versioned HIL Benchmark Matrix

`contracts/hil-benchmark-matrix-v1.toml` is the maintained latency contract for
external-RTC HIL work. It covers CPU and AMDGPU across CCD, CMOS, EMCCD, and
SAPHIRA-style detector models with Shack-Hartmann and pyramid WFS paths, plus
the Proper science arm. Each GPU sample includes backend synchronization.

Run, record, or check a baseline with:

```bash
julia --project=. scripts/benchmark_matrix.jl cpu run
julia --project=. scripts/benchmark_matrix.jl cpu record
julia --project=. scripts/benchmark_matrix.jl cpu check

julia --project=env/amdgpu scripts/benchmark_matrix.jl amdgpu run
julia --project=env/amdgpu scripts/benchmark_matrix.jl amdgpu record
julia --project=env/amdgpu scripts/benchmark_matrix.jl amdgpu check
```

Optional trailing arguments are `samples`, `seconds`, and `warmup`. Regression
checks use the per-backend median, p95, allocation, and byte thresholds declared
in the contract. Baseline timings are host-specific evidence, not portable
performance promises.

## External RTC Harness

`contracts/external-rtc-wire-v1.toml` defines the fixed-size binary protocol.
Each cycle senses the current residual, sends one slopes frame, receives the
same-sequence command frame, validates it, and stages the command for the next
AO sample. Only one request is outstanding, so there is no hidden queue or
batching policy. TCP_NODELAY is enabled.

For local bring-up, start the deterministic mock RTC and the benchmark in
separate processes. The default 64-pixel/8-actuator context uses 512 slopes and
66 commands:

```bash
AOS_RTC_SLOPES=512 AOS_RTC_COMMANDS=66 \
    julia --project=. scripts/mock_external_rtc.jl

julia --project=. scripts/benchmark_external_rtc.jl
```

The benchmark records warmed cycle and transport latency distributions in
fixed-range HdrHistogram.jl histograms, reports p50/p90/p95/p99/max overall and
per trial, retains compressed mergeable histogram payloads, tracks deadline
misses, and runs a separate steady-state allocation probe. P99 requires at
least 1,000 samples per trial; p99.9 is emitted at 10,000 or more. Exact raw
samples remain enabled by default and can be disabled for long runs with
`AOS_RTC_RETAIN_RAW_SAMPLES=false`. Histogram range and precision can be set
with `AOS_RTC_HISTOGRAM_MAX_NS` and `AOS_RTC_HISTOGRAM_SIGFIGS`.

HdrHistogram.jl is resolved from the maintained GitHub source pinned in the
project rather than from a local checkout or registry release. Histogram
recording is raw and uncorrected. This remains a closed-loop request/response
test: it measures response-dependent HIL latency, not open-loop RTC capacity,
and histogram correction is not used as a substitute for a schedule-preserving
load generator. A completed-cycle deadline is observed and counted; the
synchronous TCP read is not a failure watchdog, so a production launcher must
close or restart a failed peer.

The AO context currently defaults to CPU for HIL. A GPU context necessarily
crosses a host/device boundary to reach a conventional network RTC and should
be measured as a distinct deployment mode. Device-resident simulation remains
the preferred GPU topology.

## Allocation Decision

The CPU combined frame path is zero-allocation after warmup in the current
reference harness. GPU host allocation counts are still non-zero because some
CUDA.jl and AMDGPU.jl wrapper/synchronization paths allocate on the host; track
those separately from CPU HIL loop ownership. The external TCP transaction also
has its own small host allocation cost in Julia's socket layer; the external
benchmark reports that separately and does not weaken the zero-allocation
in-process CPU optical path.
