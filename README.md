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

SPIDERS instrument definitions, optical prescriptions, simulation profiles and
calibration recipes live in the instrument checkout. Use its shared build, run
and calibration commands. This package provides reusable Proper and detector-HIL
adapters. [AdaptiveOpticsSimPipeWireHIL](https://github.com/DarrylGamroth/AdaptiveOpticsSimPipeWireHIL)
provides the PipeWireAO transport.

For instrument regression checks, run `test/spiders-optics.jl` and
`test/spiders_dual_performance.jl` from the instrument checkout. They cover
optical validation and captured camera-to-DM exchange, respectively. The
LLOWFS and SCC branches retain separate complete-frame acquisition rates.

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
The HIL wrapper delegates capture and launch to Proper's general CUDA Graph
executor, so the same prepared propagation also works without AdaptiveOpticsSim.
`propagate_optics!` on the captured model launches asynchronously so later
same-stream device work can follow without an intermediate host barrier. Call
`CUDA.synchronize()` only where the application needs completion on the host.
The prepared arrays, FFT plans, grid, and control flow must retain their
captured identities. Recapture after replacing buffers or changing propagation
topology.

The historical SPIDERS experiment compared ordinary and graph-backed PROPER
propagation and excluded the detector, camera, DM, and host/device transfer.
Use the instrument repository's current validation selectors for new runs.

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
