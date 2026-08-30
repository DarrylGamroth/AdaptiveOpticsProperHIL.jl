# Validation Plan

Status: active

## Scope

This project validates the HIL seam between `AdaptiveOpticsSim.jl` and
`Proper.jl` without adding cross-package test burden to either source package.

## Tracks

1. Deterministic OPD convention tests for units, sign, centering, and
   orientation.
2. Integration regression tests for the total-OPD handoff through
   `prop_add_phase`.
3. Versioned command-to-detector-pixels benchmarks across CCD, CMOS, EMCCD,
   and SAPHIRA-style detectors with Shack-Hartmann and pyramid WFS paths.
4. Combined command-to-Proper-science benchmarks on CPU and AMDGPU, with CUDA
   retained as an optional target when suitable hardware is available.
5. Handoff decision: total residual OPD is primary; sampled DM surface remains
   available when the Proper prescription intentionally owns the DM surface.
6. External RTC lockstep validation through the versioned fixed-size TCP
   contract in `contracts/external-rtc-wire-v1.toml`.
7. `AdaptiveOpticsSim.AlgorithmGraphs` execution of the optional AOS-native
   Proper propagation node with exact CPU, AMDGPU, and CUDA target admission,
   including declaration through the versioned TOML graph loader.
8. Declarative, independently driven SPIDERS external-RTC LLOWFS and SCC
   topologies with rectangular observations and application-captured qualified
   or explicitly provisional prescription configurations.

## Handoff Contract

The default cross-package contract is:

- `AdaptiveOpticsSim.jl` consumes raw RTC actuator commands.
- `AdaptiveOpticsSim.jl` composes atmosphere, telescope, controllable optics,
  and DM shape into a sampled total residual OPD map in meters.
- `Proper.jl` consumes that OPD map plus the pupil mask through a native
  keyword payload and applies it with `prop_add_phase`.
- `PASSVALUE` is not used in native integration code; it remains a compatibility
  adapter for upstream PROPER-style models.
- The repeated frame loop uses `Proper.prepare_run` plus `Proper.prop_run` so
  static model assets, slot lookup, and keyword merging
  are resolved before entering the loop.
- The Proper graph node binds the exact graph OPD, pupil-amplitude, and output
  arrays after topology admission. The pupil and wavefront grid is square; the
  declared output may be rectangular. Proper-specific scratch is prepared
  separately through `prepare_proper_assets`, which receives the propagation
  and output shapes; it is not retained in the immutable propagation plan.
- The square pupil inputs may use `pupil_resolution` distinct from the square
  propagation `resolution`. The prescription owns the prepared, target-native
  embedding or resampling; the graph adapter never assumes that padding is a
  crop of the caller's pupil.
- One prepared Proper node is homogeneous in numeric type, backend, and exact
  device. Mixed-type graph nodes remain valid elsewhere, but any GPU-to-host
  publication is an explicit boundary operation.

Use a DM-only surface only for prescriptions that deliberately model the rest
of the optical train inside Proper and need to place the DM at a specific
conjugate plane.

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. scripts/validate_cpu.jl
julia --project=. scripts/benchmark_cpu.jl
julia --project=. scripts/benchmark_matrix.jl cpu check
julia --project=env/amdgpu scripts/validate_amdgpu.jl
julia --project=env/amdgpu scripts/benchmark_amdgpu.jl
julia --project=env/amdgpu scripts/benchmark_matrix.jl amdgpu check
./scripts/run_cuda_spiders.sh
julia --project=. scripts/mock_external_rtc.jl
julia --project=. scripts/benchmark_external_rtc.jl
```

The optional graph-node test has its own environment:

```bash
julia --project=test/algorithm_graph -e 'using Pkg; Pkg.instantiate()'
julia --project=test/algorithm_graph test/algorithm_graph/runtests.jl

julia --project=test/algorithm_graph_amdgpu -e 'using Pkg; Pkg.instantiate()'
julia --project=test/algorithm_graph_amdgpu test/algorithm_graph_amdgpu/runtests.jl

julia --project=test/algorithm_graph_cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=test/algorithm_graph_cuda test/algorithm_graph_cuda/runtests.jl
```

It develops this package and the sibling AdaptiveOpticsSim and Proper
checkouts. The CPU test verifies Julia- and TOML-declared graph execution,
exact buffer admission, inference, and zero warmed allocation. The GPU tests
additionally disable scalar indexing, compare the accelerator result with the
CPU result, verify exact-device residency and graph buffer identity, reject a
host buffer at accelerator admission, and publish the result through an
explicit device-to-host copy after the graph boundary.

## Current Results

The AOS-native Proper graph node and provisional SPIDERS prescriptions passed
their focused matrix on 2026-08-29:

- CPU: 165/165 checks on Julia 1.12.7. These include equivalent direct-Julia
  and TOML-defined topology, distinct pupil and propagation grids, rectangular
  output, provisional-profile provenance, both executable SPIDERS paths,
  synchronized fringed/unfringed SCC alternation, model-time/phase alignment,
  reset and exhaustion behavior, response to a deterministic pupil tilt,
  inference, and zero warmed graph-step allocation.
- AMDGPU: 63/63 checks on an AMD Radeon Graphics `gfx1030` device with
  AMDGPU.jl 2.8 and scalar indexing disabled. Both 512-grid SPIDERS paths keep
  every prepared array on the selected device. The static paths agree with CPU
  to less than `1.8e-6` maximum absolute intensity error, and both chopped
  phases pass the shared CPU/GPU numerical tolerance.
- CUDA: the earlier 18/18 generic-node checks passed on an NVIDIA GeForce RTX
  3050 Ti Laptop GPU with CUDA.jl 6.3.1 and scalar indexing disabled. The
  shared CUDA fixture now includes the static and chopped provisional SPIDERS
  paths, but that expanded fixture has not been rerun on CUDA hardware at this
  revision.

These checks establish functional execution, allocation behavior, and exact
residency for the tested configurations. They validate the declarative
SPIDERS topology boundary and the explicitly provisional optical surrogate.
They are not throughput, latency, detector-response, calibration-match, or
as-built SPIDERS prescription evidence.

The original combined AO/Proper seam results remain under `results/`:

| Backend | Median runtime | Median allocations | Median bytes |
| --- | ---: | ---: | ---: |
| CPU | 540316 ns | 0 | 0 |
| AMDGPU | 953744 ns | 3828 | 168344 |
| CUDA (historical) | 789930 ns | 2492 | 87712 |

These results exclude Julia cold-start and TTFx. They are useful for regression
tracking, not as final latency claims.

The maintained v1 matrix contract is
`contracts/hil-benchmark-matrix-v1.toml`; its machine-specific CPU and AMDGPU
references are under `baselines/`. Checks cover median and p95 latency plus
host allocation count and bytes. CUDA is not a required v1 matrix axis because
the current development environment no longer has access to the remote CUDA
target.

The pre-HdrHistogram local separate-process TCP loopback artifact at
`results/external_rtc_loopback_2026-07-13.json` retained 200 samples in each of
three trials. It measured a 225322 ns pooled cycle p50, 317875 ns p99, 36167 ns
transport p50, and 107150 ns transport p99, with 32 allocated host bytes per
transaction and no completed-cycle misses against 1 ms. This is machine-local
bring-up evidence, not a claim about an external RTC or network.

The external RTC benchmark is intentionally separate from the optical matrix.
New runs retain a compressed, mergeable HdrHistogram.jl distribution for each
cycle and transport trial, report p50/p90/p95/p99/max, report p99.9 only with
at least 10,000 samples per trial, count completed-cycle deadline misses, and
run a separate steady-state allocation probe. Exact samples are optional for
short-run diagnosis. HdrHistogram.jl is pinned directly to its GitHub source;
no local checkout or published tag is required. The request/response arrival
model remains closed-loop and uncorrected, so these distributions do not
establish open-loop RTC capacity or overload behavior.

## Output API Decision

The current Proper output path is sufficient for correctness validation and
benchmark exploration. The prepared hot-call path removes repeated keyword and
model lookup overhead, and the current CPU frame path is zero-allocation after
warmup. GPU host-side allocation counts remain non-zero because CUDA.jl and
AMDGPU.jl still allocate around some host wrapper and synchronization paths.
Do not add a stricter output API until the HIL target defines:

- frame deadline
- output format consumed by the RTC
- whether the RTC wants image intensity, complex field, cropped pixels, or a
  detector model output
- ownership and lifetime of CPU/GPU output buffers

## Decision Gate

Before claiming an instrument HIL configuration is supported, record:

- deterministic convention-test results
- CPU, AMDGPU, and CUDA command-to-pixels timings
- allocation totals for the combined frame path
- whether the model consumes total residual OPD or a DM-only surface
