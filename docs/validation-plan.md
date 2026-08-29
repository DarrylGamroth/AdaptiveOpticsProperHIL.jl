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
7. Direct and `AdaptiveOpticsSim.AlgorithmGraphs` execution of the optional
   Calculon Proper declaration with exact CPU, AMDGPU, and CUDA target
   admission.

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
- The Calculon declaration binds the exact graph OPD, pupil-amplitude, and
  output arrays after topology admission. Proper-specific scratch is prepared
  separately through `prepare_proper_assets`; it is not retained in the
  immutable algorithm plan.
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

The optional Calculon test has its own environment:

```bash
julia --project=test/calculon -e 'using Pkg; Pkg.instantiate()'
julia --project=test/calculon test/calculon/runtests.jl
```

It develops this package, the sibling AdaptiveOpticsSim, Proper, and
CalculonAlgorithms checkouts. It verifies direct and graph-bound CPU execution,
exact buffer admission, inference, and zero warmed allocation.

## Current Results

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
