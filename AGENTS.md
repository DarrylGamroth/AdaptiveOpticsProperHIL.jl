# AGENTS.md

## Scope
This repository contains cross-package HIL validation and benchmark glue for
`AdaptiveOpticsSim.jl` and `Proper.jl`.

## Source Projects
- AO runtime package: `../AdaptiveOpticsSim.jl`
- PROPER port package: `https://github.com/DarrylGamroth/Proper.jl`

## Policy
- Keep cross-package HIL tests, benchmarks, scripts, and result artifacts here
  unless a small stable API improvement belongs in one of the source packages.
- Do not add hard dependencies from either source package to this project.
- Use deterministic inputs and seeded RNGs for tests.
- Benchmark warmed steady-state execution with BenchmarkTools and report
  allocations separately from cold-start / TTFx.
- Validate CPU by default. CUDA and AMDGPU validation should be optional and
  should skip cleanly when hardware or packages are unavailable.
- For GPU tests, disable scalar indexing where supported.
