# Changelog

## v0.3.0

### Breaking
- `tFOLE`'s diffusion is now `√η` rather than `η`, so it samples 𝜋 for every η (previously 𝜋^(1/η)).
- Depends on and reexports `StochasticDiffEqLowOrder` rather than `StochasticDiffEq`. `EM` is still available; load `StochasticDiffEq` for other solvers.
- Removed `samplingefficiency` (exported but never defined), the empty `MakieExt`, and the unused `Makie` and `TimeseriesBase` weak dependencies.
- Box boundaries carry their corner element type as a type parameter.

### Fixed
- Compatibility with StochasticDiffEq v7.2 (DiffEqBase ≥ 7.21), under which every `solve` threw a `MethodError`.
- `lfsn`, `sFOLE`, `sFNS`, `bFOLE` and `bFNS` segfaulted under multithreaded FFTW (JuliaMath/FFTW.jl#236); spectral transforms now run single-threaded.
- Parameter updates via `S(; k = v)` threw for `sFOLE`, `sFNS`, `bFOLE`, `bFNS` and the adaptive samplers.
- Samplers constructed without `𝜋` now default to a standard normal target.
- `FHMC`: wrong default `u0`, ignored `boundaries`, no default algorithm.
- `Langevin`: default `u0` had one element rather than two.
- `lfsn` returned uninitialised memory for small odd `m`.
- `tFOLE`, `bFOLE` and `bFNS` failed for a tuple `tspan`.
- `space_fractional_deriv` checked only one entry of its Fourier-Laplacian assertion.
- `domain` and `in` now work for `ReentrantBox`.
- Box constructors now forward keywords, e.g. `PeriodicBox(-4 .. 4; reset = true)`.
- Adaptive samplers report a missing `boundaries` explicitly.

### Performance
- Removed per-step allocations in box boundaries, `CaputoEM`, `MultiCaputoEM`, `LevyNoise`, univariate autodiff and the `OLE`/`tFOLE` gradient. Output is unchanged.

### Documentation
- Added a documentation site and docstrings for every exported symbol.
- Corrected the docstrings of `lfsn`, `tFOLE`, `ReentrantBox` and `Sampler`, and the claim that `FNS` samples 𝜋 exactly for α < 2.

### Internal
- Tests for every sampler, including parameter updates; plotting removed from tests.
- CI tests Julia 1.12 and 1.13; Aqua's undefined-exports check is re-enabled.
- Moved exploratory scripts from `test/` to `scripts/`; removed dead code (`FourierSpectral.jl`, `Probabilities`, commented-out implementations).
- The adaptive samplers share their spectral setup.

## v0.2.0

### Breaking
- Lévy noise now uses the standard convention σ = 1 (previously σ = 1/√2).
- Sampler interface restructured: `LangevinSampler`, `LevyFlightSampler`, and `LevyWalkSampler` are replaced by a family of named constructors (below). `Sampler` now carries the target density and parameters as `p = (params, 𝜋)` with `LabelledArrays` parameter vectors.

### Samplers
- New sampler constructors, each with a long and short alias: `Langevin`, `OLE` (overdamped Langevin), `tFOLE` (temporal-fractional), `sFOLE` (space-fractional), `bFOLE` (bi-fractional), `FNS`/`FractionalNeuralSampler`, `sFNS` (space-fractional), `bFNS` (bi-fractional), and `FHMC` (fractional Hamiltonian Monte Carlo).
- Adaptive samplers with kernel-based history: `AdaptiveWalkSampler` and `AdaptiveLevySampler` (ApproxFun-backed adaptation fields).

### Solvers
- Fractional-order SDE solvers based on the L1 Euler--Maruyama approximation of the Caputo derivative: `CaputoEM` (uniform order), `MultiCaputoEM` (per-variable orders), and `PositionalCaputoEM` (fractional first variable only). All reduce exactly to `EM` at β = 1.
- `Window`: fixed-length circular buffer used for solver history.

### Noise and densities
- Linear fractional stable motion: `lfsm`/`lfsn` generators.
- `PotentialDensity` for densities specified by a potential function.
- Spectral fractional-Laplacian machinery (`PowerOperator`, Fourier spectral methods) supporting the space-fractional samplers.

### Boundaries
- `ReentrantBox` added alongside `ReflectingBox` and `PeriodicBox`; `gridaxes`/`grid` utilities; boundary initialisation via `boundary_init`.

### Extensions and dependencies
- Makie plotting moved from a hard MakieCore dependency to a `MakieExt` extension; `Interpolations` and `StatsBase` moved to weak dependencies (`InterpolationsExt`, extended `DistancesExt`); new `TimeseriesToolsExt` for `Timeseries` conversion of solutions.
- Added Accessors, ApproxFun, ComponentArrays, and LabelledArrays; removed LogDensityProblemsAD, StaticArrays, TransformVariables, and TransformedLogDensities.

### Tests
- Test suite reorganised around TestItems/TestItemRunner with per-sampler and per-solver test files; legacy prototype scripts removed.

## v0.1.0

- Initial release: `Sampler` SDE problem type, Langevin and Lévy samplers, `Density` interface with optional autodiff, box boundaries, and Lévy noise processes.
