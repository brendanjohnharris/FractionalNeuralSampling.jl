# Changelog

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
