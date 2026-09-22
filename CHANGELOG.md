# Changelog

## v0.3.0

### Breaking
- `tFOLE`'s diffusion is now `√η` rather than `η`. `gen_fbm` draws α = 2 stable increments, which carry variance 2dt under the package's σ = 1 convention rather than the dt of a Wiener process, so the old coefficient left the stationary density at 𝜋^(1/η), correct only at η = 1: against a standard normal at η = 2 it sampled 𝜋^(1/2), of standard deviation 1.41. It is now 𝜋 for every η, matching `OLE`.
- Removed `samplingefficiency`, which was exported but never defined.
- Removed the empty `MakieExt`; `Makie` is no longer a weak dependency.
- Box boundaries (`ReflectingBox`, `PeriodicBox`, `ReentrantBox`) carry their corner element type as a type parameter, so their fields are concrete.

### Fixed
- Updating a parameter with the callable form a sampler documents, `S(; γ = 1.0)`, threw a `MethodError` for `sFOLE`, `sFNS`, `bFOLE`, `bFNS` and both adaptive samplers, including for the `sFNS` example the README gives it with. Those six carry ApproxFun operators, and a transform plan, beside their scalar parameters, so their parameters are a `NamedTuple` rather than an `SLArray`, and the update went straight to `SLVector`, which has no `NamedTuple` method. `remake` was unaffected.
- Every spectral transform now runs through `FractionalNeuralSampling.serial_fftw`, which drops FFTW to one thread for the call and restores the previous count. FFTW segfaults on the in-place plans used here when it is multithreaded (JuliaMath/FFTW.jl#236), taking the session down rather than throwing, so `lfsn` and every sampler built on a spectral approximation of the target (`sFOLE`, `sFNS`, `bFOLE`, `bFNS`) crashed a default multicore session at any `approx_n_modes`. Only `lfsn` had documented the hazard, and the test suite hid it by setting the thread count to 1 before any sampler was built.
- `FNS`, `FHMC`, `sFNS` and the adaptive samplers threw a `MethodError` when constructed without a `𝜋`; their default target now works, and `Langevin`, `OLE` and `tFOLE` gained the same default.
- `FHMC` defaulted `u0` to a 1×2 matrix, ignored `boundaries` unless they were already a callback, and had no default algorithm.
- `Langevin` defaulted `u0` to `[0.0]`, one element short of the 2 a second-order sampler needs, so the default always failed `assert_dimension`. It is now `[0.0, 0.0]`.
- `lfsn` returned uninitialised memory for small odd `m`, since the FFT padding was computed before `m` was made even.
- `tFOLE`, `bFOLE` and `bFNS` failed for a tuple `tspan`, having divided the tuple by `dt`.
- The Fourier-Laplacian assertions in `space_fractional_deriv` checked one diagonal entry rather than 100 (`1:length(100)`).
- `domain` and `in` now work for `ReentrantBox`.
- Adaptive samplers now report a missing `boundaries` rather than failing inside `domain`.
- `lfsn`'s docstring gave a signature it does not have (`m` and `M` are keywords, not positional) and omitted `dt`. It now also warns that multithreaded FFTW segfaults on the in-place plan (JuliaMath/FFTW.jl#236), which takes the session down rather than throwing.
- `Sampler` now normalises its `kwargs` field to `Base.Pairs` in the positional constructor that `remake` reaches. `DiffEqBase` reads `values(prob.kwargs)` and needs a `NamedTuple` back; since DiffEqBase v7.21, `_erase_problem_callback_types` rebuilds the field as a plain `NamedTuple`, whose `values` is a `Tuple`, so every `solve` failed in `merge_problem_kwargs` with a `MethodError`. This made the package unusable with StochasticDiffEq v7.2. The normalisation is a no-op on earlier versions, so one implementation covers both.

### Performance
No change to any sampler's output; all of the following remove per-step allocations.
- Box boundary conditions test for a crossing without materialising the edge distances, and take the position partition without allocating the others.
- `CaputoEM` and `MultiCaputoEM` overwrite the dropped history element rather than allocating a new one each step.
- `OLE` and `tFOLE` take the single-vector gradient path, rather than routing a one-element collection of views through the collection method.
- Univariate autodiff uses a scalar derivative rather than a gradient over a one-element vector.
- `LevyNoise` stores `ND::Int`, and constructs its `Stable` distribution once per call rather than once per variable.

### Internal
- Removed `src/FourierSpectral.jl` (a scratch script) and the empty `Probabilities` module; stripped the superseded commented-out implementations from `Window.jl` and elsewhere.
- The two adaptive samplers share their spectral setup and kernel-gradient code.

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
