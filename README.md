# FractionalNeuralSampling.jl

[![Build Status](https://github.com/brendanjohnharris/FractionalNeuralSampling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/brendanjohnharris/FractionalNeuralSampling.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/brendanjohnharris/FractionalNeuralSampling.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/brendanjohnharris/FractionalNeuralSampling.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A Julia package for simulating fractional neural sampling: stochastic samplers driven by Lévy (α-stable) noise and fractional-order dynamics, following [Qi and Gong (2022)](https://doi.org/10.1038/s41467-022-32279-z), *Fractional neural sampling as a theory of spatiotemporal probabilistic computations in neural circuits*.

Samplers are defined as `SDEProblem`-compatible types built on the [SciML](https://sciml.ai) ecosystem ([StochasticDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)), so they compose with the standard `solve`/`init` interface, callbacks, and ensemble machinery. The package provides:

- A `Density` type wrapping target distributions (from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), plain functions, or potentials), with analytic or automatic differentiation of log-densities.
- A family of samplers: classical Langevin dynamics, Lévy-driven fractional neural sampling, and space- and time-fractional variants.
- Custom SDE solvers for Caputo fractional derivatives (L1 Euler--Maruyama schemes).
- Lévy α-stable noise processes compatible with [DiffEqNoiseProcess.jl](https://github.com/SciML/DiffEqNoiseProcess.jl), including linear fractional stable motion.
- Box boundary conditions (reflecting, periodic, reentrant) implemented as callbacks.

<br>

# Usage

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/brendanjohnharris/FractionalNeuralSampling.jl")
using FractionalNeuralSampling
```

## Densities

A `Density` represents the target distribution 𝜋 to be sampled. Construct one from any `Distributions.Distribution`:

```julia
using Distributions
D = Density(Normal(0.0, 1.0)) # Uses the analytic gradlogpdf
D = Density(MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)])) # Falls back to autodiff
```

Densities are callable (`D(x)` evaluates the pdf) and support `logdensity`, `gradlogdensity`, `potential`, and `gradpotential`. Distributions with an analytic `gradlogpdf` use it directly; otherwise gradients are computed by automatic differentiation. The AD backend defaults to ForwardDiff and can be changed persistently:

```julia
FractionalNeuralSampling.set_ad_backend!("AutoForwardDiff()") # Takes effect after restart
```

Two further constructors cover non-distribution targets: `Density{N}(f)` for a pdf given as a function of an `N`-dimensional state, and `PotentialDensity{N}(V)` for a target specified by its potential V(x) ∝ -log 𝜋(x).

## Samplers

Samplers are keyword constructors returning a `Sampler <: AbstractSDEProblem`. Each takes a target density `𝜋`, a time span `tspan`, an initial state `u0`, and model-specific parameters. Second-order (underdamped) samplers evolve position and momentum, so `u0` stacks both: for a d-dimensional target, `length(u0) == 2d`.

| Constructor | Alias | Dynamics |
|---|---|---|
| `Langevin` | `LangevinEquation` | Underdamped Langevin (Brownian noise) |
| `OLE` | `OverdampedLangevinEquation` | Overdamped Langevin |
| `FNS` | `FractionalNeuralSampler` | Underdamped sampler driven by Lévy noise |
| `sFNS` | `SpaceFractionalNeuralSampler` | FNS with a spatial fractional (Riesz) derivative of 𝜋 |
| `bFNS` | `BiFractionalNeuralSampler` | FNS with both spatial and temporal fractional orders |
| `tFOLE`, `sFOLE`, `bFOLE` | — | Temporal-, space-, and bi-fractional overdamped Langevin |
| `FHMC` | `FractionalHamiltonianMonteCarlo` | Fractional Hamiltonian Monte Carlo |

For example, a Langevin sampler of a standard normal:

```julia
using FractionalNeuralSampling
using Distributions

D = Density(Normal(0.0, 1.0))
S = Langevin(; u0 = [0.0, 0.0], tspan = (0.0, 1000.0), β = 1.0, η = 1.0, 𝜋 = D)
sol = solve(S, EM(); dt = 0.01)
x = first.(sol.u) # Position timeseries
```

Fractional neural sampling of a bimodal target uses Lévy noise with stability parameter α ∈ (1, 2]; smaller α gives heavier-tailed jumps that cross between modes more readily:

```julia
D = Density(MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)]))
S = FractionalNeuralSampler(; u0 = [0.0, 0.0], tspan = (0.0, 1000.0),
                            α = 1.4, # Lévy stability parameter
                            β = 0.1, # Momentum coupling
                            γ = 0.5, # Drift strength
                            𝜋 = D)
sol = solve(S; dt = 0.01) # Uses the default algorithm (EM) supplied at construction
```

The space-fractional sampler additionally requires a `domain` over which the fractional derivative of the density is approximated spectrally:

```julia
using IntervalSets
S = sFNS(; u0 = [0.0, 0.0], tspan = (0.0, 5000.0), α = 1.5, β = 0.05, γ = 0.5,
         𝜋 = D, domain = -15 .. 15, boundaries = PeriodicBox(-7 .. 7))
```

Samplers are immutable but callable for parameter updates, and support `remake`:

```julia
S2 = S(; γ = 1.0) # New sampler with updated parameter
```

Multivariate targets work the same way; provide a multivariate density and a correspondingly-sized `u0` (`ArrayPartition`s and `ComponentArrays` are also supported for structured states).

## Solving and solutions

`solve` returns a standard SciML solution: `sol.t` holds times and `sol.u` holds states. Ensembles follow the usual interface:

```julia
ensemble = EnsembleProblem(S)
sols = solve(ensemble, EM(); dt = 0.01, trajectories = 100)
```

With [TimeseriesTools.jl](https://github.com/brendanjohnharris/TimeseriesTools.jl) loaded, solutions convert to annotated timeseries via `Timeseries(sol)` (a package extension).

## Fractional solvers

Three custom `StochasticDiffEq`-compatible algorithms solve SDEs with Caputo fractional time derivatives of order β ∈ (0, 1], using an L1 Euler--Maruyama approximation with a truncated history of `nhist` steps:

```julia
S = OLE(; η = 0.1, u0 = [0.0], 𝜋 = Density(Normal(0.0, 1.0)), tspan = 100.0)
sol = solve(S, CaputoEM(0.6, 1000); dt = 0.001) # Order β = 0.6, 1000 history steps
```

- `CaputoEM(β, nhist)`: uniform fractional order for all variables.
- `MultiCaputoEM(βs, nhist)`: a separate order for each variable.
- `PositionalCaputoEM(β₁, nhist)`: fractional order for the first variable only; plain EM for the rest.

All three reduce exactly to `EM()` at β = 1.

## Noise processes

`LevyProcess(α)` (and the in-place `LevyProcess!`) construct α-stable noise processes usable wherever a `DiffEqNoiseProcess` is expected. Lévy steps are drawn isotropically with the standard convention σ = 1. Linear fractional stable motion is available via `lfsm`, and `lfsn` gives its increments (linear fractional stable noise), for sampling paths with tunable self-similarity and tail exponents.

## Boundaries

Boundary conditions are callbacks constructed from box types over `IntervalSets`:

```julia
using IntervalSets
S = Langevin(; u0 = [0.0, 0.0], tspan = (0.0, 1000.0), β = 1.0, η = 0.1,
             𝜋 = Density(Uniform(-5.0, 5.0)), boundaries = ReflectingBox(-5 .. 5))
```

`ReflectingBox` reflects position and momentum at the walls, `PeriodicBox` wraps positions, and `ReentrantBox` re-inserts trajectories that exit through specified faces. Multidimensional boxes take one interval per dimension, e.g. `PeriodicBox(-5 .. 5, -5 .. 5)`.

## Analysis extensions

Loading companion packages activates extensions:

- **TimeseriesTools.jl**: `Timeseries(sol)` conversion, `samplingpower`, and `samplingaccuracy` for quantifying sampler performance against the target.
- **Distances.jl + StatsBase.jl**: divergence measures (e.g. `KLDivergence()(D, x)`) between a `Density` and sampled points.
- **Makie.jl**: plotting recipes for densities.
- **Interpolations.jl**: densities interpolated from empirical grids.
