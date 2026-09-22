# Quick start

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/brendanjohnharris/FractionalNeuralSampling.jl")
```

## A first sampler

Define a target density, build a sampler over it, and solve:

```julia
using FractionalNeuralSampling
using Distributions

𝜋 = Density(Normal(0.0, 1.0))
S = Langevin(; u0 = [0.0, 0.0], tspan = (0.0, 1000.0), β = 1.0, η = 1.0, 𝜋)
sol = solve(S, EM(); dt = 0.01)
x = first.(sol.u) # Position timeseries
```

Lévy-driven sampling of a bimodal target uses a stability parameter α ∈ (1, 2]; smaller α
gives heavier-tailed jumps that cross between modes more readily:

```julia
𝜋 = Density(MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)]))
S = FractionalNeuralSampler(; u0 = [0.0, 0.0], tspan = (0.0, 1000.0),
                            α = 1.4, β = 1.0, γ = 1.0, 𝜋)
sol = solve(S, EM(); dt = 0.01)
```

## Where next

- [Densities](densities.md): specifying the target 𝜋.
- [Samplers](samplers.md): the sampler family and their parameters.
- [Noise processes](noise.md): α-stable noise and linear fractional stable motion.
- [Solvers](solvers.md): the Caputo fractional solvers.
- [Boundaries](boundaries.md): reflecting, periodic, and reentrant boxes.
