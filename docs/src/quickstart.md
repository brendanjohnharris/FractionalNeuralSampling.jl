# Quick start

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/brendanjohnharris/FractionalNeuralSampling.jl")
```

The package is not registered. Since `StochasticDiffEqLowOrder.jl` ships as a dependency,
`EM` (the only upstream solver the samplers need) is available immediately; add
`StochasticDiffEq` separately for the rest of the SciML solver set.

## Three pieces

Every simulation requires a target density, a sampler, and a solve call:

```@example quickstart
using FractionalNeuralSampling
using Distributions

𝜋 = Density(Normal(0.0, 1.0))                                   # what to sample
S = Langevin(; u0 = [0.0, 0.0], tspan = (0.0, 200.0), β = 1.0, η = 1.0, 𝜋)
sol = solve(S, EM(); dt = 0.01)                                 # a standard SciML solution
x = first.(sol.u)                                               # position timeseries
nothing # hide
```

A [`Density`](@ref) wraps the distribution to be sampled and carries that distribution's
gradient (see [Densities](densities.md)). A sampler is a keyword constructor returning an
`SDEProblem`-compatible type, so `sol` is an ordinary SciML solution: `sol.t` holds times
and `sol.u` holds states. [`Langevin`](@ref) is second order, evolving position and
momentum; since `u0` stacks both components, a one-dimensional target needs
`length(u0) == 2`.

The sampler carries its algorithm, so `solve(S; dt = 0.01)` is equivalent to the call
above.

## Lévy-driven sampling

Replacing the Brownian noise of the Langevin equation with α-stable noise gives
[`FNS`](@ref), the sampler of Qi and Gong (2022). The stability parameter α ∈ (1, 2]
controls the tails of the driving noise: at α = 2 the noise is Gaussian, and smaller α
admits jumps large enough to cross between separated modes.

```@example quickstart
using CairoMakie, Fathom, Random
using FractionalNeuralSampling: Density # Makie also exports `Density`
set_theme!(fathom())
CairoMakie.activate!(type = "png", px_per_unit = 2)
Random.seed!(42)

𝜋 = Density(MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)]))
S = FractionalNeuralSampler(; u0 = [0.0, 0.0], tspan = (0.0, 2000.0),
                            α = 1.4, β = 0.1, γ = 0.5, 𝜋)
sol = solve(S; dt = 0.01)
x = first.(sol.u)

fig = TwoPanel()
ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Position")
lines!(ax, sol.t, x; linewidth = 0.5)

ax = Axis(fig[1, 2]; xlabel = "Position", ylabel = "Density")
ziggurat!(ax, x; bins = range(-5, 5; length = 80), normalization = :pdf)
xs = range(-5, 5; length = 400)
lines!(ax, xs, 𝜋.(xs); color = bermejo, label = "Target")
axislegend(ax)
addlabels!(fig)
fig
```

The trajectory jumps between the two modes rather than diffusing across the barrier
between them, and the sampled positions recover the target. Both panels come from the same
solve, displayed as a timeseries on the left and as an empirical distribution on the
right.

## Where next

- [Densities](densities.md): specifying the target 𝜋, and where its gradients come from.
- [Samplers](samplers.md): the sampler family, their parameters, and what each one adds.
- [Adaptive samplers](adaptive.md): samplers that deposit a kernel to repel themselves from visited states.
- [Noise processes](noise.md): α-stable noise and linear fractional stable motion.
- [Solvers](solvers.md): the Caputo fractional solvers, and what a fractional time derivative does to a trajectory.
- [Boundaries](boundaries.md): reflecting, periodic, and reentrant boxes.
