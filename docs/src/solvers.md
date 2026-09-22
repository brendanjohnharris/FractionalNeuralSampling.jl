# Solvers

The samplers whose time derivative is fractional cannot be advanced by an ordinary
Euler--Maruyama step, since the state at a given time depends on the whole trajectory that
led to it. The package supplies three `StochasticDiffEqCore`-compatible algorithms that
carry that dependence.

## The Caputo derivative and its L1 approximation

For an order β ∈ (0, 1], the Caputo derivative replaces the instantaneous rate of change
with a weighted integral of past increments. Discretising that integral on a uniform grid
gives the L1 scheme, under which one step of

```math
D^{\beta}_t x = f(x) + g(x)\,\xi(t)
```

reads

```math
x_{n+1} = x_n + c\,f(x_n)\,\Delta t + c\,g(x_n)\,\eta_n \sqrt{\Delta t}
    - \sum_{j=1}^{n} w_j \, \Delta x_{n-j}\,,
```

with correction factor ``c = \Gamma(2 - \beta)\,\Delta t^{\beta - 1}`` and history weights
``w_j = (j+1)^{1-\beta} - j^{1-\beta}``. The first three terms are an Euler--Maruyama step
rescaled by c; the sum is the memory. At β = 1 every weight vanishes and c is one, so the
scheme reduces to `EM`.

The sum is truncated at `nhist` terms, which is the second argument of every constructor.
Truncation keeps the cost per step constant rather than growing with the elapsed time,
and sets the longest memory the solver can represent: `nhist * dt`. Since the
weights and the correction factor are built once for a fixed Δt, these are fixed-step
methods, and `isadaptive` returns `false`.

```@example solvers
using FractionalNeuralSampling
using CairoMakie, Fathom
using FractionalNeuralSampling: Density # Makie also exports `Density`
set_theme!(fathom())
CairoMakie.activate!(type = "png", px_per_unit = 2)

js = 1:200
fig = TwoPanel()
ax = Axis(fig[1, 1]; xlabel = "Steps into the past", ylabel = "Weight wⱼ",
          xscale = log10, yscale = log10, title = "History weights")
for (i, β) in enumerate([0.9, 0.7, 0.5, 0.3])
    w = @. (js + 1)^(1 - β) - js^(1 - β)
    lines!(ax, js, w; color = Fathom.colororder[i], label = "β = $β")
end
axislegend(ax)

ax = Axis(fig[1, 2]; xlabel = "β", ylabel = "Σⱼ wⱼ  (nhist = 1000)",
          title = "Total memory retained")
βs = range(0.05, 1.0; length = 100)
ks = 1:1000
lines!(ax, βs, map(βs) do β
    sum(@. (ks + 1)^(1 - β) - ks^(1 - β))
end)
addlabels!(fig)
fig
```

The weights fall off as a power law rather than exponentially, and that power-law decay is
the point: the trajectory never forgets the past, only discounts older steps. Lowering β flattens the decay
and increases the total weight carried by the history, so a smaller β needs a longer
`nhist` before truncation stops mattering.

## What a fractional time derivative does

Fractional order slows relaxation without changing the stationary density. Starting an
ensemble away from the mode of a standard normal target and following the ensemble mean
shows the effect directly.

```@example solvers
using Distributions, Random, Statistics

𝜋 = Density(Normal(0.0, 1.0))
ts = range(0, 20; length = 2001)

fig = OnePanel()
ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Mean position")
for (i, β) in enumerate([1.0, 0.8, 0.6])
    Random.seed!(2)
    trajectories = map(1:128) do _
        S = tFOLE(; tspan = (0.0, 20.0), dt = 0.01, η = 1.0, β, u0 = [3.0], 𝜋,
                  alg = CaputoEM(β, 500))
        only.(solve(S; dt = 0.01).u)
    end
    lines!(ax, ts, mean(trajectories); color = Fathom.colororder[i], label = "β = $β")
end
hlines!(ax, [0.0]; color = abyad, linestyle = :dash)
axislegend(ax)
fig
```

At β = 1 the mean decays exponentially, as it must for an overdamped Langevin equation on a
Gaussian target. Lowering β holds the ensemble back in proportion to how far it has already
moved, so the approach to the mode is slower and no longer exponential. The target is
unchanged, so the difference lies in how long the sampler takes to reach stationarity
rather than in what it reaches.

## The three algorithms

All three take `(β, nhist)` and differ only in which variables carry the fractional
derivative.

- [`CaputoEM`](@ref) applies one order β to every variable.
- [`MultiCaputoEM`](@ref) takes a vector of orders, one per variable, for systems whose
  components relax on different fractional scales.
- [`PositionalCaputoEM`](@ref) applies β to the first variable only and advances the rest
  by plain Euler--Maruyama; [`bFNS`](@ref) uses this algorithm, since only the position is
  fractional in time.

A solver is passed as usual, and overrides the default algorithm the sampler carries:

```julia
S = OLE(; tspan = (0.0, 100.0), η = 0.1, u0 = [0.0], 𝜋)
sol = solve(S, CaputoEM(0.6, 1000); dt = 0.001)
```

The order of the solver and the order of the noise process must agree. A sampler
constructed through [`tFOLE`](@ref), [`bFOLE`](@ref) or [`bFNS`](@ref) pairs them for you;
pairing them by hand is the caller's responsibility.

## Reference

```@docs
CaputoEM
MultiCaputoEM
PositionalCaputoEM
```
