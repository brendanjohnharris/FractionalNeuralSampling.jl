# Samplers

```@meta
CurrentModule = FractionalNeuralSampling
```

A sampler is a keyword constructor returning a [`Sampler`](@ref), which subtypes
`SciMLBase.AbstractSDEProblem`. Since a sampler is an SDE problem, it composes with
callbacks, ensembles, and the standard `solve` interface.

| Constructor | Alias | Order | Noise | Dynamics |
|---|---|---|---|---|
| [`Langevin`](@ref) | `LangevinEquation` | 2 | Brownian | Underdamped Langevin |
| [`OLE`](@ref) | `OverdampedLangevinEquation` | 1 | Brownian | Overdamped Langevin |
| [`FNS`](@ref) | `FractionalNeuralSampler` | 2 | α-stable | Lévy-driven sampling |
| [`FHMC`](@ref) | `FractionalHamiltonianMonteCarlo` | 2 | α-stable | Fractional Hamiltonian Monte Carlo |
| [`sFNS`](@ref) | `SpaceFractionalNeuralSampler` | 2 | α-stable | FNS with a fractional Laplacian of 𝜋 |
| [`bFNS`](@ref) | `BiFractionalNeuralSampler` | 2 | fractional stable | sFNS with a fractional time derivative |
| [`tFOLE`](@ref) | `TemporalFractionalOverdampedLangevinEquation` | 1 | fractional Gaussian | Overdamped, fractional in time |
| [`sFOLE`](@ref) | `SpaceFractionalOverdampedLangevinEquation` | 1 | α-stable | Overdamped, fractional in space |
| [`bFOLE`](@ref) | `BiFractionalOverdampedLangevinEquation` | 1 | fractional stable | Overdamped, fractional in both |

## The common interface

Every constructor takes a target density `𝜋`, a time span `tspan`, and an initial state
`u0`, and returns a problem carrying its own default algorithm. First-order samplers
evolve position alone; second-order samplers evolve position and momentum, so `u0` stacks
both and `length(u0) == 2 * dimension(𝜋)`. A mismatch is caught at construction rather
than at the first step.

```@example samplers
using FractionalNeuralSampling
using Distributions

𝜋 = Density(Normal(0.0, 1.0))
S = OLE(; tspan = (0.0, 100.0), η = 0.5, u0 = [0.0], 𝜋)
parameters(S)
```

Samplers are immutable, and calling one returns a copy with parameters replaced. Calling a
sampler with new keywords is how a parameter sweep is written, and the original problem
stays untouched:

```@example samplers
S2 = S(; η = 1.0)
(parameters(S).η, parameters(S2).η)
```

Solutions are ordinary SciML solutions, so ensembles need nothing special:

```@example samplers
ensemble = EnsembleProblem(S)
sols = solve(ensemble, EM(); dt = 0.01, trajectories = 20)
length(sols)
```

## Langevin dynamics

The overdamped Langevin equation, [`OLE`](@ref), is the reference point for everything
that follows. A single noise strength η scales both the drift and the diffusion, keeping the stationary
density at 𝜋:

```math
\mathrm{d}x = \eta \, \nabla \log \pi(x) \, \mathrm{d}t + \sqrt{2\eta} \, \mathrm{d}W\,.
```

Adding momentum gives the underdamped equation, [`Langevin`](@ref), in which the noise
enters through the velocity and reaches the position only by integration:

```math
\mathrm{d}x = \beta v \, \mathrm{d}t\,, \qquad
\mathrm{d}v = \left(\beta \, \nabla \log \pi(x) - \eta v\right) \mathrm{d}t + \sqrt{2\eta} \, \mathrm{d}W\,,
```

where β sets the coupling between position and momentum and η is both the damping rate and
the noise strength. The velocity smooths the trajectory, so an underdamped sampler explores
ballistically over short times and diffusively over long ones.

## Lévy-driven sampling

[`FNS`](@ref) implements the sampler of
[Qi and Gong (2022)](https://doi.org/10.1038/s41467-022-32279-z). It has the structure of
an underdamped Langevin equation, with two changes: the driving noise is α-stable rather
than Gaussian, and the drift carries a prefactor that keeps the stationary density at 𝜋
under that noise,

```math
\mathrm{d}x = \left(\gamma \, c_\alpha \nabla \log \pi(x) + \beta v \right) \mathrm{d}t
    + \gamma^{1/\alpha} \, \mathrm{d}L_\alpha\,, \qquad
\mathrm{d}v = \beta \, c_\alpha \nabla \log \pi(x) \, \mathrm{d}t\,,
```

where ``c_\alpha = \Gamma(\alpha - 1) / \Gamma(\alpha/2)^2``, γ is the drift strength, and
β is the momentum coupling. At α = 2 the prefactor is one and the noise is Brownian, so
the sampler reduces to a Langevin equation with undamped momentum.

[`FHMC`](@ref) is the sampler of
[Ye and Zhu (2018)](https://arxiv.org/abs/1811.11151), which places the Lévy noise on the
velocity rather than the position and damps that noise at rate γ:

```math
\mathrm{d}x = c_\alpha \beta v \, \mathrm{d}t\,, \qquad
\mathrm{d}v = \left(c_\alpha \beta \, \nabla \log \pi(x) - \gamma v\right) \mathrm{d}t
    + \gamma^{1/\alpha} \, \mathrm{d}L_\alpha\,,
```

with ``c_\alpha = \Gamma(\alpha + 1) / \Gamma(\alpha/2 + 1)^2``. Since the noise reaches
the position only through the velocity, a jump in FHMC becomes a run of roughly constant
velocity rather than an instantaneous displacement.

### How the three compare

The bimodal target below separates the samplers, since crossing between the two modes is
rare under Brownian noise and common under Lévy noise.

```@example samplers
using CairoMakie, Fathom, Random
using FractionalNeuralSampling: Density # Makie also exports `Density`
set_theme!(fathom())
CairoMakie.activate!(type = "png", px_per_unit = 2)

𝜋 = Density(MixtureModel(Normal, [(-3.0, 0.5), (3.0, 0.5)]))
tspan = (0.0, 1000.0)
u0 = [0.0, 0.0]

samplers = ["OLE" => OLE(; tspan, η = 0.5, u0 = [0.0], 𝜋),
            "Langevin" => Langevin(; tspan, β = 1.0, η = 0.5, u0, 𝜋),
            "FNS" => FNS(; tspan, α = 1.4, β = 1.0, γ = 5.0, u0, 𝜋)]

sols = map(samplers) do (name, S)
    Random.seed!(42)
    name => solve(S; dt = 0.002)
end

fig = FourPanel()
for (i, (name, sol)) in enumerate(sols)
    ax = Axis(fig[(i > 2) + 1, mod1(i, 2)]; xlabel = "Time", ylabel = "Position",
              title = name)
    lines!(ax, sol.t, first.(sol.u); linewidth = 0.5, color = Fathom.colororder[i])
    limits!(ax, 0, 100, -8, 8)  # a window; the densities below use the whole run
end

ax = Axis(fig[2, 2]; xlabel = "Position", ylabel = "Density", title = "Sampled")
for (i, (name, sol)) in enumerate(sols)
    ziggurat!(ax, first.(sol.u); bins = range(-8, 8; length = 80),
              normalization = :pdf, label = name, color = Fathom.colororder[i])
end
xs = range(-8, 8; length = 400)
lines!(ax, xs, 𝜋.(xs); color = :black, linestyle = :dash, label = "Target")
axislegend(ax; position = :lt)
addlabels!(fig)
fig
```

Neither Langevin sampler leaves the mode it first settles in, and the two settle in
different modes, so each returns a unimodal estimate of a bimodal target. The Lévy sampler
crosses the origin 259 times and divides its samples almost evenly, placing 48.7% of them
on the positive side.

The trajectory panels show the first 100 of the 1000 time units simulated, since at this
mixing rate the whole run is solid ink; the sampled densities use every sample.

Crossing often is necessary for a correct estimate without being sufficient. Even at this
rate the sampled distribution is not the target: each mode comes out narrower than 𝜋, at
an interquartile width of 0.36 against 0.5, and 8.4% of samples fall between or beyond the
modes, carried there by jumps in transit. The Langevin samplers have 𝜋 as their stationary
density and would recover it given long enough. For the Lévy sampler, a longer run fixes
the balance between the modes and leaves that discrepancy, which is what the
space-fractional drift below addresses.

### The role of α

Lowering α thickens the tails of the driving noise, so the trajectory is built from longer
quiet stretches punctuated by larger jumps, and crossings between the modes become
common.

```@example samplers
αs = [2.0, 1.6, 1.2]
fig = FourPanel()
axd = Axis(fig[2, 2]; xlabel = "Position", ylabel = "Density", title = "Sampled")

for (i, α) in enumerate(αs)
    Random.seed!(7)
    S = FNS(; tspan = (0.0, 1000.0), α, β = 1.0, γ = 5.0, u0 = [0.0, 0.0], 𝜋)
    x = first.(solve(S; dt = 0.002).u)

    ax = Axis(fig[(i > 2) + 1, mod1(i, 2)]; xlabel = "Time", ylabel = "Position",
              title = "α = $α")
    lines!(ax, range(0, 1000; length = length(x)), x; linewidth = 0.5,
           color = Fathom.colororder[i])
    limits!(ax, 0, 100, -8, 8)  # a window; the densities use the whole run

    ziggurat!(axd, x; bins = range(-8, 8; length = 80), normalization = :pdf,
              color = Fathom.colororder[i], label = "α = $α")
end
lines!(axd, xs, 𝜋.(xs); color = :black, linestyle = :dash, label = "Target")
axislegend(axd; position = :lt)
addlabels!(fig)
fig
```

The trajectory panels again show the first 100 time units, and the densities the whole
run. At α = 2 the sampler settles into one mode and never reaches |x| = 6, crossing the
origin only while settling. Lowering α buys crossings (158 at α = 1.6, 413 at α = 1.2) at the cost
of excursion size: 0.25% of samples at α = 1.6 and 0.62% at α = 1.2 fall beyond |x| = 8,
reaching |x| ≈ 100 and ≈ 872 respectively, so the trajectory panels are cropped to the
frame. In the density panel α = 1.6 tracks the target at both modes while α = 1.2
over-peaks them, so more crossings do not by themselves buy a closer fit. [Boundaries](boundaries.md) confine a Lévy-driven sampler to a region of
interest when those excursions are unwanted.

## Fractional samplers

Three further families replace an integer-order derivative with a fractional one. Each
pairs a modified drift or time derivative with a noise process of matching exponent, since
the stationary density is recovered only when the two agree.

**Fractional in space.** [`sFOLE`](@ref) and [`sFNS`](@ref) replace ∇log𝜋 with the
fractional analogue

```math
b(x) = \frac{\nabla (-\Delta)^{(\alpha - 2)/2} \pi(x)}{\pi(x) + \lambda}\,,
```

which is the drift that makes 𝜋 stationary under α-stable noise; λ regularises the
quotient where the density is small. The fractional Laplacian is diagonal in the Fourier
basis, so it is applied spectrally: these constructors take a `domain` over which 𝜋 is
expanded, and `approx_n_modes` sets the number of modes retained. At α = 2 the operator is
the identity and the drift returns to ∇log𝜋.

**Fractional in time.** [`tFOLE`](@ref) replaces the time derivative with a Caputo
derivative of order β ∈ (0, 1],

```math
D^{\beta}_t x = \eta \, \nabla \log \pi(x) + \sqrt{\eta} \, \xi(t)\,,
```

where ξ is fractional Gaussian noise with Hurst exponent H = 1 - β/2. The
[solver](solvers.md) carries the fractional derivative, and the noise process carries the
matching correlations. Lowering β lengthens the memory of the trajectory, which slows the
approach to stationarity without changing the density that is approached.

**Fractional in both.** [`bFOLE`](@ref) and [`bFNS`](@ref) combine the two, taking a
fractional Laplacian of order α in space and a Caputo derivative of order β in time, driven
by linear fractional stable motion with H = 1/2 - β/2 + 1/α. These constructors need both a
`domain` and a `dt`, since the noise is generated on a fixed grid ahead of the solve.

```@example samplers
using IntervalSets
Random.seed!(42)
S = sFOLE(; tspan = (0.0, 2000.0), η = 0.5, α = 1.5, 𝜋, domain = -15 .. 15,
          boundaries = PeriodicBox(-7 .. 7))
x = only.(solve(S; dt = 0.01).u)

fig = OnePanel()
ax = Axis(fig[1, 1]; xlabel = "Position", ylabel = "Density")
ziggurat!(ax, x; bins = range(-7, 7; length = 80), normalization = :pdf,
          label = "sFOLE")
lines!(ax, xs, 𝜋.(xs); color = bermejo, label = "Target")
axislegend(ax)
fig
```

The space-fractional sampler recovers the bimodal target from a first-order equation, with
no momentum to carry it across the barrier: the fractional drift and the α-stable noise do
that work between them. Over this run the two modes are visited almost equally, with 50.9%
of samples on the positive side and 94.6% of them within one unit of a mode, so the halo
that [`FNS`](@ref) leaves between the modes is largely absent.

!!! note "Parameter names differ between samplers"
    The symbol attached to each role is not uniform across the family. In [`sFNS`](@ref) γ
    is the drift strength and β the momentum coupling, while in [`bFNS`](@ref) η is the
    drift strength and γ the momentum coupling. Read `parameters(S)` when in doubt.

## Reference

```@docs
AbstractSampler
Sampler
parameters
```

```@docs
Langevin
OLE
```

```@docs
FNS
FHMC
sFNS
bFNS
```

```@docs
tFOLE
sFOLE
bFOLE
```

```@docs
samplingpower
samplingaccuracy
```
