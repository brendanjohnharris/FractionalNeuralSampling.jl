# Boundaries

```@meta
CurrentModule = FractionalNeuralSampling
```

A boundary is a callback, so it is passed to any sampler through the `boundaries` keyword
and composes with whatever other callbacks the solve already carries. Three box types are
provided, each built from one `IntervalSets` interval per dimension.

| Type | Effect when the state leaves the box |
|---|---|
| [`ReflectingBox`](@ref) | Mirror the position about the nearest face, reversing the corresponding velocity component |
| [`PeriodicBox`](@ref) | Wrap the position to the opposite face, optionally zeroing the velocity |
| [`ReentrantBox`](@ref) | Reset the position to a nominated re-entry point, zeroing the velocity by default |

A boundary for a second-order sampler acts on position and momentum together; for a
first-order sampler there is no momentum to act on, and the position rule is applied alone.

```@example boundaries
using FractionalNeuralSampling
using Distributions, IntervalSets, Random
using CairoMakie, Fathom
using FractionalNeuralSampling: Density # Makie also exports `Density`
set_theme!(fathom())
CairoMakie.activate!(type = "png", px_per_unit = 2)

𝜋 = Density(Normal(0.0, 2.0))
mk(boundaries) = FNS(; tspan = (0.0, 300.0), α = 1.3, β = 0.1, γ = 0.5,
                     u0 = [0.0, 0.0], 𝜋, boundaries)

cases = [("Unbounded", nothing, x -> trues(length(x)), Float64[]),
         ("ReflectingBox(-4 .. 4)", ReflectingBox(-4 .. 4), x -> abs.(x) .<= 4, [-4.0, 4.0]),
         ("PeriodicBox(-4 .. 4)", PeriodicBox(-4 .. 4), x -> abs.(x) .<= 4, [-4.0, 4.0]),
         ("ReentrantBox(4 => 0)", ReentrantBox(4 => 0), x -> x .<= 4, [4.0])]

fig = FourPanel()
for (i, (name, boundaries, keep, edges)) in enumerate(cases)
    Random.seed!(19)
    sol = solve(mk(boundaries); dt = 0.01)
    t, x = sol.t, first.(sol.u)
    k = keep(x)  # drop the samples the solver saves before the callback fires
    ax = Axis(fig[(i > 2) + 1, mod1(i, 2)]; xlabel = "Time", ylabel = "Position",
              title = name)
    lines!(ax, t[k], x[k]; linewidth = 0.5, color = Fathom.colororder[i])
    isempty(edges) || hlines!(ax, edges; color = abyad, linestyle = :dash)
    ylims!(ax, -8, 8)
end
addlabels!(fig)
fig
```

The unbounded trajectory wanders as far as its jumps take it. Reflection returns the
trajectory to the box each time it leaves, turning each excursion back on itself, while
wrapping continues it from the opposite face. Re-entry sends the trajectory back to a
single point whenever it passes the exit face, which is the behaviour wanted for a sampler
modelling a bounded neural state that resets.

The panels above filter the solution. A boundary callback runs
after the step, so the solver saves the uncorrected state and then saves the corrected one
at the same time; plotting `sol.u` raw therefore draws a single-sample spike at every
boundary event. In the periodic run above there are exactly 20 out-of-box samples and
exactly 20 duplicated timestamps, the first pair sitting at x = -6.5 and x = 1.5, both at
t = 0.19. Keeping only the in-box samples recovers the corrected trajectory.

Panel (d) bounds one face only: `ReentrantBox(4 => 0)` names an exit at x = 4 and a
re-entry at x = 0, leaving the trajectory free below, which is why its excursions run off
the bottom of the frame.

## Constructing boxes

`ReflectingBox` and `PeriodicBox` take one interval per dimension:

```@example boundaries
ReflectingBox(-4 .. 4)             # one dimension
PeriodicBox(-4 .. 4, -2 .. 2)      # two dimensions
PeriodicBox(-4 .. 4; reset = true) # wrap the position and zero the momentum
```

`ReentrantBox` needs an exit face and a re-entry point for each dimension, which is why it
takes pairs rather than intervals; passing an interval throws, rather than silently
guessing which end is which:

```@example boundaries
ReentrantBox(4 => 0)               # exits at x = 4, re-enters at x = 0
ReentrantBox(4 => 0, 2 => -2)      # two dimensions
```

A box also defines a region, which the adaptive samplers use to set up their
[kernel](adaptive.md) and the space-fractional samplers use to expand the target.
`domain` returns the intervals, `gridaxes` and `grid` discretise them, and `in` tests
membership:

```@example boundaries
B = PeriodicBox(-4 .. 4, -2 .. 2)
FractionalNeuralSampling.domain(B)
```

```@example boundaries
([0.0, 0.0] in B, [0.0, 5.0] in B)
```

## Reference

```@docs
AbstractBoundary
AbstractContinuousBoundary
AbstractBoxBoundary
NoBoundary
ReflectingBox
PeriodicBox
ReentrantBox
```

```@docs
gridaxes
grid
```
