# Densities

A [`Density`](@ref) wraps the target distribution 𝜋 that a sampler is to reproduce, and
gives every sampler one interface to that target. The drift of every sampler in this
package is built either from ∇log𝜋 or from a fractional derivative of 𝜋.

## Constructing a target

Three constructors cover the ways a target is usually specified.

From any `Distributions.Distribution`, univariate or multivariate:

```@example densities
using FractionalNeuralSampling
using Distributions

𝜋 = Density(Normal(0.0, 1.0))
𝜋 = Density(MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)]))
```

From a function giving the probability density of an `N`-dimensional state, where the
dimension cannot be inferred and so is supplied as a type parameter:

```@example densities
𝜋 = Density{1}(x -> exp(-only(x)^2 / 2))
```

From a potential V, where 𝜋(x) ∝ exp(-V(x)). The resulting density is not normalised,
which is immaterial to a sampler driven by ∇log𝜋, since the normalising constant vanishes
under the logarithmic derivative:

```@example densities
𝜋 = PotentialDensity{1}(x -> (only(x)^2 - 1)^2)
```

## Evaluation

A density is callable and supports the evaluation interface below. `potential` and
`gradpotential` are the negatives of `logdensity` and `gradlogdensity`; they are provided
so that a target specified by a potential reads the same way as one specified by a density
function.

```@example densities
𝜋 = Density(Normal(0.0, 1.0))
𝜋(0.5)                  # the pdf
logdensity(𝜋, 0.5)      # log 𝜋(x)
gradlogdensity(𝜋, 0.5)  # ∇log 𝜋(x), the drift of an overdamped sampler
graddensity(𝜋, 0.5)     # ∇𝜋(x)
potential(𝜋, 0.5)       # V(x) = -log 𝜋(x)
gradpotential(𝜋, 0.5)   # ∇V(x)
dimension(𝜋)            # dimension of the state 𝜋 accepts
```

The four quantities a sampler reads are related by V = -log𝜋 and ∇log𝜋 = ∇𝜋/𝜋, so the
same target can be drawn as a density, as a potential, or as the drift field that target
induces.

```@example densities
using CairoMakie, Fathom
using FractionalNeuralSampling: Density # Makie also exports `Density`
set_theme!(fathom())
CairoMakie.activate!(type = "png", px_per_unit = 2)

unimodal = Density(Normal(0.0, 1.0))
bimodal = Density(MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)]))
well = PotentialDensity{1}(x -> (only(x)^2 - 1)^2)
xs = range(-4, 4; length = 400)

fig = FourPanel()
ax = Axis(fig[1, 1]; xlabel = "Position", ylabel = "𝜋(x)", title = "Density")
lines!(ax, xs, unimodal.(xs); label = "Normal")
lines!(ax, xs, bimodal.(xs); label = "Mixture")
axislegend(ax)

ax = Axis(fig[1, 2]; xlabel = "Position", ylabel = "V(x)", title = "Potential")
lines!(ax, xs, potential.([unimodal], xs))
lines!(ax, xs, potential.([bimodal], xs))

ax = Axis(fig[2, 1]; xlabel = "Position", ylabel = "∇log 𝜋(x)", title = "Drift")
lines!(ax, xs, gradlogdensity.([unimodal], xs))
lines!(ax, xs, gradlogdensity.([bimodal], xs))
hlines!(ax, [0.0]; color = abyad, linestyle = :dash)

ax = Axis(fig[2, 2]; xlabel = "Position", ylabel = "V(x)", title = "Double well")
lines!(ax, xs, potential.([well], xs); color = qinghai)
ylims!(ax, -1, 12)
addlabels!(fig)
fig
```

The drift panel shows the force a sampler feels: the drift is negative where the density
falls away to the right, and vanishes at each mode and at the barrier between modes. Since
the drift of the mixture reverses sign three times, a sampler driven by Brownian noise must
wait for a rare fluctuation to cross from one mode to the other.

## Gradients

Distributions carrying an analytic `Distributions.gradlogpdf` use that function directly;
everything else falls back to automatic differentiation. The gradient strategy is recorded
in the type, so dispatch costs nothing at run time, and `Density` selects that strategy on
construction:

```@example densities
Density(Normal(0.0, 1.0))                                    # analytic
Density(MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)]))     # autodiff
```

Pass a `Bool` to override the choice, as `Density{false}(d)` for analytic gradients and
`Density{true}(d)` for automatic ones. The backend defaults to `AutoForwardDiff()` and is
set through a preference, so the change survives across sessions but takes effect only
after a restart:

```julia
FractionalNeuralSampling.set_ad_backend!("AutoEnzyme()")
```

A `PotentialDensity` is always differentiated automatically, since a potential supplied as
a function carries no analytic gradient.

## Reference

```@docs
AbstractDensity
Density
DistributionDensity
PotentialDensity
distribution
FractionalNeuralSampling.set_ad_backend!
```

```@docs
logdensity
gradlogdensity
graddensity
potential
gradpotential
dimension
```
