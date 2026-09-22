# ? PotentialDensity (supply a potential, get a NON_NORMALIZED density)
export PotentialDensity

"""
    PotentialDensity{N}(V)

A target specified by its potential, so that 𝜋(x) ∝ exp(-V(x)), over an `N`-dimensional
state.

The density is not normalised. This is immaterial to a sampler driven by ∇log𝜋, since the
normalising constant vanishes under the logarithmic derivative, but it does mean `𝜋(x)`
returns an unnormalised value. Gradients are always automatic, since a potential supplied
as a function carries no analytic gradient.

```julia
𝜋 = PotentialDensity{1}(x -> (only(x)^2 - 1)^2)   # a symmetric double well
```
"""
struct PotentialDensity{D, N, doAd} <: AbstractDensity{D, N, doAd}
    potential::D
end
PotentialDensity{N}(potential::D) where {D, N} = PotentialDensity{D, N, true}(potential)

capabilities(::Type{<:PotentialDensity}) = LogDensityProblems.LogDensityOrder{1}()

potential(D::PotentialDensity) = D.potential
logdensity(D::PotentialDensity) = (-) ∘ potential(D)
gradpotential(D::PotentialDensity) = (.-) ∘ gradlogdensity(D)
density(D::PotentialDensity) = exp ∘ (-) ∘ potential(D)
