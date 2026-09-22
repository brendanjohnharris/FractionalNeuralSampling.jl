# Densities

A [`Density`](@ref) wraps the target distribution 𝜋 to be sampled. Densities are callable
(`𝜋(x)` evaluates the pdf) and support `logdensity`, `gradlogdensity`, `potential`, and
`gradpotential`. Distributions with an analytic `gradlogpdf` use it directly; otherwise
gradients come from automatic differentiation, with a backend that defaults to ForwardDiff
and can be set persistently with `FractionalNeuralSampling.set_ad_backend!`.

```@docs
AbstractDensity
Density
DistributionDensity
PotentialDensity
```

## Evaluation

```@docs
logdensity
gradlogdensity
graddensity
potential
gradpotential
dimension
distribution
```
