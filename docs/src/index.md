```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "FractionalNeuralSampling"
  text: "Lévy-driven and fractional-order samplers"
  tagline: "SDE samplers for fractional neural sampling, built on the SciML interface."
  actions:
    - theme: brand
      text: Get started
      link: /quickstart
    - theme: alt
      text: Samplers
      link: /samplers
    - theme: alt
      text: View on Github
      link: https://github.com/brendanjohnharris/FractionalNeuralSampling.jl

features:
  - icon: 🎯
    title: Densities
    details: Target distributions from Distributions.jl, plain functions, or potentials, with analytic or automatic gradients.
    link: /densities
  - icon: 🎲
    title: Samplers
    details: Langevin dynamics, fractional neural sampling, and space-, time-, and bi-fractional variants.
    link: /samplers
  - icon: 🔁
    title: Adaptive samplers
    details: Samplers that deposit a decaying kernel to repel themselves from states they have already visited.
    link: /adaptive
  - icon: 〰️
    title: Noise processes
    details: Lévy α-stable noise and linear fractional stable motion for DiffEqNoiseProcess.jl.
    link: /noise
  - icon: ⚙️
    title: Solvers
    details: Euler--Maruyama schemes for Caputo fractional derivatives.
    link: /solvers
---
```

```@raw html
<p style="margin-bottom:2cm"></p>

<div class="vp-doc" style="width:80%; margin:auto">
```

## Overview

FractionalNeuralSampling simulates stochastic samplers driven by Lévy (α-stable) noise and
fractional-order dynamics, following [Qi and Gong (2022)](https://doi.org/10.1038/s41467-022-32279-z),
*Fractional neural sampling as a theory of spatiotemporal probabilistic computations in neural circuits*.

Samplers are `SDEProblem`-compatible types, so they compose with the standard `solve`/`init`
interface, callbacks, and ensemble machinery. Only
[StochasticDiffEqLowOrder.jl](https://github.com/SciML/StochasticDiffEq.jl) is a dependency,
which provides `EM`; add `StochasticDiffEq` yourself for the other upstream solvers.

```@raw html
</div>
```
