# Samplers

Samplers are keyword constructors returning a `Sampler <: AbstractSDEProblem`. Each takes a
target density `𝜋`, a time span `tspan`, an initial state `u0`, and model-specific
parameters. Second-order (underdamped) samplers evolve position and momentum, so `u0` stacks
both: for a d-dimensional target, `length(u0) == 2d`.

| Constructor | Alias | Dynamics |
|---|---|---|
| `Langevin` | `LangevinEquation` | Underdamped Langevin (Brownian noise) |
| `OLE` | `OverdampedLangevinEquation` | Overdamped Langevin |
| `FNS` | `FractionalNeuralSampler` | Underdamped sampler driven by Lévy noise |
| `sFNS` | `SpaceFractionalNeuralSampler` | FNS with a spatial fractional (Riesz) derivative of 𝜋 |
| `bFNS` | `BiFractionalNeuralSampler` | FNS with both spatial and temporal fractional orders |
| `tFOLE`, `sFOLE`, `bFOLE` | — | Temporal-, space-, and bi-fractional overdamped Langevin |
| `FHMC` | `FractionalHamiltonianMonteCarlo` | Fractional Hamiltonian Monte Carlo |

```@docs
AbstractSampler
Sampler
parameters
```

## Langevin dynamics

```@docs
Langevin
OLE
```

## Fractional neural sampling

```@docs
FNS
sFNS
bFNS
FHMC
```

## Fractional overdamped Langevin

```@docs
tFOLE
sFOLE
bFOLE
```

## Adaptive samplers

```@docs
AdaptiveWalkSampler
AdaptiveLevySampler
```

## Diagnostics

```@docs
samplingpower
samplingaccuracy
```
