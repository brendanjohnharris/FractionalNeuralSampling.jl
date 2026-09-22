# Boundaries

Box boundary conditions are implemented as callbacks, and are passed to a sampler
constructor as its `boundaries` argument.

```@docs
AbstractBoundary
AbstractContinuousBoundary
AbstractBoxBoundary
NoBoundary
ReflectingBox
PeriodicBox
ReentrantBox
```

## Grids

```@docs
gridaxes
grid
```
