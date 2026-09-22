# Solvers

Custom SDE solvers implementing L1 Euler--Maruyama schemes for Caputo fractional
derivatives. They plug into the usual `solve(problem, solver; dt)` call.

```@docs
CaputoEM
MultiCaputoEM
PositionalCaputoEM
```
