# Noise processes

Every sampler is driven by a noise process, and for the fractional samplers the exponent of
that process must match the exponent of the drift. This page covers the two processes the
package supplies: α-stable (Lévy) noise, and linear fractional stable motion.

## α-stable noise

[`LevyProcess!`](@ref) builds a `DiffEqNoiseProcess`-compatible process whose increments are
drawn isotropically: a direction uniform on the sphere, and a magnitude from a stable
distribution with stability α, skewness β, scale σ and location μ. The increment over a
step is scaled by ``|\mathrm{d}t|^{1/\alpha}``, which is the scaling that makes the process
self-similar with exponent 1/α.

```@example noise
using FractionalNeuralSampling
using DiffEqNoiseProcess, Random
using CairoMakie, Fathom
set_theme!(fathom())
CairoMakie.activate!(type = "png", px_per_unit = 2)

αs = [2.0, 1.5, 1.2]
paths = map(αs) do α
    Random.seed!(11)
    W = LevyProcess!(α; W0 = [0.0])
    sol = solve(NoiseProblem(W, (0.0, 100.0)); dt = 0.01)
    α => only.(sol.u)
end

fig = TwoPanel()
ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Position", title = "Sample paths")
for (i, (α, x)) in enumerate(paths)
    lines!(ax, range(0, 100; length = length(x)), x; linewidth = 0.5,
           color = Fathom.colororder[i], label = "α = $α")
end
axislegend(ax; position = :lt)

ax = Axis(fig[1, 2]; xlabel = "|Δx|", ylabel = "P(|Δx| > y)",
          xscale = log10, yscale = log10, title = "Increment tails")
for (i, (α, x)) in enumerate(paths)
    d = sort(abs.(diff(x)))
    d = d[d .> 0]
    lines!(ax, d, range(1, 0; length = length(d)); color = Fathom.colororder[i],
           label = "α = $α")
end
xlims!(ax, 1.0e-4, 1.0e1)
ylims!(ax, 1.0e-5, 1.0)
axislegend(ax; position = :lb)
addlabels!(fig)
fig
```

At α = 2 the stable distribution is Gaussian and the path is a rescaled Brownian motion;
below 2 the survival function of the increments decays as a power law, so the path is built
from many small steps and occasional large ones. Those large steps let a
Lévy-driven sampler cross a barrier that a Brownian sampler must wait out.

The process is in place only. [`LevyProcess`](@ref), the out-of-place constructor, throws
on construction rather than failing at the first step of a solve.

## Linear fractional stable motion

The fractional-in-time samplers need noise that is both heavy-tailed and correlated, which
linear fractional stable motion supplies. The process is the stable-law analogue of
fractional Brownian motion: a moving average of α-stable increments with a power-law
kernel, self-similar with a Hurst exponent H that is set independently of α.

[`lfsn`](@ref) returns the increments and [`lfsm`](@ref) their cumulative sum. Two
parameters control the approximation: `m`, the number of points between successive motion
points, and `M`, the length of the lookback window.

```@example noise
Random.seed!(3)
ts = range(0, 1; length = 2000)

fig = TwoPanel()
ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Position", title = "Varying H (α = 1.8)")
for (i, H) in enumerate([0.3, 0.5, 0.8])
    Random.seed!(3)
    lines!(ax, ts, lfsm(2000, 1.8, H; dt = 1 / 2000); linewidth = 0.5,
           color = Fathom.colororder[i], label = "H = $H")
end
axislegend(ax; position = :lt)

ax = Axis(fig[1, 2]; xlabel = "Time", ylabel = "Position", title = "Varying α (H = 0.5)")
for (i, α) in enumerate([2.0, 1.5, 1.2])
    Random.seed!(3)
    lines!(ax, ts, lfsm(2000, α, 0.5; dt = 1 / 2000); linewidth = 0.5,
           color = Fathom.colororder[i], label = "α = $α")
end
axislegend(ax; position = :lt)
addlabels!(fig)
fig
```

The two exponents change the path in different ways. H controls the correlation between
successive increments, so H > 1/2 gives persistent, smooth-looking excursions and H < 1/2
gives antipersistent, jagged ones. α controls the tails of the increments, so lowering it
introduces jumps at every scale while leaving the correlation structure alone.

Increments are normalised by their interquartile range, so the returned series has scale 1
at `dt = 1` and scales as ``\mathrm{d}t^{H}`` otherwise. That normalisation lets `tFOLE`,
`bFOLE` and `bFNS` generate a noise grid with the right exponent and a known amplitude
before the solve begins.

!!! note "FFTW threading"
    The moving average is evaluated with an in-place FFT, which segfaults under
    multithreaded FFTW ([JuliaMath/FFTW.jl#236](https://github.com/JuliaMath/FFTW.jl/issues/236)).
    Every transform in the package therefore runs through
    `FractionalNeuralSampling.serial_fftw`, which restores the caller's thread count
    afterwards. No action is needed by the caller.

## Fractional operators

The space-fractional samplers need the fractional Laplacian ``(-\Delta)^{(\alpha-2)/2}``,
which is diagonal in the Fourier basis and so is built as a power of an `ApproxFun`
operator. [`Power`](@ref) wraps any diagonal operator and raises each diagonal entry to a
given power, leaving zero entries alone so that the zero-frequency mode survives a negative
exponent.

```@docs
Power
```

## Reference

```@docs
LevyProcess
LevyProcess!
lfsn
lfsm
FractionalNeuralSampling.serial_fftw
```
