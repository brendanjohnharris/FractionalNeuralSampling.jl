# Adaptive samplers

The samplers of the previous page are memoryless: the drift at a point depends on the
target alone, so a trajectory that has already visited a mode is no less likely to stay
there. The adaptive samplers add a second, time-dependent potential built from where the
trajectory has been, so visited regions are progressively filled in and the sampler is
pushed out of them. The construction follows metadynamics, and the same idea appears in
neural circuits as spike-frequency adaptation.

## The adaptation kernel

At every step the sampler deposits a copy of a user-supplied `kernel`, centred on the
current position, into an accumulated field K. That field also decays, so K tracks a
recent history rather than the whole trajectory:

```math
\frac{\mathrm{d}K}{\mathrm{d}t} = -\frac{K}{\tau_d} + \frac{k(\cdot - x(t))}{\tau_r}\,,
```

where τ_r is the deposition time and τ_d the decay time. The ratio of the two sets how
deep the accumulated field can become, and τ_r → ∞ switches adaptation off.

K is held by its coefficients in a Fourier basis, so depositing a kernel is a transform
and taking ∇K is a diagonal operator. The basis needs a domain, which is taken from the
sampler's `boundaries`; an adaptive sampler constructed without boundaries therefore
throws rather than choosing a domain for you.

Both adaptive samplers are first order, and differ only in their noise. The walk sampler
takes Brownian noise,

```math
\mathrm{d}x = -\gamma \left(\nabla V(x) + \nabla K(x)\right) \mathrm{d}t
    + \sqrt{2\gamma} \, \mathrm{d}W\,,
```

with V = -log𝜋, and the Lévy sampler takes α-stable noise with the drift prefactor
``c_\alpha = \Gamma(\alpha-1)/\Gamma(\alpha/2)^2`` that [`FNS`](@ref) also carries:

```math
\mathrm{d}x = -\gamma \, c_\alpha \left(\nabla V(x) + \nabla K(x)\right) \mathrm{d}t
    + \gamma^{1/\alpha} \, \mathrm{d}L_\alpha\,.
```

With an empty kernel the walk sampler reduces to [`OLE`](@ref), and the Lévy sampler at
α = 2 reduces to the same equation.

## Watching the kernel fill

A sampler mutates its kernel in place as it runs, so solving in stages and reading
`parameters(S).a_K` between them shows the field accumulating. Here a walk sampler explores
a single Gaussian target, and the kernel grows under wherever the trajectory has dwelt.

```@example adaptive
using FractionalNeuralSampling
using Distributions, IntervalSets, ApproxFun, Random
using CairoMakie, Fathom
using FractionalNeuralSampling: Density # Makie also exports `Density`
set_theme!(fathom())
CairoMakie.activate!(type = "png", px_per_unit = 2)
Random.seed!(42)

kernel(x) = exp(-only(x)^2 / 2)
𝜋 = Density(Normal(0.0, 1.0))
boundaries = PeriodicBox(-6 .. 6)

S = AdaptiveWalkSampler(kernel, 64; tspan = (0.0, 50.0), γ = 0.5,
                        τ_r = 2.0, τ_d = 200.0, 𝜋, boundaries)

xs = range(-6, 6; length = 400)
snapshots = let u0 = S.u0, out = []
    for stage in 1:4
        sol = solve(S(; tspan = (0.0, 50.0), u0); dt = 0.01)
        u0 = copy(sol.u[end])                      # carry the state into the next stage
        K = Fun(parameters(S).sp, copy(parameters(S).a_K))
        push!(out, (50.0 * stage, first.(sol.u), K.(xs)))
    end
    out
end

fig = TwoPanel()
ax = Axis(fig[1, 1]; xlabel = "Position", ylabel = "K(x)", title = "Adaptation kernel")
for (i, (t, _, Kx)) in enumerate(snapshots)
    lines!(ax, xs, Kx; color = Fathom.colororder[i], label = "t = $(Int(t))")
end
axislegend(ax; position = :lt)

ax = Axis(fig[1, 2]; xlabel = "Time", ylabel = "Position", title = "Trajectory")
for (i, (t, x, _)) in enumerate(snapshots)
    lines!(ax, range(t - 50, t; length = length(x)), x; linewidth = 0.5,
           color = Fathom.colororder[i])
end
addlabels!(fig)
fig
```

The kernel deepens over each stage, from a peak near 9 after the first 50 time units to
near 19 after the fourth. Since the deposited bumps are as wide as the target and the box
is only twelve units across, the field fills the whole interval rather than carving a hole
at one point, and growth slows between t = 150 and t = 200 as decay at rate 1/τ_d begins to
balance deposition.

## What adaptation buys

On a double-well target, adaptation converts the escape from a waiting problem into a
driven one: rather than waiting for a fluctuation large enough to clear the barrier, the
sampler raises the floor of the well it sits in until the barrier is gone.

```@example adaptive
𝜋 = PotentialDensity{1}(x -> 20 * (only(x)^2 - 1)^2)
boundaries = PeriodicBox(-4 .. 4)
tspan = (0.0, 400.0)

Random.seed!(1)
adaptive = AdaptiveWalkSampler(kernel, 64; tspan, γ = 0.5, τ_r = 5.0, τ_d = 500.0,
                               u0 = [-1.0], 𝜋, boundaries)
xa = only.(solve(adaptive; dt = 0.01).u)

Random.seed!(1)
plain = AdaptiveWalkSampler(kernel, 64; tspan, γ = 0.5, τ_r = Inf, τ_d = Inf,
                            u0 = [-1.0], 𝜋, boundaries)
xp = only.(solve(plain; dt = 0.01).u)

ts = range(0, 400; length = length(xa))
fig = TwoPanel()
ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Position")
lines!(ax, ts, xp; linewidth = 0.5, label = "τ_r = ∞")
lines!(ax, ts, xa; linewidth = 0.5, label = "τ_r = 5")
axislegend(ax; position = :lt)

ax = Axis(fig[1, 2]; xlabel = "Position", ylabel = "Density")
ziggurat!(ax, xp; bins = range(-2, 2; length = 60), normalization = :pdf)
ziggurat!(ax, xa; bins = range(-2, 2; length = 60), normalization = :pdf)
zs = range(-2, 2; length = 400)
lines!(ax, zs, 𝜋.(zs) ./ sum(𝜋.(zs) .* step(zs)); color = :black, linestyle = :dash,
       label = "Target")
axislegend(ax)
addlabels!(fig)
fig
```

The adapted trajectory reaches the second well; the unadapted one never leaves the first.
The price is that the sampled distribution is no longer the target, since the accumulated
kernel flattens whichever well the trajectory has occupied longest. An adaptive run
therefore reports where a target has support rather than how much support it has, and
recovering an unbiased estimate needs a reweighting the package does not currently
provide.

## Reference

```@docs
AdaptiveWalkSampler
AdaptiveLevySampler
```
