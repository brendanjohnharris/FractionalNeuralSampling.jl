import SpecialFunctions: gamma

# * "Fractional neural sampling as a theory of spatiotemporal probabilistic computations in neural circuits", Qi and Gong
function fns_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, β, γ = ps
    x, v = divide_dims(u, dimension(𝜋))
    b = gradlogdensity(𝜋)(x) * gamma(α - 1) / (gamma(α / 2) .^ 2)
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= γ .* b .+ β .* v
    return dv .= β .* b
end
function fns_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, γ = ps
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= γ^(1 / α) # ? × dL in the integrator.
    return dv .= 0.0
end

"""
    FNS(; tspan, α, β, γ, u0 = [0.0, 0.0], 𝜋 = default_density(u0), kwargs...)

Fractional neural sampling: underdamped dynamics driven by α-stable (Lévy) noise, after
[Qi and Gong (2022)](https://doi.org/10.1038/s41467-022-32279-z).

```math
\\mathrm{d}x = (\\gamma c_\\alpha \\nabla \\log \\pi(x) + \\beta v) \\, \\mathrm{d}t
    + \\gamma^{1/\\alpha} \\, \\mathrm{d}L_\\alpha, \\qquad
\\mathrm{d}v = \\beta c_\\alpha \\nabla \\log \\pi(x) \\, \\mathrm{d}t
```

The prefactor ``c_α = Γ(α-1)/Γ(α/2)^2`` rescales the drift for Lévy noise; at α = 2 the
noise is Brownian and ``c_α = 1``. Below α = 2 the sampled distribution retains a
discrepancy from `𝜋` that a longer run does not remove, since the drift is built from
∇log𝜋 rather than from its fractional analogue; [`sFNS`](@ref) uses the latter.

Second order, so `u0` stacks
position and momentum. Aliased as `FractionalNeuralSampler`.

# Arguments
- `tspan`: time span, as a tuple or a final time
- `α`: stability of the driving noise, in (1, 2]; smaller α gives heavier tails
- `β`: coupling between position and momentum
- `γ`: drift strength, which also scales the noise as ``γ^{1/α}``
- `u0`: initial `[position; momentum]`
- `𝜋`: target [`Density`](@ref)
- `boundaries`: an [`AbstractBoundary`](@ref FractionalNeuralSampling.Boundaries.AbstractBoundary), or `nothing`
- `noise`: defaults to [`LevyProcess!`](@ref) of the same α

Remaining keywords pass through to [`Sampler`](@ref).
"""
function FNS(;
        tspan, α, β, γ, u0 = [0.0, 0.0],
        boundaries = nothing,
        noise_rate_prototype = similar(u0),
        𝜋 = default_density(u0),
        noise = NoiseProcesses.LevyProcess!(
            α; ND = dimension(𝜋),
            W0 = zero(u0)
        ),
        alg = EM(),
        callback = (),
        kwargs...
    )
    p = SLVector(; α, β, γ)

    return Sampler(
        fns_f!, fns_g!;
        callback = CallbackSet(boundary_init(boundaries), callback...),
        kwargs...,
        u0,
        noise_rate_prototype, noise,
        tspan, p, 𝜋, alg
    ) |> assert_dimension(; order = 2)
end

const FractionalNeuralSampler = FNS
export FNS, FractionalNeuralSampler
