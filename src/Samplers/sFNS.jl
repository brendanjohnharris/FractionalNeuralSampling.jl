import SpecialFunctions: gamma

# * "Fractional neural sampling as a theory of spatiotemporal probabilistic computations in
#   neural circuits", Qi and Gong
function maybeonly(x)
    if length(x) == 1
        return only(x)
    else
        return x
    end
end
function sfns_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, β, γ, ∇𝒟𝜋, 𝜋s, λ = ps
    x, v = divide_dims(u, dimension(𝜋))
    b = ∇𝒟𝜋(maybeonly(x)) / (𝜋s(maybeonly(x)) + λ)
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= γ .* b .+ β .* v
    return dv .= β .* b
end
function sfns_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, γ = ps
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= γ^(1 / α) # ? × dL in the integrator.
    return dv .= 0.0
end

"""
    sFNS(; tspan, α, β, γ, domain, λ = 1e-4, u0 = [0.0, 0.0], 𝜋, kwargs...)

Space-fractional neural sampling: [`FNS`](@ref) with the drift ∇log𝜋 replaced by its
fractional analogue,

```math
b(x) = \\frac{\\nabla (-\\Delta)^{(\\alpha - 2)/2} \\pi(x)}{\\pi(x) + \\lambda},
```

which is the drift that makes `𝜋` stationary under α-stable noise. The fractional
Laplacian is diagonal in the Fourier basis, so `𝜋` is expanded spectrally over `domain`
and the operator applied as a [`Power`](@ref). At α = 2 the operator is the identity and
`b` returns to ∇log𝜋. Aliased as `SpaceFractionalNeuralSampler`.

# Arguments
- `tspan`: time span, as a tuple or a final time
- `α`: fractional order in space, which is also the stability of the driving noise
- `β`: coupling between position and momentum
- `γ`: drift strength, which also scales the noise as ``γ^{1/α}``
- `domain`: interval over which `𝜋` is expanded; take it wider than the region sampled
- `λ`: regularises the quotient where the density is small
- `approx_n_modes`: number of Fourier modes retained, `1000` by default
- `u0`: initial `[position; momentum]`
- `𝜋`: target [`Density`](@ref)

Remaining keywords pass through to [`Sampler`](@ref).
"""
function sFNS(;
        tspan, α, β, γ, λ = 1.0e-4, u0 = [0.0, 0.0],
        boundaries = nothing,
        domain, # The domain for the spatial fractional derivative
        approx_n_modes = 1000,
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
    ∇𝒟𝜋, 𝜋s = space_fractional_drift(𝜋; α, domain, approx_n_modes)
    p = (; α, β, γ, ∇𝒟𝜋, 𝜋s, λ)
    return Sampler(
        sfns_f!, sfns_g!;
        callback = CallbackSet(boundary_init(boundaries), callback...),
        kwargs...,
        u0,
        noise_rate_prototype, noise,
        tspan, p, 𝜋, alg
    ) |> assert_dimension(; order = 2)
end

const SpaceFractionalNeuralSampler = sFNS
export sFNS, SpaceFractionalNeuralSampler
