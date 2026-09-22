import SpecialFunctions: gamma

# * "Stochastic Fractional Hamiltonian Monte Carlo", Nanyang Ye & Zhanxing Zhu
# * Levy walk sampler (noise on velocity)
function fractional_hmc_f!(du, u, p, t) # Eq. 15
    ps, 𝜋 = p
    @unpack α, β, γ = ps
    x, v = divide_dims(u, dimension(𝜋))
    c_α = gamma(α + 1) / (gamma(α / 2 + 1) .^ 2)
    ∇V = -gradlogdensity(𝜋)(x) # ? Should this be in-place
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= c_α .* β .* v
    return dv .= -c_α .* β .* ∇V - γ .* v
end
function fractional_hmc_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, β, γ = ps
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= 0.0 # ? No noise on position
    return dv .= γ^(1 / α) # ? × dL in the integrator.
end

"""
    FHMC(; tspan, α, β, γ, u0 = [0.0, 0.0], 𝜋 = default_density(u0), kwargs...)

Fractional Hamiltonian Monte Carlo, after
[Ye and Zhu (2018)](https://arxiv.org/abs/1811.11151), which places the α-stable noise on
the velocity rather than the position.

```math
\\mathrm{d}x = c_\\alpha \\beta v \\, \\mathrm{d}t, \\qquad
\\mathrm{d}v = (c_\\alpha \\beta \\nabla \\log \\pi(x) - \\gamma v) \\, \\mathrm{d}t
    + \\gamma^{1/\\alpha} \\, \\mathrm{d}L_\\alpha
```

with ``c_α = Γ(α+1)/Γ(α/2+1)^2``. Since the noise reaches the position only through the
velocity, a jump appears as a run of roughly constant velocity rather than as an
instantaneous displacement. Aliased as `FractionalHamiltonianMonteCarlo`.

# Arguments
- `tspan`: time span, as a tuple or a final time
- `α`: stability of the driving noise, in (1, 2]
- `β`: coupling between position and momentum
- `γ`: damping rate, which also scales the noise as ``γ^{1/α}``
- `u0`: initial `[position; momentum]`
- `𝜋`: target [`Density`](@ref)

Remaining keywords pass through to [`Sampler`](@ref).
"""
function FHMC(;
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
    return Sampler(
        fractional_hmc_f!, fractional_hmc_g!;
        callback = CallbackSet(boundary_init(boundaries), callback...),
        kwargs..., u0,
        noise_rate_prototype, noise,
        tspan, p = SLVector(; α, β, γ), 𝜋, alg
    ) |> assert_dimension(; order = 2)
end

const FractionalHamiltonianMonteCarlo = FHMC
export FractionalHamiltonianMonteCarlo, FHMC
