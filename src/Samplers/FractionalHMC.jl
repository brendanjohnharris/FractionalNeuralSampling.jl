import SpecialFunctions: gamma
export FractionalHMC

# * "Stochastic Fractional Hamiltonian Monte Carlo", Nanyang Ye & Zhanxing Zhu
# * Levy walk sampler (noise on velocity)
function fractional_hmc_f!(du, u, p, t) # Eq. 15
    (α, β, γ), 𝜋 = p
    x, v = divide_dims(u, length(u) ÷ 2)
    c_α = gamma(α + 1) / (gamma(α / 2 + 1) .^ 2)
    ∇V = -gradlogdensity(𝜋)(x) # ? Should this be in-place
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= c_α .* β .* v
    dv .= -c_α .* β .* ∇V - γ .* v
end
function fractional_hmc_g!(du, u, p, t)
    (α, β, γ), 𝜋 = p
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= 0.0 # ? No noise on position
    dv .= sqrt(2) * γ^(1 / α) # ? × dL in the integrator.
end

function FractionalHMC(;
                              tspan, α, β, γ, u0 = [0.0 0.0],
                              boundaries = nothing,
                              noise_rate_prototype = similar(u0),
                              𝜋 = Density(default_density(first(u0))),
                              noise = NoiseProcesses.LevyProcess!(α; ND = 2,
                                                                  W0 = zero(u0)),
                              kwargs...)
    Sampler(fractional_hmc_f!, fractional_hmc_g!; callback = boundaries, kwargs..., u0,
            noise_rate_prototype, noise,
            tspan, p = ((α, β, γ), 𝜋))
end
