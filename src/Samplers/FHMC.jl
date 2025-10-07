import SpecialFunctions: gamma

# * "Stochastic Fractional Hamiltonian Monte Carlo", Nanyang Ye & Zhanxing Zhu
# * Levy walk sampler (noise on velocity)
function fractional_hmc_f!(du, u, p, t) # Eq. 15
    ps, 𝜋 = p
    @unpack α, β, γ = ps
    x, v = divide_dims(u, dimension(𝜋))
    c_α = gamma(α + 1) / (gamma(α / 2 + 1) .^ 2)
    ∇V = -gradlogdensity(𝜋)(x) # ? Should this be in-place
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= c_α .* β .* v
    dv .= -c_α .* β .* ∇V - γ .* v
end
function fractional_hmc_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, β, γ = ps
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= 0.0 # ? No noise on position
    dv .= sqrt(2) * γ^(1 / α) # ? × dL in the integrator.
    # ! No SQRT 2???
end

function FHMC(;
              tspan, α, β, γ, u0 = [0.0 0.0],
              boundaries = nothing,
              noise_rate_prototype = similar(u0),
              𝜋 = Density(default_density(first(u0))),
              noise = NoiseProcesses.LevyProcess!(α; ND = dimension(𝜋),
                                                  W0 = zero(u0)),
              kwargs...)
    Sampler(fractional_hmc_f!, fractional_hmc_g!; callback = boundaries, kwargs..., u0,
            noise_rate_prototype, noise,
            tspan, p = SLVector(; α, β, γ), 𝜋)
end

const FractionalHamiltonianMonteCarlo = FHMC
export FractionalHamiltonianMonteCarlo, FHMC
