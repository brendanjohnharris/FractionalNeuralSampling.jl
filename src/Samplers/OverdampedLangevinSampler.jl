# * Overdamped Langevin sampler (Brownian motion)
function overdamped_langevin_f!(dx, x, p, t)
    γ, 𝜋 = p
    b = gradlogdensity(𝜋)(x)
    dx .= γ .* b
end
function overdamped_langevin_g!(dx, x, p, t)
    γ, 𝜋 = p
    dx .= sqrt(2) * sqrt(γ)
end

function OverdampedLangevinSampler(; tspan, γ, u0 = [0.0], boundaries = nothing,
                                   noise_rate_prototype = zero(u0),
                                   𝜋 = Density(default_density(first(u0))),
                                   noise = WienerProcess!(length(tspan) > 1 ? first(tspan) :
                                                          0.0,
                                                          zero(u0)),
                                   kwargs...)
    Sampler(overdamped_langevin_f!, overdamped_langevin_g!; callback = boundaries,
            kwargs..., u0,
            noise_rate_prototype, noise,
            tspan, p = (γ, 𝜋))
end
