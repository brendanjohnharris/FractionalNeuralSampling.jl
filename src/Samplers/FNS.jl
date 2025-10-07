import SpecialFunctions: gamma

# * "Fractional neural sampling as a theory of spatiotemporal probabilistic computations in neural circuits", Qi and Gong
function fns_f!(du, u, p, t)
    (α, β, γ), 𝜋 = p
    x, v = divide_dims(u, dimension(𝜋))
    b = gradlogdensity(𝜋)(x) * gamma(α - 1) / (gamma(α / 2) .^ 2)
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= γ .* b .+ β .* v
    dv .= β .* b
end
function fns_g!(du, u, p, t)
    (α, β, γ), 𝜋 = p
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= γ^(1 / α) # ? × dL in the integrator.
    dv .= 0.0
end

function FNS(;
             tspan, α, β, γ, u0 = [0.0, 0.0],
             boundaries = nothing,
             noise_rate_prototype = similar(u0),
             𝜋 = Density(default_density(first(u0))),
             noise = NoiseProcesses.LevyProcess!(α; ND = dimension(𝜋),
                                                 W0 = zero(u0)),
             kwargs...)
    assert_dimension(u0; order = 2, dimension = dimension(𝜋))
    Sampler(fns_f!, fns_g!; callback = boundaries, kwargs..., u0,
            noise_rate_prototype, noise,
            tspan, p = SLVector(; α, β, γ), 𝜋)
end

const FractionalNeuralSampler = FNS
export FNS, FractionalNeuralSampler
