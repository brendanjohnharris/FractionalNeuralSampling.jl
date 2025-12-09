import SpecialFunctions: gamma

# * "Fractional neural sampling as a theory of spatiotemporal probabilistic computations in neural circuits", Qi and Gong
function sfns_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, β, γ, ∇𝒟𝜋, 𝜋s, λ = ps
    x, v = divide_dims(u, dimension(𝜋))
    b = ∇𝒟𝜋(only(x)) / (𝜋s(only(x)) + λ)
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= γ .* b .+ β .* v
    dv .= β .* b
end
function sfns_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, γ = ps
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= γ^(1 / α) # ? × dL in the integrator.
    dv .= 0.0
end

function sFNS(;
              tspan, α, β, γ, λ = 1e-4, u0 = [0.0, 0.0],
              boundaries = nothing,
              domain, # The domain for the spatial fractional derivative
              approx_n_modes = 1000,
              noise_rate_prototype = similar(u0),
              𝜋 = Density(default_density(first(u0))),
              noise = NoiseProcesses.LevyProcess!(α; ND = dimension(𝜋),
                                                  W0 = zero(u0)),
              alg = EM(),
              callback = (),
              kwargs...)
    ∇𝒟𝜋, 𝜋s = space_fractional_drift(𝜋; α, domain, approx_n_modes)
    p = (; α, β, γ, ∇𝒟𝜋, 𝜋s, λ)
    Sampler(sfns_f!, sfns_g!;
            callback = CallbackSet(init(boundaries), callback...),
            kwargs...,
            u0,
            noise_rate_prototype, noise,
            tspan, p, 𝜋, alg) |> assert_dimension(; order = 2)
end

const SpaceFractionalNeuralSampler = sFNS
export sFNS, SpaceFractionalNeuralSampler
