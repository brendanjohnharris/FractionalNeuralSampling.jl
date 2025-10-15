function bfns_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, η, γ, ∇𝒟𝜋, λ = ps
    x, v = divide_dims(u, dimension(𝜋))
    b = ∇𝒟𝜋(only(x)) / (𝜋(x) + λ)
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= η .* b .+ γ .* v
    dv .= γ .* b
end
function bfns_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η, α = ps
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= η^(1 / α) # ? × dL in the integrator.
    dv .= 0.0
end

function gen_lfsm_fns(α, β; u0, tspan, dt) # * 1D with zeros for momentum noise
    tmin = length(tspan) == 2 ? minimum(tspan) : 0
    tmax = maximum(tspan)
    H = 1 / 2 - β / 2 + 1 / α
    N = Int(tspan / dt) + 1
    x = cumsum(lfsn(N, α, H; dt))
    x = hcat(x, zeros(N)) # Zeros for momentum
    ts = range(tmin, step = dt, length = N)
    @assert last(ts) == tmax
    return NoiseGrid(ts, eachrow(x))
end

"""
Bi-fractional neural sampling
"""
function bFNS(;
              tspan,
              dt,
              α, # Fractional order space
              β, # Fractional order time
              γ, # Momentum coupling
              η, # Noise strength
              𝜋, # Target distribution
              domain, # An Interval
              approx_n_modes = 10000,
              λ = 0.001, # Regularization to avoid overflow in low-prob regions
              u0 = [0.0, 0.0],
              boundaries = nothing,
              noise_rate_prototype = similar(u0),
              noise = gen_lfsm_fns(α, β; u0, tspan, dt),
              alg = CaputoEM(β, 1000), # Should match the order of the noise
              callback = (),
              kwargs...)
    ∇𝒟𝜋 = space_fractional_drift(𝜋; α, domain, approx_n_modes)
    Sampler(bfns_f!, bfns_g!;
            callback = CallbackSet(init(boundaries), callback...),
            u0,
            noise_rate_prototype,
            noise,
            tspan,
            dt,
            p = (; α, β, γ, η, ∇𝒟𝜋, λ),
            𝜋,
            alg,
            kwargs...) |> assert_dimension(; order = 2)
end

const BiFractionalNeuralSampler = bFNS
export bFNS, BiFractionalNeuralSampler
