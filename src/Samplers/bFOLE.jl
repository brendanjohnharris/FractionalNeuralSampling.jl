
function gen_lfsm(α, β; u0, tspan, dt, seed) # * 1d for now
    tmin = length(tspan) == 2 ? minimum(tspan) : 0
    tmax = maximum(tspan)
    H = 1 / 2 - β / 2 + 1 / α
    N = Int(tspan / dt) + 1
    x = cumsum(lfsn(N, α, H; dt, rng = Xoshiro(seed)))
    ts = range(tmin, step = dt, length = N)
    @assert last(ts) == tmax
    return NoiseGrid(ts, x)
end

"""
Bi fractional overdamped langevin equation
"""
function bFOLE(;
               tspan,
               dt,
               η, # Noise strength
               α, # Fractional order space
               β, # Fractional order time
               𝜋, # Target distribution
               domain, # An Interval
               λ = 0.001, # Regularization to avoid overflow in low-prob regions
               u0 = [0.0],
               boundaries = nothing,
               noise_rate_prototype = similar(u0),
               seed = nothing,
               noise = gen_lfsm(α, β; u0, tspan, dt, seed),
               approx_n_modes = 10000,
               alg = CaputoEM(β, 1000), # Should match the order of the noise
               callback = (),
               kwargs...)
    ∇𝒟𝜋 = space_fractional_drift(𝜋; α, domain, approx_n_modes)
    Sampler(sfole_f!, sfole_g!;
            callback = CallbackSet(init(boundaries), callback...),
            u0,
            noise_rate_prototype,
            noise,
            tspan,
            dt,
            p = (; η, α, β, ∇𝒟𝜋, λ),
            𝜋,
            seed = rand(Xoshiro(seed), UInt),
            alg,
            kwargs...) |> assert_dimension(; order = 1)
end

const BiFractionalOverdampedLangevinEquation = bFOLE
export bFOLE, BiFractionalOverdampedLangevinEquation
