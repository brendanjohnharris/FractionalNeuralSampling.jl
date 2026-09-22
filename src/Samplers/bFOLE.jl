function gen_lfsm(α, β; u0, tspan, dt, seed) # * 1d for now
    tmin = length(tspan) == 2 ? minimum(tspan) : 0
    tmax = maximum(tspan)
    H = 1 / 2 - β / 2 + 1 / α
    N = Int((tmax - tmin) / dt) + 1
    x = cumsum(lfsn(N, α, H; dt, rng = Xoshiro(seed)))
    ts = range(tmin, step = dt, length = N)
    @assert last(ts) == tmax
    return NoiseGrid(ts, x)
end

"""
    bFOLE(; tspan, dt, η, α, β, 𝜋, domain, λ = 1e-4, u0 = [0.0], kwargs...)

Overdamped Langevin dynamics that are fractional in space and in time: the fractional drift
of [`sFOLE`](@ref) with the Caputo time derivative of [`tFOLE`](@ref), driven by linear
fractional stable motion with H = 1/2 - β/2 + 1/α. The noise grid is generated ahead of the
solve, so `dt` is required at construction and must match the `dt` passed to `solve`.
Aliased as `BiFractionalOverdampedLangevinEquation`.

# Arguments
- `tspan`: time span, as a tuple or a final time
- `dt`: step size, used to generate the noise grid
- `η`: noise strength
- `α`: fractional order in space, which is also the stability of the driving noise
- `β`: fractional order in time, in (0, 1]
- `𝜋`: target [`Density`](@ref)
- `domain`: interval over which `𝜋` is expanded
- `λ`: regularises the fractional drift where the density is small
- `u0`: initial position
- `alg`: default solver, [`CaputoEM`](@ref)`(β, 1000)`, whose order must match the noise

Remaining keywords pass through to [`Sampler`](@ref).
"""
function bFOLE(;
        tspan,
        dt,
        η, # Noise strength
        α, # Fractional order space
        β, # Fractional order time
        𝜋, # Target distribution
        domain, # An Interval
        λ = 1.0e-4, # Regularization to avoid overflow in low-prob regions
        u0 = [0.0],
        boundaries = nothing,
        noise_rate_prototype = similar(u0),
        seed = rand(UInt32),
        noise = gen_lfsm(α, β; u0, tspan, dt, seed = rand(Xoshiro(seed), UInt32)),
        approx_n_modes = 1000,
        alg = CaputoEM(β, 1000), # Should match the order of the noise
        callback = (),
        kwargs...
    )
    ∇𝒟𝜋, 𝜋s = space_fractional_drift(𝜋; α, domain, approx_n_modes)
    return Sampler(
        sfole_f!, sfole_g!;
        callback = CallbackSet(boundary_init(boundaries), callback...),
        u0,
        noise_rate_prototype,
        noise,
        tspan,
        dt,
        p = (; η, α, β, ∇𝒟𝜋, 𝜋s, λ),
        𝜋,
        seed,
        alg,
        kwargs...
    ) |> assert_dimension(; order = 1)
end

const BiFractionalOverdampedLangevinEquation = bFOLE
export bFOLE, BiFractionalOverdampedLangevinEquation
