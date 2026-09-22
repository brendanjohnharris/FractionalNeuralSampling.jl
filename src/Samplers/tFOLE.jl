function tfole_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η = ps
    x = first_dims(u, dimension(𝜋))
    b = gradlogdensity(𝜋, x)
    return du .= η .* b
end
function tfole_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η = ps
    # √η, not the √(2η) of `ole_g!`: `gen_fbm` draws α = 2 stable increments, which carry
    # variance 2dt under the package's σ = 1 convention rather than the dt of a Wiener
    # process. Both choices leave the stationary density at 𝜋; `η` alone gave 𝜋^(1/η)
    return du .= sqrt(only(η)) # ? × dW in the integrator.
end

function gen_fbm(β; u0, tspan, dt, seed) # * 1d for now
    α = 2
    tmin = length(tspan) == 2 ? minimum(tspan) : 0
    tmax = maximum(tspan)
    H = 1 - β / 2
    N = Int((tmax - tmin) / dt) + 1
    x = cumsum(lfsn(N, α, H; dt, rng = Xoshiro(seed)))
    ts = range(tmin, step = dt, length = N)
    @assert last(ts) == tmax
    return NoiseGrid(ts, x)
end

"""
    tFOLE(; tspan, dt, η, β, u0 = [0.0], 𝜋, kwargs...)

Overdamped Langevin dynamics that are fractional in time: the time derivative is a Caputo
derivative of order β ∈ (0, 1], driven by fractional Gaussian noise of matching exponent,

```math
D^{\\beta}_t x = \\eta \\, \\nabla \\log \\pi(x) + \\sqrt{\\eta} \\, \\xi(t),
```

where ξ has Hurst exponent H = 1 - β/2. The noise grid is generated ahead of the solve, so
`dt` is required at construction and must match the `dt` passed to `solve`. Lowering β
lengthens the memory of the trajectory and slows the approach to stationarity, leaving the
stationary density unchanged. At β = 1 it reduces to [`OLE`](@ref). Aliased as
`TemporalFractionalOverdampedLangevinEquation`.

# Arguments
- `tspan`: time span, as a tuple or a final time
- `dt`: step size, used to generate the noise grid
- `η`: noise strength
- `β`: fractional order in time, in (0, 1]
- `u0`: initial position
- `𝜋`: target [`Density`](@ref)
- `seed`: seed for the noise grid
- `alg`: default solver, [`CaputoEM`](@ref)`(β, 1000)`

Remaining keywords pass through to [`Sampler`](@ref).
"""
function tFOLE(;
        tspan,
        dt,
        η, # Noise strength
        β, # Fractional order in time
        u0 = [0.0],
        boundaries = nothing,
        noise_rate_prototype = similar(u0),
        seed = rand(UInt32),
        noise = gen_fbm(β; u0, tspan, dt, seed),
        𝜋 = default_density(u0; dims = length(u0)),
        callback = (),
        alg = CaputoEM(β, 1000),
        kwargs...
    )
    return Sampler(
        tfole_f!, tfole_g!;
        𝜋,
        callback = CallbackSet(boundary_init(boundaries), callback...),
        u0,
        noise_rate_prototype,
        noise,
        tspan,
        p = SLVector(; η, β),
        dt,
        seed = rand(Xoshiro(seed), UInt),
        alg,
        kwargs...
    ) |> assert_dimension(; order = 1)
end

const TemporalFractionalOverdampedLangevinEquation = tFOLE
export tFOLE, TemporalFractionalOverdampedLangevinEquation
