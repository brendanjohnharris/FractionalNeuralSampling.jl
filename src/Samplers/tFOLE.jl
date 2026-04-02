
function tfole_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η = ps
    x = divide_dims(u, dimension(𝜋))
    b = gradlogdensity(𝜋)(x)
    du .= only(η .* b)
end
function tfole_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η = ps
    du .= only(η) # ? × dW in the integrator.
end

function gen_fbm(β; u0, tspan, dt, seed) # * 1d for now
    α = 2
    tmin = length(tspan) == 2 ? minimum(tspan) : 0
    tmax = maximum(tspan)
    H = 1 - β / 2
    N = Int(tspan / dt) + 1
    x = cumsum(lfsn(N, α, H; dt, rng=Xoshiro(seed)))
    ts = range(tmin, step=dt, length=N)
    @assert last(ts) == tmax
    return NoiseGrid(ts, x)
end

"""
Overdamped langevin equation
"""
function tFOLE(;
    tspan,
    dt,
    η, # Noise strength
    β, # Fractional order in time
    u0=[0.0],
    boundaries=nothing,
    noise_rate_prototype=similar(u0),
    seed=rand(UInt32),
    noise=gen_fbm(β; u0, tspan, dt, seed),
    callback=(),
    alg=CaputoEM(β, 1000),
    kwargs...)
    Sampler(tfole_f!, tfole_g!;
        callback=CallbackSet(boundary_init(boundaries), callback...),
        u0,
        noise_rate_prototype,
        noise,
        tspan,
        p=SLVector(; η, β),
        dt,
        seed=rand(Xoshiro(seed), UInt),
        alg,
        kwargs...) |> assert_dimension(; order=1)
end

const TemporalFractionalOverdampedLangevinEquation = tFOLE
export tFOLE, TemporalFractionalOverdampedLangevinEquation
