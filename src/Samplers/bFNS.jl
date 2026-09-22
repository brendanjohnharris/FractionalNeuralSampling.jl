function bfns_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, η, γ, ∇𝒟𝜋, 𝜋s, λ = ps
    x, v = divide_dims(u, dimension(𝜋))
    b = ∇𝒟𝜋(only(x)) / (𝜋s(only(x)) + λ)
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= η .* b .+ γ .* v
    return dv .= γ .* b
end
function bfns_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η, α = ps
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= η^(1 / α) # ? × dL in the integrator.
    return dv .= 0.0
end

function gen_lfsm_fns(α, β; tspan, dt, seed, nhist) # * 1D with zeros for momentum noise
    tmin = length(tspan) == 2 ? minimum(tspan) : 0
    tmax = maximum(tspan)
    H = one(α) / 2 - β / 2 + 1 / α
    N = Int((tmax - tmin) / dt) + 1
    x = cumsum(lfsn(N, α, H; dt, rng = Xoshiro(seed), M = nhist))
    x = hcat(x, zero(x)) # Zeros for momentum
    ts = range(tmin, step = dt, length = N)
    @assert last(ts) == tmax
    return NoiseGrid(ts, eachrow(x))
end

"""
    bFNS(; tspan, dt, α, β, γ, η, 𝜋, domain, u0 = [0.0, 0.0], kwargs...)

Bi-fractional neural sampling: [`sFNS`](@ref) with the position also fractional in time.
The fractional drift is taken in space at order α and the Caputo derivative in time at
order β, driven by linear fractional stable motion with H = 1/2 - β/2 + 1/α. The momentum
is advanced by plain Euler--Maruyama, which
[`PositionalCaputoEM`](@ref) provides. Aliased as `BiFractionalNeuralSampler`.

Note that the roles of the parameters differ from [`sFNS`](@ref): here `η` is the drift
strength and `γ` the momentum coupling.

# Arguments
- `tspan`: time span, as a tuple or a final time
- `dt`: step size, used to generate the noise grid
- `α`: fractional order in space, which is also the stability of the driving noise
- `β`: fractional order in time, in (0, 1]
- `γ`: coupling between position and momentum
- `η`: drift strength, which also scales the noise as ``η^{1/α}``
- `𝜋`: target [`Density`](@ref)
- `domain`: interval over which `𝜋` is expanded
- `τ`: history length for the solver and the noise; one tenth of the time span by default
- `u0`: initial `[position; momentum]`

Remaining keywords pass through to [`Sampler`](@ref).
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
        approx_n_modes = 1000,
        λ = 1.0e-4, # Regularization to avoid overflow in low-prob regions
        τ = length(tspan) == 2 ? (tspan[2] - tspan[1]) / 10 : tspan / 10, # History length for caputo and lfsn
        u0 = [0.0, 0.0],
        boundaries = nothing,
        seed = rand(UInt32),
        noise_rate_prototype = similar(u0),
        noise = gen_lfsm_fns(
            α, β; tspan, dt, seed = rand(Xoshiro(seed), UInt32),
            nhist = round(Int, τ / dt)
        ),
        alg = PositionalCaputoEM(β, round(Int, τ / dt)), # Should match the order of the noise
        callback = (),
        kwargs...
    )
    if length(tspan) == 1
        tspan = (0, tspan...)
    end
    ∇𝒟𝜋, 𝜋s = space_fractional_drift(𝜋; α, domain, approx_n_modes)
    return Sampler(
        bfns_f!, bfns_g!;
        callback = CallbackSet(boundary_init(boundaries), callback...),
        u0,
        noise_rate_prototype,
        noise,
        tspan,
        dt,
        p = (; α, β, γ, η, ∇𝒟𝜋, 𝜋s, λ),
        𝜋,
        alg,
        seed,
        kwargs...
    ) |> assert_dimension(; order = 2)
end

const BiFractionalNeuralSampler = bFNS
export bFNS, BiFractionalNeuralSampler
