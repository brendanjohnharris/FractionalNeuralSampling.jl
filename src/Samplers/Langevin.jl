function langevin_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack β, η = ps
    x, v = divide_dims(u, dimension(𝜋))
    b = gradlogdensity(𝜋)(x)
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= β .* v
    return dv .= β .* b - η .* v
end
function langevin_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack β, η = ps
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= 0.0
    return dv .= sqrt(2 * η)
end

"""
    Langevin(; tspan, β, η, u0 = [0.0, 0.0], 𝜋 = default_density(u0), kwargs...)

Underdamped Langevin dynamics driven by Brownian noise, with stationary density `𝜋`.

```math
\\mathrm{d}x = \\beta v \\, \\mathrm{d}t, \\qquad
\\mathrm{d}v = (\\beta \\nabla \\log \\pi(x) - \\eta v) \\, \\mathrm{d}t + \\sqrt{2\\eta} \\, \\mathrm{d}W
```

Second order, so `u0` stacks position and momentum and `length(u0) == 2 * dimension(𝜋)`.
Aliased as `LangevinEquation`.

# Arguments
- `tspan`: time span, as a tuple or a final time
- `β`: coupling between position and momentum
- `η`: damping rate, which is also the noise strength
- `u0`: initial `[position; momentum]`
- `𝜋`: target [`Density`](@ref); defaults to a standard normal over the position
- `boundaries`: an [`AbstractBoundary`](@ref), or `nothing`
- `alg`: default solver, `EM()`

Remaining keywords pass through to [`Sampler`](@ref).
"""
function Langevin(;
        tspan,
        β, # Momentum coupling parameter
        η, # Noise strength
        u0 = [0.0, 0.0],
        boundaries = nothing,
        noise_rate_prototype = similar(u0),
        noise = WienerProcess!(0.0, zero(u0)),
        𝜋 = default_density(u0),
        callback = (),
        alg = EM(),
        kwargs...
    )
    return Sampler(
        langevin_f!, langevin_g!;
        𝜋,
        callback = CallbackSet(boundary_init(boundaries), callback...),
        u0,
        noise_rate_prototype,
        noise,
        tspan,
        p = SLVector(; β, η),
        alg,
        kwargs...
    ) |> assert_dimension(; order = 2)
end

const LangevinEquation = Langevin
export Langevin, LangevinEquation
