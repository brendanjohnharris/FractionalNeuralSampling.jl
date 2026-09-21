function ole_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η = ps
    x = first_dims(u, dimension(𝜋))
    b = gradlogdensity(𝜋, x)
    return du .= η .* b
end
function ole_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η = ps
    return du .= sqrt(2 * only(η)) # ? × dW in the integrator.
end

"""
Overdamped langevin equation
"""
function OLE(;
        tspan,
        η, # Noise strength
        u0 = [0.0],
        boundaries = nothing,
        noise_rate_prototype = similar(u0),
        noise = WienerProcess!(0.0, zero(u0)),
        𝜋 = default_density(u0; dims = length(u0)),
        callback = (),
        alg = EM(),
        kwargs...
    )
    return Sampler(
        ole_f!, ole_g!;
        𝜋,
        callback = CallbackSet(boundary_init(boundaries), callback...),
        u0,
        noise_rate_prototype,
        noise,
        tspan,
        p = SLVector(; η),
        alg,
        kwargs...
    ) |> assert_dimension(; order = 1)
end

const OverdampedLangevinEquation = OLE
export OLE, OverdampedLangevinEquation
