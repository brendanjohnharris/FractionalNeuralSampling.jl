function ole_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η = ps
    x = divide_dims(u, length(u))
    b = gradlogdensity(𝜋)(x)
    du .= only(η .* b)
end
function ole_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η = ps
    du .= sqrt(2 * only(η)) # ? × dW in the integrator.
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
             callback = (),
             alg = EM(),
             kwargs...)
    Sampler(ole_f!, ole_g!;
            callback = CallbackSet(boundaries(), callback...),
            u0,
            noise_rate_prototype,
            noise,
            tspan,
            p = SLVector(; η),
            alg,
            kwargs...)
end

const OverdampedLangevinEquation = OLE
export OLE, OverdampedLangevinEquation
