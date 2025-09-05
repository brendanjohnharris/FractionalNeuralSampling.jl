
export OFLSampler

function ofl_f!(du, u, p, t)
    (β, η), 𝜋 = p
    x = divide_dims(u, length(u))
    b = gradlogdensity(𝜋)(x)
    dx = divide_dims(du, length(du))
    dx .= η .* b
end
function ofl_g!(du, u, p, t)
    (β, η), 𝜋 = p
    dx = divide_dims(du, length(du))
    dx .= sqrt(2 * η) # ? × dW in the integrator.
end

function OFLSampler(;
                    tspan,
                    β, # Tail index of fractional temporal derivative
                    η, # Noise strength
                    u0 = [0.0],
                    boundaries = nothing,
                    noise_rate_prototype = similar(u0),
                    𝜋 = Density(default_density(first(u0))),
                    noise = WienerProcess!(0.0, zero(u0)),
                    kwargs...)
    Sampler(ofl_f!, ofl_g!;
            callback = boundaries,
            u0,
            noise_rate_prototype,
            noise,
            tspan,
            p = ((β, η), 𝜋),
            kwargs...)
end
