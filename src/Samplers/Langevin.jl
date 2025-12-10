function langevin_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack β, η = ps
    x, v = divide_dims(u, dimension(𝜋))
    b = gradlogdensity(𝜋)(x)
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= β .* v
    dv .= β .* b - η .* v
end
function langevin_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack β, η = ps
    dx, dv = divide_dims(du, dimension(𝜋))
    dx .= 0.0
    dv .= sqrt(2 * η)
end

"""
Langevin equation
"""
function Langevin(;
                  tspan,
                  β, # Momentum coupling parameter
                  η, # Noise strength
                  u0 = [0.0],
                  boundaries = nothing,
                  noise_rate_prototype = similar(u0),
                  noise = WienerProcess!(0.0, zero(u0)),
                  callback = (),
                  alg = EM(),
                  kwargs...)
    Sampler(langevin_f!, langevin_g!;
            callback = CallbackSet(boundary_init(boundaries), callback...),
            u0,
            noise_rate_prototype,
            noise,
            tspan,
            p = SLVector(; β, η),
            alg,
            kwargs...) |> assert_dimension(; order = 2)
end

const LangevinEquation = Langevin
export Langevin, LangevinEquation
