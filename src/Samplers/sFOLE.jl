import ApproxFun: Operator, Derivative, Fun, Fourier, Laplacian, ApproxFunBase
import ..FractionalNeuralSampling: Power

function sfole_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η, α, ∇𝒟𝜋 = ps
    x = divide_dims(u, length(u)) |> only
    # b = ∇𝒟𝜋(only(x)) / 𝜋(x) # For 1D atm
    b = gradlogdensity(𝜋)(x)
    du .= only(η .* b)
end
function sfole_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η, α = ps
    du .= sqrt(2) * only(η)^(1 / α) # ? × dW in the integrator. ! WHy sqrt(2)???
end

function maybeLaplacian(S::ApproxFunBase.DirectSumSpace) # * If the space is 1D, use regular second derivative
    Derivative(S, 2)
end
function maybeLaplacian(S::ApproxFunBase.AbstractProductSpace) # * If the space is multidimensional, use Laplacian
    Laplacian(S)
end

"""
Space fractional overdamped langevin equation
"""
function sFOLE(;
               tspan,
               η, # Noise strength
               α, # Fractional order
               𝜋, # Target distribution
               u0 = [0.0],
               boundaries = nothing,
               noise_rate_prototype = similar(u0),
               noise = NoiseProcesses.LevyProcess!(α; ND = dimension(𝜋),
                                                   W0 = zero(u0)),
               callback = (),
               approx_n_modes = 1000,
               domain, # An Interval
               kwargs...)
    S = Fourier(domain) # Could use Laurent for complex functions
    D = Derivative(S, 1)
    Δ = maybeLaplacian(S)
    @assert isdiag(Δ)
    @assert all([Δ[i, i] for i in 1:length(100)] .<= 0.0) # * Should be negative for Fourier domain
    𝒟 = Power(-Δ, (α - 2) / 2) # The fractional LAPLACIAN
    𝜋s = Fun(𝜋, S, approx_n_modes)
    ∇𝒟𝜋 = D * 𝒟 * 𝜋s

    Sampler(sfole_f!, sfole_g!;
            callback = CallbackSet(boundaries, callback...),
            u0,
            noise_rate_prototype,
            noise,
            tspan,
            p = (; η, α, ∇𝒟𝜋),
            𝜋,
            kwargs...)
end

const SpaceFractionalOverdampedLangevinEquation = sFOLE
export sFOLE, SpaceFractionalOverdampedLangevinEquation
