import ApproxFun: Operator, Derivative, Fun, Fourier, Laplacian, ApproxFunBase
import ..FractionalNeuralSampling: Power
using StaticArraysCore

function space_fractional_deriv(::Val{1}; α, domain)
    S = Fourier(domain) # Could use Laurent for complex functions
    D = Derivative(S, 1)
    Δ = maybeLaplacian(S)
    @assert isdiag(Δ)
    @assert all([Δ[i, i] for i in 1:length(100)] .<= 0.0) # * Should be negative for Fourier domain
    𝒟 = Power(-Δ, (α - 2) / 2) # The fractional LAPLACIAN
    return S, D, 𝒟
end
function space_fractional_deriv(::Val{2}; α, domain)
    S = reduce(*, map(Fourier, domain))  # Could use Laurent for complex functions
    D = [Derivative(S, SVector{2}([1, 0])); Derivative(S, SVector{2}([0, 1]))]
    Δ = maybeLaplacian(S)
    @assert isdiag(Δ)
    @assert all([Δ[i, i] for i in 1:length(100)] .<= 0.0) # * Should be negative for Fourier domain
    𝒟 = Power(-Δ, (α - 2) / 2) # The fractional LAPLACIAN
    return S, D, 𝒟
end

function space_fractional_drift(𝜋; approx_n_modes = 10000, kwargs...)
    S, D, 𝒟 = space_fractional_deriv(Val(dimension(𝜋)); kwargs...)
    𝜋s = Fun(𝜋, S, approx_n_modes)
    ∇𝒟𝜋 = D * 𝒟 * 𝜋s
    return ∇𝒟𝜋, 𝜋s
end

function sfole_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η, α, ∇𝒟𝜋, 𝜋s, λ = ps
    x = divide_dims(u, dimension(𝜋)) |> only
    b = ∇𝒟𝜋(only(x)) / (𝜋s(only(x)) + λ)
    du .= only(η .* b)
end
function sfole_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack η, α = ps
    du .= only(η)^(1 / α) # ? × dW in the integrator.
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
               domain, # An Interval
               λ = 0.001, # Regularization to avoid overflow in low-prob regions
               u0 = [0.0],
               boundaries = nothing,
               noise_rate_prototype = similar(u0),
               noise = NoiseProcesses.LevyProcess!(α; ND = dimension(𝜋),
                                                   W0 = zero(u0)),
               approx_n_modes = 1000,
               alg = EM(),
               callback = (),
               kwargs...)
    ∇𝒟𝜋, 𝜋s = space_fractional_drift(𝜋; α, domain, approx_n_modes)
    Sampler(sfole_f!, sfole_g!;
            callback = CallbackSet(boundary_init(boundaries), callback...),
            u0,
            noise_rate_prototype,
            noise,
            tspan,
            p = (; η, α, ∇𝒟𝜋, 𝜋s, λ),
            𝜋,
            alg,
            kwargs...) |> assert_dimension(; order = 1)
end

const SpaceFractionalOverdampedLangevinEquation = sFOLE
export sFOLE, SpaceFractionalOverdampedLangevinEquation
