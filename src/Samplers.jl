"""
    Sampler
Defines the `DifferentialEquations`-compatible FNS sampling algorithm.
"""
module Samplers
using SciMLBase
using DiffEqNoiseProcess
using SciMLBase
using SpecialFunctions
using StaticArrays
using StochasticDiffEq
using LogDensityProblems
using Distributions
using LinearAlgebra
import ..NoiseProcesses
import ..FractionalNeuralSampling.divide_dims
using ..Densities
import ..Densities.Density
import SciMLBase: AbstractSDEProblem, AbstractSDEFunction, NullParameters,
                  prepare_initial_state,
                  promote_tspan, warn_paramtype, @add_kwonly
export AbstractSampler, Sampler, parameters,
       LangevinSampler, LevyFlightSampler, LevyWalkSampler

abstract type AbstractSampler{uType, tType, isinplace, ND} <:
              AbstractSDEProblem{uType, tType, isinplace, ND} end
function SciMLBase.solve(P::AbstractSampler; kwargs...)
    SciMLBase.solve(P, EM(); kwargs...)
end
mutable struct Sampler{uType, tType, isinplace, P, NP, F, G, K, ND, D} <:
               AbstractSampler{uType, tType, isinplace, ND}
    f::F
    g::G
    u0::uType
    tspan::tType
    p::Tuple{P, D} # = (params, 𝜋)
    noise::NP
    kwargs::K
    noise_rate_prototype::ND
    seed::UInt64
end
parameters(S::Sampler) = first(S.p)
Density(S::Sampler) = last(S.p)

function default_density(u0; dims = length(u0) ÷ 2)
    u0 = divide_dims(u0, dims) |> first
    if length(u0) == 1
        D = Normal(0.0, 1.0)
    else
        D = MvNormal(zeros(length(u0)), I(length(u0)))
    end
    D = Density(D)
    return D
end
function default_density(u0::Real)
    Normal(0.0, 1.0) |> Density
end
function Sampler{iip}(f::AbstractSDEFunction{iip}, u0, tspan,
                      p = (NullParameters(), Density(default_density(first(u0)))); # Assume momentum term
                      noise_rate_prototype = nothing,
                      noise = nothing,
                      seed = UInt64(0),
                      kwargs...) where {iip}
    _u0 = prepare_initial_state(u0)
    _tspan = promote_tspan(tspan)
    warn_paramtype(p)
    Sampler{typeof(_u0), typeof(_tspan),
            isinplace(f), typeof(first(p)),
            typeof(noise), typeof(f), typeof(f.g), typeof(kwargs),
            typeof(noise_rate_prototype), typeof(last(p))}(f, f.g, _u0, _tspan, p,
                                                           noise,
                                                           kwargs,
                                                           noise_rate_prototype, seed)
end
function Sampler{iip}(f::AbstractSDEFunction{iip}; u0, tspan,
                      p = (NullParameters(), Density(default_density(first(u0)))),
                      kwargs...) where {iip}
    Sampler{iip}(f, u0, tspan, p; kwargs...)
end
function Sampler{iip}(; f, g = nothing, kwargs...) where {iip}
    if f isa AbstractSDEFunction
        Sampler{iip}(f; kwargs...)
    elseif !isnothing(g)
        Sampler{iip}(f, g; kwargs...)
    else
        throw(ArgumentError("You must specify an f::AbstractSDEFunction, or both f and g"))
    end
end
function Sampler(f::AbstractSDEFunction, args...;
                 kwargs...)
    Sampler{isinplace(f)}(f, args...; kwargs...)
end
function Sampler(f, g, args...; kwargs...)
    Sampler(SDEFunction{isinplace(f, 4)}(f, g), args...; kwargs...)
end

# * Langevin sampler (Brownian motion)
function langevin_f!(du, u, p, t)
    (β, γ), 𝜋 = p
    x, v = eachcol(u)
    b = gradlogdensity(𝜋)(x) # ? Should this be in-place
    du[:, 1] .= γ .* b .+ β .* v
    du[:, 2] .= β .* b
end
function langevin_g!(du, u, p, t)
    (β, γ), 𝜋 = p
    dx, dv = eachcol(du)
    dx .= sqrt(2) * γ^(1 // 2) # * dW
    dv .= 0.0
end

function LangevinSampler(; tspan, β, γ, u0 = [0.0 0.0], boundaries = nothing,
                         noise_rate_prototype = nothing,
                         𝜋 = Density(default_density(first(u0))),
                         noise = nothing,#WienerProcess(first(tspan), zero(noise_rate_prototype)),
                         kwargs...)
    Sampler(langevin_f!, langevin_g!; callback = boundaries, kwargs..., u0,
            noise_rate_prototype, noise,
            tspan, p = ((β, γ), 𝜋))
end

# * Modulated langevin sampler
# function modulated_langevin_f!(du, u, p, t)
#     (β, γ), 𝜋 = p
#     x, v = eachcol(u)
#     b = gradlogdensity(𝜋)(x) # ? Should this be in-place
#     du[:, 1] .= γ .* b .+ β .* v
#     du[:, 2] .= β .* b
# end
# function modulated_langevin_g!(du, u, p, t)
#     (β, γ), 𝜋 = p
#     dx, dv = eachcol(du)
#     dx .= sqrt(2) * γ^(1 // 2) # * dW
#     dv .= 0.0
# end

# function ModulatedLangevinSampler(; tspan, β, γ, u0 = [0.0 0.0], boundaries = nothing,
#                                   noise_rate_prototype = nothing,
#                                   𝜋 = Density(default_density(first(u0))),
#                                   noise = nothing,#WienerProcess(first(tspan), zero(noise_rate_prototype)),
#                                   kwargs...)
#     Sampler(langevin_f!, langevin_g!; callback = boundaries, kwargs..., u0,
#             noise_rate_prototype, noise,
#             tspan, p = ((β, γ), 𝜋))
# end

# * Levy flight sampler (noise on position)
function levy_flight_f!(du, u, p, t)
    (α, β, γ), 𝜋 = p
    x, v = divide_dims(u, length(u) ÷ 2)
    b = gradlogdensity(𝜋)(x) * gamma(α - 1) / (gamma(α / 2) .^ 2) # ? Should this be in-place
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= γ .* b .+ β .* v
    dv .= β .* b
end
function levy_flight_g!(du, u, p, t)
    (α, β, γ), 𝜋 = p
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= sqrt(2) * γ^(1 / α) # ? × dL in the integrator. This is matrix multiplication
    dv .= 0.0
end

function LevyFlightSampler(;
                           tspan, α, β, γ, u0 = [0.0 0.0],
                           boundaries = nothing,
                           noise_rate_prototype = zeros(size(u0)),
                           𝜋 = Density(default_density(first(u0))),
                           noise = NoiseProcesses.LevyProcess!(α; ND = 2,
                                                               W0 = Diagonal(zeros(length(u0),
                                                                                   length(u0)))),
                           kwargs...)
    Sampler(levy_flight_f!, levy_flight_g!; callback = boundaries, kwargs..., u0,
            noise_rate_prototype, noise,
            tspan, p = ((α, β, γ), 𝜋))
end

# !! Need to update equations, and check they are consistent
# * Levy walk sampler (noise on velocity)
function levy_walk_f!(du, u, p, t)
    (α, β, γ), 𝜋 = p
    x, v = divide_dims(u, length(u) ÷ 2)
    b = gradlogdensity(𝜋)(x) * gamma(α - 1) / (gamma(α / 2) .^ 2) # ? Should this be in-place
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= β .* v
    dv .= β .* b - γ .* v
end
function levy_walk_g!(du, u, p, t)
    (α, β, γ), 𝜋 = p
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= 0.0
    dv .= sqrt(2) * γ^(1 / α) # ? × dL in the integrator. This is matrix multiplication
end

function LevyWalkSampler(;
                         tspan, α, β, γ, u0 = [0.0 0.0],
                         boundaries = nothing,
                         noise_rate_prototype = zeros(size(u0)),
                         𝜋 = Density(default_density(first(u0))),
                         noise = NoiseProcesses.LevyProcess!(α; ND = 2,
                                                             W0 = Diagonal(zeros(length(u0),
                                                                                 length(u0)))),
                         kwargs...)
    Sampler(levy_walk_f!, levy_walk_g!; callback = boundaries, kwargs..., u0,
            noise_rate_prototype, noise,
            tspan, p = ((α, β, γ), 𝜋))
end

end # module
