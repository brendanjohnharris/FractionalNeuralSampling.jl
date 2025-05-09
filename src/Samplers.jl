"""
    Sampler
Defines the `DifferentialEquations`-compatible FNS sampling algorithm.
"""
module Samplers
using SciMLBase
using DiffEqNoiseProcess
using SciMLBase
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
       OverdampedLangevinSampler, UnderdampedLangevinSampler,
       LevyFlightSampler, LevyWalkSampler

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
SciMLBase.is_diagonal_noise(S::Sampler) = true

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

include("Samplers/LangevinSampler.jl")
include("Samplers/OverdampedLangevinSampler.jl")
include("Samplers/FractionalHMC.jl")
include("Samplers/FractionalNeuralSampler.jl")
include("Samplers/AdaptiveSamplers.jl")
end # module
