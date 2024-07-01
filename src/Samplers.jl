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
using ..Densities
import ..Densities.Density
import SciMLBase: AbstractRODEProblem, RODEFunction, NullParameters, prepare_initial_state, promote_tspan, warn_paramtype, @add_kwonly
export AbstractSampler, Sampler, LangevinSampler, parameters

abstract type AbstractSampler{uType,tType,isinplace,ND} <:
              AbstractRODEProblem{uType,tType,isinplace,ND} end

mutable struct Sampler{uType,tType,isinplace,P,NP,F,K,ND,D} <:
               AbstractSampler{uType,tType,isinplace,ND}
    f::F
    u0::uType
    tspan::tType
    p::Tuple{P,D} # = (params, 𝜋)
    noise::NP
    kwargs::K
    rand_prototype::ND
    seed::UInt64
end
parameters(S::Sampler) = first(S.p)
Density(S::Sampler) = last(S.p)

function default_distribution(u0::AbstractVector)
    if length(u0) < 3
        return Normal(0.0, 1.0)
    else
        MvNormal(zeros(length(u0) - 1), I(length(u0) - 1))
    end
end
function default_distribution(u0::Real)
    Normal(0.0, 1.0)
end
function Sampler{iip}(f::RODEFunction{iip}, u0, tspan,
    p=NullParameters(),
    𝜋=Density(default_distribution(u0)); # Assume momentum term
    rand_prototype=nothing,
    noise=nothing, seed=UInt64(0),
    kwargs...) where {iip}
    _u0 = prepare_initial_state(u0)
    _tspan = promote_tspan(tspan)
    warn_paramtype(p)
    Sampler{typeof(_u0),typeof(_tspan),
        isinplace(f),typeof(p),
        typeof(noise),typeof(f),typeof(kwargs),
        typeof(rand_prototype),typeof(𝜋)}(f, _u0, _tspan, (p, 𝜋), noise, kwargs,
        rand_prototype, seed)
end
function Sampler{iip}(f::RODEFunction{iip}; u0, tspan, p=NullParameters(), 𝜋=Density(default_distribution(u0)), kwargs...) where {iip}
    Sampler{iip}(f, u0, tspan, p, 𝜋; kwargs...)
end
function Sampler{iip}(; f, kwargs...) where {iip}
    Sampler{iip}(f; kwargs...)
end
function Sampler(f::RODEFunction, args...;
    kwargs...)
    Sampler{isinplace(f)}(f, args...; kwargs...)
end
function Sampler(f, args...; kwargs...)
    Sampler(RODEFunction{isinplace(f, 5)}(f), args...; kwargs...)
end




# * Langevin sampler (Brownian motion)
function langevin_sampler!(du, u, p, t, W)
    (β, γ), 𝜋 = p
    x, v = u
    b = gradlogdensity(𝜋)(x) # ? Should this be in-place
    du[1] = γ * b + β * v + γ^(1 // 2) * W[1] # W is a Wiener process, so α = 2
    du[2] = β * b
end

LangevinSampler(; β, γ, u0=[0.0, 0.0], rand_prototype=[zero(eltype(u0))], kwargs...) = Sampler(langevin_sampler!; kwargs..., u0, rand_prototype, p=(β, γ))

end # module
