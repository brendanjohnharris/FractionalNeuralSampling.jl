"""
    Sampler
Defines the `DifferentialEquations`-compatible FNS sampling algorithm.
"""
module Samplers
using SciMLBase
using DiffEqNoiseProcess
using SciMLBase
using StaticArrays
using LogDensityProblems
using Distributions
using LinearAlgebra
using ..Densities
import ..Densities.Density
import SciMLBase: AbstractSDEProblem, AbstractSDEFunction, NullParameters, prepare_initial_state,
    promote_tspan, warn_paramtype, @add_kwonly
export AbstractSampler, Sampler, LangevinSampler, parameters

abstract type AbstractSampler{uType,tType,isinplace,ND} <:
              AbstractSDEProblem{uType,tType,isinplace,ND} end

mutable struct Sampler{uType,tType,isinplace,P,NP,F,G,K,ND,D} <:
               AbstractSampler{uType,tType,isinplace,ND}
    f::F
    g::G
    u0::uType
    tspan::tType
    p::Tuple{P,D} # = (params, 𝜋)
    noise::NP
    kwargs::K
    noise_rate_prototype::ND
    seed::UInt64
end
parameters(S::Sampler) = first(S.p)
Density(S::Sampler) = last(S.p)

function default_distribution(u0::AbstractVector)
    if length(u0) == 1
        return Normal(0.0, 1.0)
    else
        MvNormal(zeros(length(u0)), I(length(u0)))
    end
end
function default_distribution(u0::Real)
    Normal(0.0, 1.0)
end
function Sampler{iip}(f::AbstractSDEFunction{iip}, u0::AbstractMatrix, tspan,
    p=NullParameters(),
    𝜋=Density(default_distribution(first(u0))); # Assume momentum term
    noise_rate_prototype=nothing,
    noise=nothing,
    seed=UInt64(0),
    kwargs...) where {iip}
    _u0 = prepare_initial_state(u0)
    _tspan = promote_tspan(tspan)
    warn_paramtype(p)
    Sampler{typeof(_u0),typeof(_tspan),
        isinplace(f),typeof(p),
        typeof(noise),typeof(f),typeof(f.g),typeof(kwargs),
        typeof(noise_rate_prototype),typeof(𝜋)}(f, f.g, _u0, _tspan, (p, 𝜋), noise, kwargs,
        noise_rate_prototype, seed)
end
function Sampler{iip}(f::AbstractSDEFunction{iip}; u0, tspan, p=NullParameters(),
    𝜋=Density(default_distribution(first(u0))), kwargs...) where {iip}
    Sampler{iip}(f, u0, tspan, p, 𝜋; kwargs...)
end
# function Sampler{iip}(; f, kwargs...) where {iip}
#     Sampler{iip}(f; kwargs...)
# end
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
    dx .= γ^(1 // 2) # * dW
    dv .= 0.0
end

function LangevinSampler(; tspan, β, γ, u0=[0.0; 0.0], boundaries=nothing,
    noise_rate_prototype=nothing,
    noise=nothing,#WienerProcess(first(tspan), zero(noise_rate_prototype)),
    kwargs...)
    Sampler(langevin_f!, langevin_g!; callback=boundaries, kwargs..., u0, noise_rate_prototype, noise,
        tspan, p=(β, γ))
end

end # module
