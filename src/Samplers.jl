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

import SciMLBase: AbstractRODEProblem, RODEFunction, @add_kwonly
export AbstractSampler, Sampler

abstract type AbstractSampler{uType, tType, isinplace, ND} <:
              AbstractRODEProblem{uType, tType, isinplace, ND} end

mutable struct Sampler{uType, tType, isinplace, P, NP, F, K, ND, D} <:
               AbstractSampler{uType, tType, isinplace, ND}
    f::F
    u0::uType
    tspan::tType
    p::P
    noise::NP
    kwargs::K
    rand_prototype::ND
    seed::UInt64
    𝜋::D
    @add_kwonly function Sampler{iip}(f::RODEFunction{iip}, u0, tspan,
                                      p = NullParameters(),
                                      𝜋 = MvNormal(zeros(length(u)), I(length(u)));
                                      rand_prototype = nothing,
                                      noise = nothing, seed = UInt64(0),
                                      kwargs...) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        new{typeof(_u0), typeof(_tspan),
            isinplace(f), typeof(p),
            typeof(noise), typeof(f), typeof(kwargs),
            typeof(rand_prototype), typeof(𝜋)}(f, _u0, _tspan, p, noise, kwargs,
                                               rand_prototype, seed, 𝜋)
    end
    function Sampler{iip}(f, args...;
                          kwargs...) where {iip}
        Sampler(RODEFunction{iip}(f), args...; kwargs...)
    end
end

function Sampler(f::RODEFunction, args...;
                 kwargs...)
    Sampler{isinplace(f)}(f, args...; kwargs...)
end

function Sampler(f, args...; kwargs...)
    Sampler(RODEFunction(f), args...; kwargs...)
end


end
