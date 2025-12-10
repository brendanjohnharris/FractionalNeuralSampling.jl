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
using LabelledArrays
using UnPack
using Accessors
using Random

import ..NoiseProcesses
import ..FractionalNeuralSampling: divide_dims
import ..NoiseProcesses: lfsn
import ..Solvers: CaputoEM, MultiCaputoEM, PositionalCaputoEM
using ..Densities
import ..Densities.Density
import ..Boundaries: boundary_init
import SciMLBase: AbstractSDEProblem, AbstractSDEFunction, NullParameters,
    prepare_initial_state,
    promote_tspan, warn_paramtype, @add_kwonly

import StochasticDiffEq: EM

export AbstractSampler, Sampler, parameters

abstract type AbstractSampler{uType,tType,isinplace,ND} <:
              AbstractSDEProblem{uType,tType,isinplace,ND} end

const compatible_solvers = (:EM, :CaputoEM)

function SciMLBase.solve(P::AbstractSampler; kwargs...)
    if haskey(P.kwargs, :alg)
        solve(P, P.kwargs[:alg]; kwargs...)
    else
        throw(ArgumentError("Use `solve(S::Sampler, alg; kwargs...)`. Compatible algorithms: $(join(compatible_solvers, ", "))"))
    end
end

const Labelled = Union{SLArray,LArray,NamedTuple}

struct Sampler{uType,tType,isinplace,P<:Labelled,NP,F,G,K,ND,
    D<:Union{AbstractDensity,Function}} <:
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
function Sampler{isinplace}(f::F, g::G, u0::uType, # For @set
    tspan::tType,
    p::Tuple{P,D},
    noise::NP, kwargs::K,
    noise_rate_prototype::ND,
    seed::UInt64) where {uType,
    tType,
    isinplace,
    P<:Labelled,
    NP,F,
    G,K,
    ND,D}
    Sampler{uType,tType,isinplace,P,NP,F,G,K,ND,D}(f, g, u0, tspan, p, noise,
        kwargs, noise_rate_prototype,
        seed)
end
parameters(S::Sampler) = first(S.p)
Density(S::Sampler) = last(S.p)
SciMLBase.is_diagonal_noise(S::Sampler) = true

function default_density(u0; dims=length(u0) ÷ 2)
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
    p::Tuple{<:Labelled,D}=(NullParameters(),
        (default_density ∘ first)(u0));
    noise_rate_prototype=nothing,
    noise=nothing,
    seed=UInt64(0),
    kwargs...) where {iip,D<:Union{AbstractDensity,Function}}
    _u0 = prepare_initial_state(u0)
    _tspan = promote_tspan(tspan)
    warn_paramtype(p)
    Sampler{typeof(_u0),typeof(_tspan),
        isinplace(f),typeof(first(p)),
        typeof(noise),typeof(f),typeof(f.g),typeof(kwargs),
        typeof(noise_rate_prototype),D}(f, f.g, _u0, _tspan, p,
        noise,
        kwargs,
        noise_rate_prototype, seed)
end
function Sampler{iip}(f::AbstractSDEFunction{iip}; u0, tspan,
    p,
    kwargs...) where {iip}
    Sampler{iip}(f, u0, tspan, p; kwargs...)
end
function Sampler{iip}(; f, g=nothing, kwargs...) where {iip}
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
# function Sampler(f, g, args...; p, kwargs...)
#     Sampler(SDEFunction{isinplace(f, 4)}(f, g), args...; p, kwargs...)
# end
function Sampler(f, g, args...; p, 𝜋, kwargs...)
    p = (p, 𝜋)
    Sampler(SDEFunction{isinplace(f, 4)}(f, g), args...; p, kwargs...)
end

function (S::AbstractSampler)(; kwargs...)
    # First update any direct fields of the sampler
    for k in filter(k -> k ∈ propertynames(S), keys(kwargs))
        S = set(S, PropertyLens(k), kwargs[k])
    end

    # * Deepcopy parameters
    if haskey(kwargs, :p)
        return @set S.p = kwargs[:p]
    end

    if haskey(kwargs, :𝜋)
        𝜋 = kwargs[:𝜋]
    else
        𝜋 = Density(S)
    end

    ps = parameters(S)
    pkeys = filter(k -> k in keys(ps), keys(kwargs))
    if !isempty(pkeys)
        ps = deepcopy(ps)
        ps = SLVector(ps; kwargs[pkeys]...)
    end

    S = set(S, PropertyLens(:p), (ps, 𝜋))
    return S
end

function assert_dimension(u0; order, dimension)
    if length(u0) != order * dimension
        throw(ArgumentError("Initial condition u0 must have length $(order * dimension) for dimension $dimension and order $order, got length $(length(u0))"))
    end
end

function assert_dimension(S::AbstractSampler; order)
    u0 = S.u0
    assert_dimension(u0; order, dimension=dimension(Density(S)))
    return S
end

assert_dimension(; order) = x -> assert_dimension(x; order)

include("Samplers/Langevin.jl")
include("Samplers/OLE.jl")
include("Samplers/tFOLE.jl")
include("Samplers/sFOLE.jl")
include("Samplers/bFOLE.jl")
include("Samplers/FHMC.jl")
include("Samplers/FNS.jl")
include("Samplers/sFNS.jl")
include("Samplers/bFNS.jl")
include("Samplers/AdaptiveSamplers.jl")
end # module
