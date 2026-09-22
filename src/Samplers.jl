"""
    Samplers

The sampler types and constructors. Each constructor returns a [`Sampler`](@ref), which
subtypes `SciMLBase.AbstractSDEProblem`.
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
import ..FractionalNeuralSampling: divide_dims, first_dims, serial_fftw
import ..NoiseProcesses: lfsn
import ..Solvers: CaputoEM, MultiCaputoEM, PositionalCaputoEM
using ..Densities
import ..Densities.Density
import ..Boundaries: boundary_init
import SciMLBase: AbstractSDEProblem, AbstractSDEFunction, NullParameters,
    prepare_initial_state,
    promote_tspan, warn_paramtype, @add_kwonly

import StochasticDiffEqLowOrder: EM

export AbstractSampler, Sampler, parameters

"""
    AbstractSampler{uType, tType, isinplace, ND} <: AbstractSDEProblem

Supertype of every sampler in the package. Since a sampler is an `SDEProblem`, it composes
with callbacks, ensembles, and the standard `solve` interface. [`Sampler`](@ref) is the
only concrete subtype; the constructors listed in the manual all return one.
"""
abstract type AbstractSampler{uType, tType, isinplace, ND} <:
AbstractSDEProblem{uType, tType, isinplace, ND} end

const compatible_solvers = (:EM, :CaputoEM, :MultiCaputoEM, :PositionalCaputoEM)

function SciMLBase.solve(P::AbstractSampler; kwargs...)
    return if haskey(P.kwargs, :alg)
        solve(P, P.kwargs[:alg]; kwargs...)
    else
        throw(ArgumentError("Use `solve(S::Sampler, alg; kwargs...)`. Compatible algorithms: $(join(compatible_solvers, ", "))"))
    end
end

const Labelled = Union{SLArray, LArray, NamedTuple}

"""
    Sampler <: AbstractSampler

An `SDEProblem` carrying a target density beside its parameters. The parameter field `p` is
the pair `(parameters, 𝜋)`, read back with [`parameters`](@ref) and `Density`.

Samplers are immutable. Calling one returns a copy with the named parameters, fields or
target replaced, which is how a parameter sweep is written:

```julia
S2 = S(; γ = 1.0)          # a new sampler; S is untouched
S3 = S(; u0 = [1.0, 0.0])  # fields work the same way
```

`remake` is also supported. Construct a sampler through one of the named constructors
([`Langevin`](@ref), [`FNS`](@ref), and the rest) rather than directly.
"""
struct Sampler{
        uType, tType, isinplace, P <: Labelled, NP, F, G, K, ND,
        D <: Union{AbstractDensity, Function},
    } <:
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
"""
Positional constructor, reached by `@set` and `ConstructionBase.setproperties`, and so by
every `remake` of a sampler.

`kwargs` is normalised to `Base.Pairs`, which the solver stack requires: `DiffEqBase` reads
`values(prob.kwargs)` and needs a `NamedTuple` back, whereas `values` of a plain
`NamedTuple` is a `Tuple`. Since DiffEqBase v7.21, `_erase_problem_callback_types` rebuilds
the field as a plain `NamedTuple`, so without this the first `solve` fails in
`merge_problem_kwargs`. The normalisation is a no-op on earlier versions.
"""
function Sampler{isinplace}(
        f::F, g::G, u0::uType, # For @set
        tspan::tType,
        p::Tuple{P, D},
        noise::NP, kwargs,
        noise_rate_prototype::ND,
        seed::UInt64
    ) where {
        uType,
        tType,
        isinplace,
        P <: Labelled,
        NP, F,
        G,
        ND, D,
    }
    _kwargs = kwargs isa Base.Pairs ? kwargs : pairs(kwargs)
    return Sampler{uType, tType, isinplace, P, NP, F, G, typeof(_kwargs), ND, D}(
        f, g, u0, tspan, p, noise,
        _kwargs, noise_rate_prototype,
        seed
    )
end
"""
    parameters(S::Sampler)

The parameter container of a sampler, holding the scalars its drift and diffusion read.

Plain samplers carry an `SLArray`, so `parameters(S).γ` works and the whole container is
static. The spectral samplers ([`sFOLE`](@ref), [`sFNS`](@ref), [`bFOLE`](@ref),
[`bFNS`](@ref)) and the adaptive ones carry a `NamedTuple` instead, since no static vector
holds the `ApproxFun` operators and transform plans they need beside their scalars.
"""
parameters(S::Sampler) = first(S.p)
Density(S::Sampler) = last(S.p)
SciMLBase.is_diagonal_noise(S::Sampler) = true

"""
    default_density(u0; dims)

A standard normal target over the first `dims` coordinates of `u0`; the fallback when a
sampler is constructed without a `𝜋`. `dims` defaults to half the state, as for the
second-order samplers that carry both position and momentum.
"""
function default_density(u0; dims = length(u0) ÷ 2)
    dims < 1 &&
        throw(ArgumentError("Cannot infer a default target from a state of length $(length(u0)); supply `𝜋`, or a `u0` long enough for the sampler's order"))
    x = first_dims(u0, dims)
    D = length(x) == 1 ? Normal(0.0, 1.0) : MvNormal(zeros(length(x)), I(length(x)))
    return Density(D)
end
function default_density(u0::Real; kwargs...)
    return Normal(0.0, 1.0) |> Density
end
function Sampler{iip}(
        f::AbstractSDEFunction{iip}, u0, tspan,
        p::Tuple{<:Labelled, D} = (
            NullParameters(),
            (default_density ∘ first)(u0),
        );
        noise_rate_prototype = nothing,
        noise = nothing,
        seed = UInt64(0),
        kwargs...
    ) where {iip, D <: Union{AbstractDensity, Function}}
    _u0 = prepare_initial_state(u0)
    _tspan = promote_tspan(tspan)
    warn_paramtype(p)
    return Sampler{
        typeof(_u0), typeof(_tspan),
        isinplace(f), typeof(first(p)),
        typeof(noise), typeof(f), typeof(f.g), typeof(kwargs),
        typeof(noise_rate_prototype), D,
    }(
        f, f.g, _u0, _tspan, p,
        noise,
        kwargs,
        noise_rate_prototype, seed
    )
end
function Sampler{iip}(
        f::AbstractSDEFunction{iip}; u0, tspan,
        p,
        kwargs...
    ) where {iip}
    return Sampler{iip}(f, u0, tspan, p; kwargs...)
end
function Sampler{iip}(; f, g = nothing, kwargs...) where {iip}
    return if f isa AbstractSDEFunction
        Sampler{iip}(f; kwargs...)
    elseif !isnothing(g)
        Sampler{iip}(f, g; kwargs...)
    else
        throw(ArgumentError("You must specify an f::AbstractSDEFunction, or both f and g"))
    end
end
function Sampler(
        f::AbstractSDEFunction, args...;
        kwargs...
    )
    return Sampler{isinplace(f)}(f, args...; kwargs...)
end
# function Sampler(f, g, args...; p, kwargs...)
#     Sampler(SDEFunction{isinplace(f, 4)}(f, g), args...; p, kwargs...)
# end
function Sampler(f, g, args...; p, 𝜋, kwargs...)
    p = (p, 𝜋)
    return Sampler(SDEFunction{isinplace(f, 4)}(f, g), args...; p, kwargs...)
end

"""
Replace named entries of a sampler's parameter container.

The plain samplers carry an `SLArray`. The spectral samplers (`sFOLE`, `sFNS`, `bFOLE`,
`bFNS`) and the adaptive ones carry ApproxFun operators, and a transform plan, beside their
scalars; no static vector holds those, so their parameters stay a `NamedTuple`.
"""
set_parameters(ps::NamedTuple; kwargs...) = merge(ps, values(kwargs))
set_parameters(ps; kwargs...) = SLVector(ps; kwargs...)

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
        ps = set_parameters(ps; kwargs[pkeys]...)
    end

    S = set(S, PropertyLens(:p), (ps, 𝜋))
    return S
end

function assert_dimension(u0; order, dimension)
    return if length(u0) != order * dimension
        throw(ArgumentError("Initial condition u0 must have length $(order * dimension) for dimension $dimension and order $order, got length $(length(u0))"))
    end
end

function assert_dimension(S::AbstractSampler; order)
    u0 = S.u0
    assert_dimension(u0; order, dimension = dimension(Density(S)))
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
