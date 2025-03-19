module Densities
using Distributions
using LogDensityProblems
using TransformVariables
using DifferentiationInterface
using TransformedLogDensities
import LogDensityProblems: logdensity, logdensity_and_gradient, dimension, capabilities
import Distributions: gradlogpdf
import FractionalNeuralSampling.AD_BACKEND

export AbstractDensity, AbstractUnivariateDensity, Density, GradDensity, PotentialDensity

export potential, logdensity, gradlogdensity, gradpotential, dimension

abstract type AbstractDensity{D, N, doAd} end
const AbstractUnivariateDensity{D, doAd} = AbstractDensity{D, 1, doAd} where {D, doAD}

logdensity(D::AbstractDensity, x) = logdensity(D)(x)
gradlogdensity(D::AbstractDensity) = Base.Fix1(gradlogdensity, D)

potential(D::AbstractDensity, x) = -logdensity(D, x)
potential(D::AbstractDensity, x::Tuple) = potential(D, collect(x))
potential(D::AbstractDensity) = Base.Fix1(potential, D)

gradpotential(D::AbstractDensity, x) = -gradlogdensity(D, x)
gradpotential(D::AbstractDensity) = (-) ∘ gradlogdensity(D)

(D::AbstractDensity)(x) = density(D)(x)
(D::AbstractDensity)(x::Tuple) = D(collect(x))
(D::AbstractUnivariateDensity)(x::AbstractVector) = D(only(x))

function LogDensityProblems.dimension(d::AbstractDensity{D, N, doAd}) where {D, N,
                                                                             doAd}
    return N
end
doautodiff(d::AbstractDensity{D, N, doAd}) where {D, N, doAd} = doAd
export dimension

# * Automatic autodiff
const AdDensity{D} = AbstractDensity{D, N, true} where {D, N}
function _gradlogdensity(D::AdDensity, x::Real)
    gradient(x -> logdensity(D, only(x)), AD_BACKEND, [x]) |> only
end
function _gradlogdensity(D::AdDensity, x::AbstractVector{<:Real})
    f = logdensity(D)
    extras = prepare_gradient(f, AD_BACKEND, x)
    gradient(f, extras, AD_BACKEND, x)
end
function _gradlogdensity(D::AdDensity,
                         x::AbstractVector{<:AbstractVector{T}}) where {T}
    f = logdensity(D)
    extras = prepare_gradient(f, AD_BACKEND, first(x))
    grad = map(similar, x)
    map(grad, x) do _grad, _x
        gradient!(f, _grad, extras, AD_BACKEND, _x)
    end
    return grad
end

function gradlogdensity(d::AbstractUnivariateDensity, x::T) where {T <: Real}
    _gradlogdensity(d, x)::T
end
function gradlogdensity(d::AbstractUnivariateDensity,
                        x::AbstractVector{T}) where {T <: Real} # For 1 element vectors
    convert(Vector{T}, [_gradlogdensity(d, only(x))])
end
function gradlogdensity(d::AbstractDensity, x)
    _gradlogdensity(d, x)
end
function logdensity_and_gradient(D::AbstractDensity, x)
    (logdensity(D, x), gradlogdensity(D, x))
end

begin # * See here for the Density interface: define these methods and traits. Custom differentiation functions can also be added; see Distributions.jl
    struct Density{D, N, doAd} <: AbstractDensity{D, N, doAd}
        density::D
    end
    function Density{N, doAd}(density::D) where {D, N, doAd}
        Density{D, N, doAd}(density)
    end
    function Density{N}(density::D, doAd::Bool) where {D, N}
        Density{N, doAd}(density)
    end
    function Density{N}(density::D) where {D, N}
        Density{N, true}(density) # Default to true autodiff
    end
    capabilities(::Type{<:Density}) = LogDensityProblems.LogDensityOrder{1}()
    density(D::Density) = D.density
    logdensity(D::Density) = log ∘ density(D)
end
begin # ! PotentialDensity (supply a potential, get a POTENTIALLY NON_NORMALIZED density)
    struct PotentialDensity{D, N, doAd} <: AbstractDensity{D, N, doAd}
        potential::D
    end
    PotentialDensity{N}(potential::D) where {D, N} = PotentialDensity{D, N, true}(potential)

    capabilities(::Type{<:PotentialDensity}) = LogDensityProblems.LogDensityOrder{1}()

    potential(D::PotentialDensity) = D.potential
    logdensity(D::PotentialDensity) = (-) ∘ potential(D)
    gradpotential(D::PotentialDensity) = (.-) ∘ gradlogdensity(D)
    density(D::PotentialDensity) = exp ∘ (-) ∘ potential(D)
end
# ? Density interface: just need to define the following methods and traits. E.g. for
# ? Densities with supplied gradients:
# begin # * GradDensity
#     struct GradDensity{D, G} <: AbstractDensity{D, false} # You are supplying a gradient function, so don't autodiff
#         density::D # Should be f([x, y]) -> d
#         gradlogdensity::G # Should be f([x, y]) -> [∂x, ∂y]
#         dimension::Int
#     end
#     GradDensity(d::D, g::G; dimension) where {D, G} = GradDensity{D, G}(d, g, dimension)

#     gradlogdensity(d::D) where {D <: GradDensity} = d.gradlogdistribution
#     gradlogdensity(d::D, x) where {D <: GradDensity} = d.gradlogdensity(x)
#     (D::GradDensity)(x) = density(D)(x)
#     Distributions.logpdf(d::GradDensity, x) = (log ∘ density(d))(x)
#     Distributions.gradlogpdf(d::GradDensity, x) = d.gradlogdensity(x)

#     function LogDensityProblems.capabilities(::Type{<:GradDensity})
#         LogDensityProblems.LogDensityOrder{1}()
#     end
#     LogDensityProblems.dimension(d::GradDensity) = d.dimension
#     LogDensityProblems.logdensity(d::GradDensity, x) = (log ∘ density(d))(x)
#     function logdensity_and_gradient(D::GradDensity, x)
#         (LogDensityProblems.logdensity(D, x), gradlogdensity(D, x))
#     end
# end

begin # * PotentialDensity
end

include("Distributions.jl")
include("../ext/InterpolationsExt.jl")
end # module
