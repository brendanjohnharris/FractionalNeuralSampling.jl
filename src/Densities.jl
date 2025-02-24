module Densities
using Distributions
using DistributionsAD
using LogDensityProblems
using TransformVariables
using DifferentiationInterface
using TransformedLogDensities
import LogDensityProblems: logdensity, logdensity_and_gradient, dimension
import Distributions: gradlogpdf
import FractionalNeuralSampling.AD_BACKEND

export AbstractDensity, Density, distribution, potential, logdensity, gradlogdensity

abstract type AbstractDensity{D, doAd} end
const AbstractDistributionDensity{D, doAd} = AbstractDensity{D,
                                                             doAd} where {D <: Distribution,
                                                                          doAd}
const DistributionDensity{D} = AbstractDensity{D, false} where {D <: Distribution}
const AdDistributionDensity{D} = AbstractDensity{D, true} where {D <: Distribution}
const AdDensity{D} = AbstractDensity{D, true} where {D}
potential(D::AbstractDensity, x) = -logdensity(D, x)
potential(D::AbstractDensity) = Base.Fix1(potential, D)
logdensity(D::AbstractDensity) = Base.Fix1(logdensity, D)
gradlogdensity(D::AbstractDensity) = Base.Fix1(gradlogdensity, D)
function LogDensityProblems.dimension(D::AbstractDensity)
    LogDensityProblems.dimension(distribution(D))
end
export dimension

struct Density{D, doAd} <: AbstractDensity{D, doAd}
    distribution::D
    doAd::Bool
end
distribution(D::Density) = D.distribution

Density{doAd}(distribution::D) where {D, doAd} = Density{D, doAd}(distribution, doAd)
function Density(d::D) where {D <: UnivariateDistribution}
    Density{!hasmethod(Distributions.gradlogpdf, (D, Real))}(d)
end
function Density(d::D) where {D <: MultivariateDistribution}
    Density{!hasmethod(Distributions.gradlogpdf, (D, Vector{Real}))}(d)
end
function Density(d::D) where {D <: MixtureModel}
    Density{!hasmethod(Distributions.gradlogpdf, (D, Vector{Real}))}(d)
end

function _gradlogdensity(D::AdDensity, x::Real)
    gradient(x -> logdensity(D, only(x)), AD_BACKEND, [x]) |> only
end
function gradlogdensity!(grad::AbstractVector{T}, D::AdDensity,
                         x::AbstractVector{T},
                         extras = prepare_gradient(logdensity(D), AD_BACKEND, x)) where {T}
    gradient!(logdensity(D), grad, extras, AD_BACKEND, x)
end
function _gradlogdensity(D::AdDensity, x::AbstractVector{<:Real})
    grad = similar(x)
    extras = prepare_gradient(logdensity(D), AD_BACKEND, x)
    gradlogdensity!(grad, D, x, extras)
    return grad
end
function _gradlogdensity(D::AdDensity,
                         x::AbstractVector{<:AbstractVector{T}}) where {T}
    grad = similar(x)
    extras = prepare_gradient(logdensity(D), AD_BACKEND, first(x))
    gradlogdensity!(grad, D, x, extras)
    return grad
end

# * Distributions based on densities
#! format: off
const UnivariateDistributionDensity{D, doAd} = AbstractDistributionDensity{D,
                                                                           doAd} where {D <: Distribution{Univariate}, doAd}
const MultivariateDistributionDensity{D, doAd} = AbstractDistributionDensity{D,
                                                                             doAd} where {D <: Distribution{Multivariate}, doAd}
# ! format: on
_gradlogdensity(D::DistributionDensity, x) = Distributions.gradlogpdf(D, x)
function gradlogdensity(D::UnivariateDistributionDensity, x::T) where {T<:Real}
    _gradlogdensity(D, x)::T
end
function gradlogdensity(D::UnivariateDistributionDensity, x::AbstractVector{<:Number})
    _gradlogdensity.([D], x)
end
function gradlogdensity(D::DistributionDensity, x::AbstractVector{<:AbstractVector{<:Number}})
    _gradlogdensity.([D], x)
end
function gradlogdensity(D::AbstractDistributionDensity, x)
    _gradlogdensity(D, x)
end

(D::AbstractDistributionDensity)(x) = pdf(distribution(D), x)
(D::UnivariateDistributionDensity)(x::AbstractVector) = D(only(x))
function Distributions.logpdf(D::AbstractDistributionDensity, x)
    Distributions.logpdf(distribution(D), x)
end
function Distributions.gradlogpdf(D::AbstractDistributionDensity, x)
    Distributions.gradlogpdf(distribution(D), x)
end

function LogDensityProblems.capabilities(::Type{<:AbstractDistributionDensity})
    LogDensityProblems.LogDensityOrder{1}()
end
LogDensityProblems.dimension(D::AbstractDistributionDensity) = length(distribution(D))
LogDensityProblems.logdensity(D::AbstractDistributionDensity, x) = logpdf(D, x)
function logdensity_and_gradient(D::AbstractDistributionDensity, x)
    (LogDensityProblems.logdensity(D, x), gradlogdensity(D, x))
end

include("InterpolationsExt.jl")
end # module
