module Densities
using Distributions
using LogDensityProblems
using TransformVariables
using DifferentiationInterface
using TransformedLogDensities
import LogDensityProblems: logdensity, logdensity_and_gradient, dimension
import Distributions: gradlogpdf
import FractionalNeuralSampling.AD_BACKEND

export AbstractDensity, AbstractUnivariateDensity, Density, GradDensity, PotentialDensity

export distribution, potential, logdensity, gradlogdensity, gradpotential, dimension

abstract type AbstractDensity{D, N, doAd} end
const AbstractUnivariateDensity{D, doAd} = AbstractDensity{D, 1, doAd} where {D, doAD}

logdensity(D::AbstractDensity, x) = (log ∘ distribution(D))(x)
gradlogdensity(D::AbstractDensity) = Base.Fix1(gradlogdensity, D)

potential(D::AbstractDensity, x) = -logdensity(D, x)
potential(D::AbstractDensity) = Base.Fix1(potential, D)

gradpotential(D::AbstractDensity, x) = -gradlogdensity(D, x)
gradpotential(D::AbstractDensity) = (-) ∘ gradlogdensity(D)

(D::AbstractDensity)(x) = distribution(D)(x)
(D::AbstractUnivariateDensity)(x::AbstractVector) = D(only(x))

function LogDensityProblems.dimension(d::AbstractDensity{D, N, doAd}) where {D, N, doAd}
    return N
end
doautodiff(d::AbstractDensity{D, N, doAd}) where {D, N, doAd} = doAd
export dimension

# * Automatic autodiff
const AdDensity{D} = AbstractDensity{D, N, true} where {D, N}
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

function gradlogdensity(d::AbstractUnivariateDensity, x::T) where {T <: Real}
    _gradlogdensity(d, x)::T
end
function gradlogdensity(d::AbstractUnivariateDensity,
                        x::T) where {T <: AbstractVector{<:Real}}# For 1 element vectors
    convert(T, [_gradlogdensity(d, only(x))])
end
function gradlogdensity(d::AbstractDensity,
                        x::AbstractVector{<:AbstractVector{T}}) where {T <: Real}
    map(Base.Fix1(gradlogdensity, d), x)
end
function gradlogdensity(d::AbstractDensity, x)
    _gradlogdensity(d, x)
end
function logdensity_and_gradient(D::AbstractDensity, x)
    (logdensity(D, x), gradlogdensity(D, x))
end

begin # * See here for the Density interface: define these methods and traits. Custom differentiation functions can also be added; see Distributions.jl
    struct Density{D, N, doAd} <: AbstractDensity{D, N, doAd}
        distribution::D
    end
    Density{N}(distribution::D) where {D, N} = Density{D, N, true}(distribution)
    Density{N, doAd}(distribution::D) where {D, N, doAd} = Density{D, N, doAd}(distribution)

    function LogDensityProblems.capabilities(::Type{<:Density})
        LogDensityProblems.LogDensityOrder{1}()
    end
    distribution(D::Density) = D.distribution
    logdensity(D::Density) = log ∘ distribution(D)
end
begin # ! PotentialDensity (supply a potential, get a POTENTIALLY NON_NORMALIZED density)
    struct PotentialDensity{D, N, doAd} <: AbstractDensity{D, N, doAd}
        potential::D
    end
    PotentialDensity{N}(potential::D) where {D, N} = PotentialDensity{D, N, true}(potential)

    function LogDensityProblems.capabilities(::Type{<:PotentialDensity})
        LogDensityProblems.LogDensityOrder{1}()
    end

    potential(D::PotentialDensity) = D.potential
    logdensity(D::PotentialDensity) = (-) ∘ potential(D)
    gradpotential(D::PotentialDensity) = (.-) ∘ gradlogdensity(D)
    distribution(D::PotentialDensity) = exp ∘ (-) ∘ potential(D)
end
# ? Density interface: just need to define the following methods and traits. E.g. for
# ? Densities with supplied gradients:
# begin # * GradDensity
#     struct GradDensity{D, G} <: AbstractDensity{D, false} # You are supplying a gradient function, so don't autodiff
#         distribution::D # Should be f([x, y]) -> d
#         gradlogdistribution::G # Should be f([x, y]) -> [∂x, ∂y]
#         dimension::Int
#     end
#     GradDensity(d::D, g::G; dimension) where {D, G} = GradDensity{D, G}(d, g, dimension)

#     gradlogdensity(d::D) where {D <: GradDensity} = d.gradlogdistribution
#     gradlogdensity(d::D, x) where {D <: GradDensity} = d.gradlogdistribution(x)
#     (D::GradDensity)(x) = distribution(D)(x)
#     Distributions.logpdf(d::GradDensity, x) = (log ∘ distribution(d))(x)
#     Distributions.gradlogpdf(d::GradDensity, x) = d.gradlogdistribution(x)

#     function LogDensityProblems.capabilities(::Type{<:GradDensity})
#         LogDensityProblems.LogDensityOrder{1}()
#     end
#     LogDensityProblems.dimension(d::GradDensity) = d.dimension
#     LogDensityProblems.logdensity(d::GradDensity, x) = (log ∘ distribution(d))(x)
#     function logdensity_and_gradient(D::GradDensity, x)
#         (LogDensityProblems.logdensity(D, x), gradlogdensity(D, x))
#     end
# end

begin # * PotentialDensity
end

include("Distributions.jl")
include("../ext/InterpolationsExt.jl")
end # module
