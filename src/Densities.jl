module Densities
using Distributions
using LogDensityProblems
using TransformVariables
using DifferentiationInterface
using TransformedLogDensities
import LogDensityProblems: logdensity, logdensity_and_gradient, dimension
import Distributions: gradlogpdf
import FractionalNeuralSampling.AD_BACKEND

export AbstractDensity, Density, GradDensity, distribution, potential, logdensity,
       gradlogdensity

abstract type AbstractDensity{D, doA, N <: Int} end
const AdDensity{D} = AbstractDensity{D, true} where {D}
potential(D::AbstractDensity, x) = -logdensity(D, x)
potential(D::AbstractDensity) = Base.Fix1(potential, D)
logdensity(D::AbstractDensity) = Base.Fix1(logdensity, D)
gradlogdensity(D::AbstractDensity) = Base.Fix1(gradlogdensity, D)
function LogDensityProblems.dimension(d::AbstractDensity{D, doAd, N}) where {D, doAd, N}
    return N
end
export dimension

struct Density{D, doAd, N} <: AbstractDensity{D, doAd, N}
    distribution::D
end
distribution(D::Density) = D.distribution

function Density{doAd}(distribution::D; dimension) where {D, doAd}
    Density{D, doAd, dimension}(distribution)
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
function LogDensityProblems.capabilities(::Type{<:Density})
    LogDensityProblems.LogDensityOrder{1}()
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

# include("Distributions.jl")
# include("../ext/InterpolationsExt.jl")
end # module
