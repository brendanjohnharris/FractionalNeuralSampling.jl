module Densities
using Distributions
using LogDensityProblems
using DifferentiationInterface
import LogDensityProblems: logdensity, logdensity_and_gradient, dimension, capabilities
import Distributions: gradlogpdf
import FractionalNeuralSampling.AD_BACKEND

export AbstractDensity, AbstractUnivariateDensity, Density

export potential, logdensity, gradlogdensity, graddensity, gradpotential, dimension

"""
    AbstractDensity{D, N, doAd}

Supertype of the target densities. `D` is the wrapped object, `N` the dimension of the
state it accepts, and `doAd` records whether gradients come from automatic differentiation
or from an analytic method.

Subtypes are [`Density`](@ref) for a distribution or a probability density function,
[`DistributionDensity`](@ref) for a `Distributions.Distribution`, and
[`PotentialDensity`](@ref) for a target given by its potential. All of them are callable
and support [`logdensity`](@ref), [`gradlogdensity`](@ref), [`graddensity`](@ref),
[`potential`](@ref), [`gradpotential`](@ref) and [`dimension`](@ref).
"""
abstract type AbstractDensity{D, N, doAd} end
const AbstractUnivariateDensity{D, doAd} = AbstractDensity{D, 1, doAd}

"""
    logdensity(𝜋::AbstractDensity, x)

log𝜋(x). Up to an additive constant for a [`PotentialDensity`](@ref), which is not
normalised.
"""
logdensity(D::AbstractDensity, x) = logdensity(D)(x)
"""
    gradlogdensity(𝜋::AbstractDensity, x)
    gradlogdensity(𝜋::AbstractDensity)

∇log𝜋(x), the drift of an overdamped sampler. Called with the density alone, returns the
function of `x`.

Evaluated analytically where the wrapped distribution defines
`Distributions.gradlogpdf`, and by automatic differentiation otherwise.
"""
gradlogdensity(D::AbstractDensity) = Base.Fix1(gradlogdensity, D)
"""
    graddensity(𝜋::AbstractDensity, x)
    graddensity(𝜋::AbstractDensity)

∇𝜋(x), the gradient of the density itself rather than of its logarithm. Called with the
density alone, returns the function of `x`.
"""
graddensity(D::AbstractDensity) = Base.Fix1(graddensity, D)

"""
    potential(𝜋::AbstractDensity, x)
    potential(𝜋::AbstractDensity)

V(x) = -log𝜋(x). Called with the density alone, returns the function of `x`.
"""
potential(D::AbstractDensity, x) = -logdensity(D, x)
potential(D::AbstractDensity, x::Tuple) = potential(D, collect(x))
potential(D::AbstractDensity) = Base.Fix1(potential, D)

"""
    gradpotential(𝜋::AbstractDensity, x)
    gradpotential(𝜋::AbstractDensity)

∇V(x) = -∇log𝜋(x), the force a sampler feels. Called with the density alone, returns the
function of `x`.
"""
gradpotential(D::AbstractDensity, x) = -gradlogdensity(D, x)
gradpotential(D::AbstractDensity) = (-) ∘ gradlogdensity(D)

(D::AbstractDensity)(x) = density(D)(x)
(D::AbstractDensity)(x::Tuple) = D(collect(x))
(D::AbstractUnivariateDensity)(x::AbstractVector) = D(only(x))

"""
    dimension(𝜋::AbstractDensity)

The dimension of the state `𝜋` accepts. A second-order sampler needs
`length(u0) == 2 * dimension(𝜋)`, and a first-order one `length(u0) == dimension(𝜋)`.
"""
function LogDensityProblems.dimension(d::AbstractDensity{D, N, doAd}) where {
        D, N,
        doAd,
    }
    return N
end
doautodiff(d::AbstractDensity{D, N, doAd}) where {D, N, doAd} = doAd

# * Automatic autodiff
const AdDensity{D} = AbstractDensity{D, N, true} where {D, N}
function _gradlogdensity(D::AdDensity, x::Real)
    return derivative(logdensity(D), AD_BACKEND, x)
end
function _gradlogdensity(D::AdDensity, x::AbstractVector{<:Real})
    f = logdensity(D)
    extras = prepare_gradient(f, AD_BACKEND, x)
    return gradient(f, extras, AD_BACKEND, x)
end
function _gradlogdensity(
        D::AdDensity,
        x::AbstractVector{<:AbstractVector{T}}
    ) where {T}
    f = logdensity(D)
    extras = prepare_gradient(f, AD_BACKEND, first(x))
    grad = map(similar, x)
    foreach(grad, x) do _grad, _x
        gradient!(f, _grad, extras, AD_BACKEND, _x)
    end
    return grad
end

function gradlogdensity(d::AbstractUnivariateDensity, x::T) where {T <: Real}
    return _gradlogdensity(d, x)::T
end
function gradlogdensity(
        d::AbstractUnivariateDensity,
        x::AbstractVector{T}
    ) where {T <: Real} # For 1 element vectors
    return convert(Vector{T}, [_gradlogdensity(d, only(x))])
end
function gradlogdensity(d::AbstractDensity, x)
    return _gradlogdensity(d, x)
end
function logdensity_and_gradient(D::AbstractDensity, x)
    return (logdensity(D, x), gradlogdensity(D, x))
end

# * Gradient of density
function _graddensity(D::AdDensity, x::Real)
    return derivative(density(D), AD_BACKEND, x)
end
function _graddensity(D::AdDensity, x::AbstractVector{<:Real})
    f = density(D)
    extras = prepare_gradient(f, AD_BACKEND, x)
    return gradient(f, extras, AD_BACKEND, x)
end
function _graddensity(
        D::AdDensity,
        x::AbstractVector{<:AbstractVector{T}}
    ) where {T}
    f = density(D)
    extras = prepare_gradient(f, AD_BACKEND, first(x))
    grad = map(similar, x)
    foreach(grad, x) do _grad, _x
        gradient!(f, _grad, extras, AD_BACKEND, _x)
    end
    return grad
end

function graddensity(d::AbstractUnivariateDensity, x::T) where {T <: Real}
    return _graddensity(d, x)::T
end
function graddensity(
        d::AbstractUnivariateDensity,
        x::AbstractVector{T}
    ) where {T <: Real} # For 1 element vectors
    return convert(Vector{T}, [_graddensity(d, only(x))])
end
function graddensity(d::AbstractDensity, x)
    return _graddensity(d, x)
end

begin # * See here for the Density interface: define these methods and traits. Custom differentiation functions can also be added; see Densities/Distributions.jl
    """
        Density(d::Distribution)
        Density{N}(f)
        Density{N, doAd}(f)

    The target distribution 𝜋 that a sampler is to reproduce.

    Given a `Distributions.Distribution`, the dimension is taken from the distribution and
    a [`DistributionDensity`](@ref) is returned; gradients are analytic where the
    distribution defines `Distributions.gradlogpdf` and come from automatic differentiation
    otherwise. Given a function, `N` is the dimension of the state it accepts and cannot be
    inferred, so it is supplied as a type parameter.

    `doAd` overrides the choice of gradient: `Density{false}(d)` forces the analytic method
    and `Density{true}(d)` forces automatic differentiation. The backend defaults to
    `AutoForwardDiff()` and is changed with
    `FractionalNeuralSampling.set_ad_backend!`, which takes effect after a restart.

    ```julia
    Density(Normal(0.0, 1.0))                 # analytic gradient
    Density(MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)]))  # autodiff
    Density{1}(x -> exp(-only(x)^2 / 2))      # a pdf given as a function
    ```
    """
    struct Density{D, N, doAd} <: AbstractDensity{D, N, doAd}
        density::D
    end
    function Density{N, doAd}(density::D) where {D, N, doAd}
        return Density{D, N, doAd}(density)
    end
    function Density{N}(density::D, doAd::Bool) where {D, N}
        return Density{N, doAd}(density)
    end
    function Density{N}(density::D) where {D, N}
        return Density{N, true}(density) # Default to true autodiff
    end
    capabilities(::Type{<:Density}) = LogDensityProblems.LogDensityOrder{1}()
    density(D::Density) = D.density
    logdensity(D::Density) = log ∘ density(D)
end

include("Densities/Distributions.jl")
include("Densities/PotentialDensity.jl")

# * Extensions
function vignette end

end # module
