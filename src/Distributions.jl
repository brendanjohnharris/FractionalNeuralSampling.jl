using Distributions
using DistributionsAD
using Random
import LogDensityProblems: capabilities, LogDensityOrder
import Distributions: Distribution, MultivariateDistribution, UnivariateDistribution,
                      MixtureModel
import Distributions: gradlogpdf, pdf, logpdf

export DistributionDensity, distribution

begin # * Distribution densities; supply a Distributions.Distribution, get a density
    struct DistributionDensity{D, N, doAd} <: AbstractDensity{D, N, doAd}
        distribution::D
    end

    capabilities(::Type{<:DistributionDensity}) = LogDensityProblems.LogDensityOrder{1}()
    distribution(D::DistributionDensity) = D.distribution
    density(D::DistributionDensity) = Base.Fix1(pdf, distribution(D))
    logdensity(D::DistributionDensity) = Base.Fix1(logpdf, distribution(D))
    gradlogpdf(D::DistributionDensity) = Base.Fix1(gradlogpdf, distribution(D))
    # function _gradlogdensity(d::DistributionDensity{D, N, false},
    #                          x::AbstractVector{T}) where {D, N, T <: Real}
    #     gradlogpdf(D)(x)
    # end
    # function _gradlogdensity(d::DistributionDensity{D, N, false},
    #                          x::AbstractVector{<:AbstractVector{T}}) where {D, N, T <: Real}
    #     map(Base.Fix1(gradlogpdf, D), x)
    # end
    function _gradlogdensity(d::DistributionDensity{D, N, false}, x) where {D, N}
        gradlogpdf(d)(x)
    end
end

const UnivariateDistributionDensity = typeintersect(DistributionDensity,
                                                    AbstractUnivariateDensity)
univariate_fix(f, x) = f(x)
univariate_fix(f, x::AbstractVector{T}) where {T <: Real} = f(only(x))
function univariate_fix(f, x::AbstractVector{<:AbstractVector{T}}) where {T}
    map(univariate_fix(f), x)
end
univariate_fix(f) = Base.Fix1(univariate_fix, f)
function density(D::UnivariateDistributionDensity)
    univariate_fix(Base.Fix1(pdf, distribution(D)))
end
function logdensity(D::UnivariateDistributionDensity)
    univariate_fix(Base.Fix1(logpdf, distribution(D)))
end
function gradlogpdf(D::UnivariateDistributionDensity)
    univariate_fix(Base.Fix1(gradlogpdf, distribution(D)))
end

# * Constructors
function Density{doAd}(d::D) where {D <: Distribution, doAd}
    N = length(d)
    DistributionDensity{D, N, doAd}(d)
end
Density(D::Distribution, args...) = DistributionDensity(D, args...)
function DistributionDensity(distribution::D, doAd::Bool) where {D <: Distribution}
    DistributionDensity{D, length(distribution), doAd}(distribution)
end
function DistributionDensity(d::D) where {D <: UnivariateDistribution}
    doAd = !hasmethod(gradlogpdf, (D, Real))
    N = length(d)
    DistributionDensity{D, N, doAd}(d)
end
function DistributionDensity(d::D) where {D <: MultivariateDistribution}
    doAd = !hasmethod(gradlogpdf, (D, Vector{Real}))
    N = length(d)
    DistributionDensity{D, N, doAd}(d)
end
function DistributionDensity(d::D) where {D <: MixtureModel}
    doAd = !hasmethod(gradlogpdf, (D, Vector{Real}))
    N = length(d)
    DistributionDensity{D, N, doAd}(d)
end
Random.rand(rng::AbstractRNG, d::DistributionDensity) = rand(rng, distribution(d))
