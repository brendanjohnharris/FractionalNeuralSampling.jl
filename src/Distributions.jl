using Distributions
using DistributionsAD

const AbstractDistributionDensity{D, doAd} = AbstractDensity{D,
                                                             doAd} where {D <: Distribution,
                                                                          doAd}
const DistributionDensity{D} = AbstractDensity{D, false} where {D <: Distribution}
const AdDistributionDensity{D} = AbstractDensity{D, true} where {D <: Distribution}
function Density(d::D) where {D <: UnivariateDistribution}
    Density{!hasmethod(Distributions.gradlogpdf, (D, Real))}(d)
end
function Density(d::D) where {D <: MultivariateDistribution}
    Density{!hasmethod(Distributions.gradlogpdf, (D, Vector{Real}))}(d)
end
function Density(d::D) where {D <: MixtureModel}
    Density{!hasmethod(Distributions.gradlogpdf, (D, Vector{Real}))}(d)
end

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
