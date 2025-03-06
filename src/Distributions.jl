using Distributions
using DistributionsAD

const AbstractDistributionDensity{D, N, doAd} = AbstractDensity{D, N,
                                                                doAd} where {
                                                                             D <:
                                                                             Distribution,
                                                                             doAd}
const DistributionDensity{D} = AbstractDensity{D, N, false} where {N, D <: Distribution}
const AdDistributionDensity{D} = AbstractDensity{D, N, true} where {N, D <: Distribution}
function Density(d::D) where {D <: UnivariateDistribution}
    doAd = !hasmethod(Distributions.gradlogpdf, (D, Real))
    N = length(distribution(D))
    Density{N, doAd}(d)
end
function Density(d::D) where {D <: MultivariateDistribution}
    doAd = !hasmethod(Distributions.gradlogpdf, (D, Vector{Real}))
    N = length(distribution(D))
    Density{N, doAd}(d)
end
function Density(d::D) where {D <: MixtureModel}
    doAd = !hasmethod(Distributions.gradlogpdf, (D, Vector{Real}))
    N = length(distribution(D))
    Density{N, doAd}(d)
end

#! format: off
const UnivariateDistributionDensity{D, N, doAd} = AbstractDistributionDensity{D,
                                                                           doAd} where {N, D <: Distribution{Univariate}, doAd}
const MultivariateDistributionDensity{D, N, doAd} = AbstractDistributionDensity{D,
                                                                             doAd} where {N, D <: Distribution{Multivariate}, doAd}
# ! format: on
_gradlogdensity(D::DistributionDensity, x) = Distributions.gradlogpdf(D, x)
function Distributions.logpdf(D::AbstractDistributionDensity, x)
    Distributions.logpdf(distribution(D), x)
end
function Distributions.gradlogpdf(D::AbstractDistributionDensity, x)
    Distributions.gradlogpdf(distribution(D), x)
end

function LogDensityProblems.capabilities(::Type{<:AbstractDistributionDensity})
    LogDensityProblems.LogDensityOrder{1}()
end
LogDensityProblems.logdensity(D::AbstractDistributionDensity) = Base.Fix1(logpdf, D)
