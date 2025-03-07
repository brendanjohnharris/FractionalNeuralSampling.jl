using Distributions
using DistributionsAD

const DistributionDensity{D, N} = Density{D, N, false} where {N, D <: Distribution}
const AdDistributionDensity{D, N} = Density{D, N, true} where {N, D <: Distribution}
const UnivariateDistributionDensity{D, N, doAd} = Density{D, N,
                                                          doAd} where {N,
                                                                       D <:
                                                                       Distribution{Univariate},
                                                                       doAd}
const MultivariateDistributionDensity{D, N, doAd} = Density{D, N,
                                                            doAd} where {N,
                                                                         D <:
                                                                         Distribution{Multivariate},
                                                                         doAd}

function Density(d::D) where {D <: UnivariateDistribution}
    doAd = !hasmethod(Distributions.gradlogpdf, (D, Real))
    N = length(d)
    Density{N, doAd}(d)
end
function Density(d::D) where {D <: MultivariateDistribution}
    doAd = !hasmethod(Distributions.gradlogpdf, (D, Vector{Real}))
    N = length(d)
    Density{N, doAd}(d)
end
function Density(d::D) where {D <: MixtureModel}
    doAd = !hasmethod(Distributions.gradlogpdf, (D, Vector{Real}))
    N = length(d)
    Density{N, doAd}(d)
end

_gradlogdensity(D::DistributionDensity, x) = Distributions.gradlogpdf(D, x)
function Distributions.logpdf(D::DistributionDensity, x)
    Distributions.logpdf(distribution(D), x)
end
function Distributions.gradlogpdf(D::DistributionDensity, x)
    Distributions.gradlogpdf(distribution(D), x)
end

function LogDensityProblems.capabilities(::Type{<:DistributionDensity})
    LogDensityProblems.LogDensityOrder{1}()
end
LogDensityProblems.logdensity(D::DistributionDensity) = Base.Fix1(logpdf, D)
