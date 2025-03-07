module DistancesExt
using Distances
using StatsBase
using FractionalNeuralSampling

function (K::Distances.KLDivergence)(D::Density, x::AbstractVector)
    h = fit(Histogram, x) |> StatsBase.normalize
    xs = h.edges |> first
    xs = (xs[1:(end - 1)] .+ xs[2:end]) ./ 2
    idxs = h.weights .> 0
    xs = xs[idxs]
    logp̂ = h.weights[idxs] .|> log
    logp = logdensity(D, xs)
    p = map(D, xs)
    sum(p .* (logp .- logp̂))
end

end
