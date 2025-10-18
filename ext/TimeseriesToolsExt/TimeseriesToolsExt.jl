module TimeseriesToolsExt
using TimeseriesTools
using FractionalNeuralSampling
using Distributions
import FractionalNeuralSampling: samplingpower, samplingaccuracy, samplingefficiency,
                                 _samplingaccuracy

function cdf_quantiles(n, dist::UnivariateDistribution)
    map(Base.Fix1(quantile, dist), (0.5:n) ./ n)
end
function wasserstein(samples, quantiles, domain; p::Int = 1)
    sorted_samples = sort(samples)
    idxs = map(∈(domain), sorted_samples)
    differences = abs.(sorted_samples .- quantiles)
    differences = differences[idxs] # Only in-domain samples
    if p == 1
        return mean(differences)
    else
        return mean(differences .^ p)^(1 / p)
    end
end

"""
The variance of differences scaled by the time step.
Also known as rate of quadratic variation.
For p != 2, this uses other norms of the increments (generalized variation).
"""
function samplingpower(x, dt; p = 2)
    # # # Compute squared increments
    # # dx² = map(Base.Fix1(sum, abs2), diff(x))

    # # # Return energy input rate
    # # return mean(dx²) / dt
    # return var(diff(x)) / dt

    Δx = diff(x)
    p_var = sum(abs.(Δx) .^ p)
    T = length(x) * dt
    return (p_var)^(1 / p) / T
end

function _samplingaccuracy(x, 𝜋::AbstractDensity, τs::AbstractVector = 2:100; p = 0, # No overlap by default
                           domain = nothing)
    if minimum(τs) < 2
        error("Minimum τ (samples) must be at least 2")
    end

    map(τs) do τ
        # * Calculate wasserstein distance
        samples = buffer(x, τ, p)
        quantiles = cdf_quantiles(τ, distribution(𝜋))
        ws = map(samples) do s
            wasserstein(s, quantiles, domain)
        end
    end
end

function samplingaccuracy(x, 𝜋::AbstractDensity, τs::AbstractVector = 2:100; kwargs...)
    ws = _samplingaccuracy(x, 𝜋, τs; kwargs...)
    return ws
end

samplingpower(x::RegularTimeseries) = samplingpower(x, step(x))

"""
Taus in unit steps
"""
function samplingaccuracy(x::RegularTimeseries, 𝜋::AbstractDensity, τs::AbstractVector;
                          kwargs...)
    y = samplingaccuracy(parent(x), 𝜋, τs; kwargs...)
    return ToolsArray(y, 𝑡(τs .* samplingperiod(x)))
end

# """
# Taus in time units
# """
# function samplingefficiency(x::RegularTimeseries, 𝜋::AbstractDensity,
#                             τs::AbstractVector = samplingperiod(x) .* (2:100);
#                             downsample = 10,
#                             kwargs...)
#     dt = samplingperiod(x)
#     _τs = round.(Int, τs ./ dt)
#     ΔI = _samplingaccuracy(x[1:downsample:end], 𝜋, _τs; kwargs...)
#     vd = samplingpower(x, dt)
#     se = map(x -> exp.(-x .* vd), ΔI)
#     return ToolsArray(se, 𝑡(τs))
# end

end
