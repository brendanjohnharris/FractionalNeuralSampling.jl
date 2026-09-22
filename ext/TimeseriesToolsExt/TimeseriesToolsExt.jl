module TimeseriesToolsExt
using TimeseriesTools
using FractionalNeuralSampling
using Distributions
import FractionalNeuralSampling: samplingpower, samplingaccuracy, _samplingaccuracy

function wasserstein(samples, quantiles, domain; p::Int = 1)
    sorted_samples = sort(samples)
    differences = abs.(sorted_samples .- quantiles)
    if !isnothing(domain)
        idxs = map(∈(domain), sorted_samples)
        differences = differences[idxs] # Only in-domain samples
    end
    if p == 1
        return mean(differences)
    else
        return mean(differences .^ p)^(1 / p)
    end
end

"""
The p-variation of the increments per unit time; for p = 2 this is a rate of quadratic
variation.
"""
function samplingpower(x, dt; p = 2)
    Δx = diff(x)
    p_var = sum(abs.(Δx) .^ p)
    T = length(x) * dt
    return (p_var)^(1 / p) / T
end

function _samplingaccuracy(x, 𝜋::AbstractDensity; domain = nothing)
    τ = length(x)
    if τ < 2
        error("Minimum τ (samples) must be at least 2")
    end
    # * Calculate wasserstein distance
    quantiles = quantile(distribution(𝜋), (0.5:τ) ./ τ)

    return wasserstein(x, quantiles, domain)
end

function _samplingaccuracy(
        x, 𝜋::AbstractDensity, τs::AbstractVector; p = 0, # No overlap by default
        domain = nothing
    )
    if minimum(τs) < 2
        error("Minimum τ (samples) must be at least 2")
    end

    return map(τs) do τ
        # * Calculate wasserstein distance
        samples = buffer(x, τ, p)
        quantiles = quantile(distribution(𝜋), (0.5:τ) ./ τ)

        map(samples) do s
            wasserstein(s, quantiles, domain)
        end
    end
end

function samplingaccuracy(x, 𝜋::AbstractDensity, args...; kwargs...)
    return _samplingaccuracy(x, 𝜋, args...; kwargs...)
end

samplingpower(x::RegularTimeseries) = samplingpower(x, step(x))

"""
    samplingaccuracy(x::RegularTimeseries, 𝜋::AbstractDensity, τs::AbstractVector; kwargs...)

`τs` is given in samples, and the result is returned over time, with each window length
converted to a duration by the sampling period of `x`.
"""
function samplingaccuracy(
        x::RegularTimeseries, 𝜋::AbstractDensity, τs::AbstractVector;
        kwargs...
    )
    y = samplingaccuracy(parent(x), 𝜋, τs; kwargs...)
    return ToolsArray(y, 𝑡(τs .* samplingperiod(x)))
end


end # module
