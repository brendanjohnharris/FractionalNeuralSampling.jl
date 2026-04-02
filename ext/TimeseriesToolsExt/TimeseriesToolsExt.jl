module TimeseriesToolsExt
using TimeseriesTools
using FractionalNeuralSampling
using Distributions
import FractionalNeuralSampling: samplingpower, samplingaccuracy, samplingefficiency,
    _samplingaccuracy

function wasserstein(samples, quantiles, domain; p::Int=1)
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
The variance of differences scaled by the time step.
Also known as rate of quadratic variation.
For p != 2, this uses other norms of the increments (generalized variation).
"""
function samplingpower(x, dt; p=2)
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

function _samplingaccuracy(x, 𝜋::AbstractDensity; domain=nothing)
    τ = length(x)
    if τ < 2
        error("Minimum τ (samples) must be at least 2")
    end
    # * Calculate wasserstein distance
    quantiles = quantile(distribution(𝜋), (0.5:τ) ./ τ)

    wasserstein(x, quantiles, domain)
end

function _samplingaccuracy(x, 𝜋::AbstractDensity, τs::AbstractVector; p=0, # No overlap by default
    domain=nothing)
    if minimum(τs) < 2
        error("Minimum τ (samples) must be at least 2")
    end

    map(τs) do τ
        # * Calculate wasserstein distance
        samples = buffer(x, τ, p)
        quantiles = quantile(distribution(𝜋), (0.5:τ) ./ τ)

        ws = map(samples) do s
            wasserstein(s, quantiles, domain)
        end
    end
end

# """
# K-S divergence version using CDF (faster for distributions without closed-form quantiles)
# """
# function _samplingaccuracy(x, 𝜋::AbstractDensity; domain=nothing)
#     τ = length(x)
#     if τ < 2
#         error("Minimum τ (samples) must be at least 2")
#     end

#     # Sort samples for empirical CDF
#     sorted_x = sort(x)

#     # Filter by domain if specified
#     if !isnothing(domain)
#         sorted_x = filter(∈(domain), sorted_x)
#         τ = length(sorted_x)
#     end

#     # Evaluate theoretical CDF at sample points (fast!)
#     dist = distribution(𝜋)
#     theoretical_cdf = cdf.(dist, sorted_x)

#     # Empirical CDF values
#     empirical_cdf = (1:τ) ./ τ

#     # K-S statistic: maximum absolute difference
#     return maximum(abs.(empirical_cdf .- theoretical_cdf))
# end

# function _samplingaccuracy(x, 𝜋::AbstractDensity, τs::AbstractVector; p=0, # No overlap by default
#     domain=nothing)
#     if minimum(τs) < 2
#         error("Minimum τ (samples) must be at least 2")
#     end

#     map(τs) do τ
#         # Get buffered samples
#         samples = buffer(x, τ, p)
#         dist = distribution(𝜋)

#         # Calculate K-S distance for each sample buffer
#         ks = map(samples) do s
#             # Sort samples
#             sorted_s = sort(s)

#             # Filter by domain if specified
#             if !isnothing(domain)
#                 sorted_s = filter(∈(domain), sorted_s)
#                 n = length(sorted_s)
#             else
#                 n = τ
#             end

#             # Evaluate CDF at sample points
#             theoretical_cdf = cdf.(dist, sorted_s)
#             empirical_cdf = (1:n) ./ n

#             # K-S statistic
#             maximum(abs.(empirical_cdf .- theoretical_cdf))
#         end
#     end
# end

function samplingaccuracy(x, 𝜋::AbstractDensity, args...; kwargs...)
    ws = _samplingaccuracy(x, 𝜋, args...; kwargs...)
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
