module ComplexityMeasuresExt
using Statistics
using LinearAlgebra
using ComplexityMeasures
using TimeseriesBase # For windowing
using FractionalNeuralSampling
import FractionalNeuralSampling: samplingpower, samplingaccuracy, _samplingaccuracy

"""
The variance of differences scaled by the time step.
Also known as the quadratic variation divided by the number of samples
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
    return (p_var / T)^(1 / p)
end

function _samplingaccuracy(x, 𝜋::AbstractDensity, τs::AbstractVector=2:100; p=10)
    if minimum(τs) < 2
        error("Minimum τ (samples) must be at least 2")
    end

    P = map(logdensity(𝜋), x)

    S = Base.Fix1(information, Kraskov(Shannon(; base=ℯ); k=3))

    map(τs) do τ

        # * KL divergence can be estimated from cross entropy and sample entropy of
        # * trajectory segments

        # * Calculate the cross-entropy part, which is independent of any windowing
        CE = -map(mean, window(P, τ, p))

        # ** Estimate the differential entropy of each window
        DE = map(S, window(x, τ, p))

        # * Approximate the final KL divergence
        ΔI = CE - DE
    end
end

function samplingaccuracy(x, 𝜋::AbstractDensity, τs::AbstractVector=2:100; kwargs...)
    ΔI = _samplingaccuracy(x, 𝜋, τs; kwargs...)
    return map(x -> exp.(-x), ΔI)
end

end
