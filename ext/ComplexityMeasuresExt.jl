module ComplexityMeasuresExt
using Statistics
using ComplexityMeasures
using TimeseriesBase # For windowing
using FractionalNeuralSampling
import FractionalNeuralSampling: samplingpower, samplingaccuracy

"""
This is just the variance of difference, normalized by the timestep
"""
function samplingpower(x, dt)
    dx² = map(norm, diff(x))  # Assumes input is stationary
    return var(dx²) / (dt * 2)
end

function samplingaccuracy(x, 𝜋::AbstractDensity, τs::AbstractVector = 2:100; p = 10)
    P = map(logdensity(𝜋), x)

    S = Base.Fix1(information, Kraskov(Shannon(; base = ℯ); k = 3))

    map(τs) do τ

        # * KL divergence can be estimated from cross entropy and sample entropy of
        # * trajectory segments

        # * Calculate the cross-entropy part, which is independent of any windowing
        CE = -map(mean, window(P, τ, p))

        # ** Estimate the differential entropy of each window
        DE = map(S, window(x, τ, p))

        # * Approximate the final KL divergence
        ΔI = CE - DE

        # * Finally, the accuracy is the exponential of the KL divergence.
        # * Bounded between 0 - and 1 since kl is 0 to Inf
        exp.(-ΔI)
    end
end

end
