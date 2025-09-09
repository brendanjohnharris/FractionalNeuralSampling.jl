using FractionalNeuralSampling
using StochasticDiffEq
using Distributions
using TimeseriesTools
using LinearAlgebra
using ComplexityMeasures
using MoreMaps
using TimeseriesMakie
using CairoMakie
using Foresight
Foresight.set_theme!(Foresight.foresight(:physics))

import FractionalNeuralSampling: Density

export samplingpower, samplingaccuracy

"""
This is just the variance of difference, normalized by the timestep
"""
function samplingpower(x, dt)
    dx² = map(norm, diff(x))  # Assumes input is stationary
    return var(dx²) / (dt * 2)
end

samplingpower(x::RegularTimeseries) = samplingpower(x, step(x))

function samplingaccuracy(x, 𝜋::AbstractDensity, τs::AbstractVector = 2:100; p = 1)
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

        # * Finally, the accuracy is the eponential of the KL divergence.
        # * Bounded between 0 - and 1 since kl is 0 to Inf
        exp.(-ΔI)
    end
end

function samplingaccuracy(x::RegularTimeseries, 𝜋::AbstractDensity, τs::AbstractVector;
                          dt = step(x), kwargs...)
    y = samplingaccuracy(parent(x), 𝜋, τs; kwargs...)
    return ToolsArray(y, 𝑡(τs .* step(x)))
end

begin
    etas = 0.2:0.2:2.0
    dt = 0.1
    τs = round.(Int, logrange(1, 1000, 10)) .÷ dt .|> Int
    𝜋 = MixtureModel([Normal(-2, 0.5), Normal(2, 0.5)]) |> Density
    u0 = [0.0]
    tspan = 5000.00

    xs = map(Dim{:η}(etas)) do η
        S = OLE(; η, u0, 𝜋, tspan)
        sol = solve(S, EM(); dt) |> Timeseries |> eachcol |> first
        return rectify(sol, dims = 𝑡; tol = 1)
    end

    accuracy = map(Chart(Threaded(), ProgressLogger()), xs) do x
        y = samplingaccuracy(x, 𝜋, τs)  #/ sqrt(samplingpower(x, dt))
        ToolsArray(y, 𝑡(τs))
    end |> stack
    accuracy = map(mean, accuracy)
end

begin
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Time lag", ylabel = "Accuracy", xscale = log10)
    p = traces!(ax, accuracy, linewidth = 2)
    Colorbar(f[1, 2], p; label = "η")
    display(f)
end
