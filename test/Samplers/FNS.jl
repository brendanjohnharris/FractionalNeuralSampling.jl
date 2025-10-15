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

begin
    alphas = 1.4:0.2:2
    γ = 0.5
    dt = 0.01
    τs = round.(Int, logrange(1, 1000, 10)) .÷ dt .|> Int
    𝜋 = MixtureModel([Normal(-2, 0.5), Normal(2, 0.5)]) |> Density
    domain = -5 .. 5
    boundaries = PeriodicBox(domain)
    u0 = [0.0, 0.0]
    α = 1.6
    β = 0.4
    tspan = 5000.00

    xs = map(Dim{:α}(alphas)) do α
        S = FNS(; γ, β, α, u0, 𝜋, tspan, boundaries)
        sol = solve(S, EM(); dt) |> Timeseries |> eachcol |> first
        return rectify(sol, dims = 𝑡; tol = 1)
    end

    accuracy = map(Chart(Threaded(), ProgressLogger()), xs) do x
        y = samplingaccuracy(x, 𝜋, τs; p = 1000)  #/ sqrt(samplingpower(x, dt))
        ToolsArray(y, 𝑡(τs))
    end |> stack
    accuracy = map(mean, accuracy)

    # _τs = τs * dt # For efficiency
    # efficiency = map(Chart(Threaded(), ProgressLogger()), xs) do x
    #     y = samplingefficiency(x, 𝜋, _τs; downsample = 5, p = 1000)
    #     ToolsArray(y, 𝑡(_τs))
    # end |> stack
    # efficiency = map(mean, efficiency)
end

begin
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Time lag", ylabel = "Accuracy", xscale = log10)
    p = traces!(ax, accuracy, linewidth = 2)
    hlines!(ax, [1.0]; color = :gray, linestyle = :dash)
    Colorbar(f[1, 2], p; label = "α")
    display(f)
end

begin # * Plot distribution for long time lag
    S = FNS(; γ, β, α = 1.5, u0, 𝜋, tspan, boundaries)
    sol = solve(S, EM(); dt) |> Timeseries |> eachcol |> first
    sol = rectify(sol, dims = 𝑡; tol = 1)
    hist(sol; normalization = :pdf, bins = -3:0.1:3)
    lines!(-3:0.1:3, 𝜋.(-3:0.1:3), color = :red, linewidth = 2)
    current_figure()
end
# begin
#     ts = 1:100
#     vd = map(Chart(Threaded(), ProgressLogger()), xs) do x
#         y = map(ts) do t
#             samplingpower(x[1:t:end])
#         end
#         ToolsArray(y, 𝑡(ts))
#     end |> stack

#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Time step", ylabel = "Sampling power", xscale = log10)
#     p = traces!(ax, vd, linewidth = 2)
#     Colorbar(f[1, 2], p; label = "α")
#     display(f)
# end

# begin
#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Time lag", ylabel = "Efficiency", xscale = log10)
#     p = traces!(ax, efficiency, linewidth = 2)
#     hlines!(ax, [1.0]; color = :gray, linestyle = :dash)
#     Colorbar(f[1, 2], p; label = "α")
#     display(f)
# end
