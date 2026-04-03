using FractionalNeuralSampling
using Random
using StochasticDiffEq
using Distributions
using TimeseriesTools
using LinearAlgebra
using MoreMaps
using TimeseriesMakie
using CairoMakie
using Foresight
Foresight.set_theme!(Foresight.foresight(:physics))

import FractionalNeuralSampling: Density

begin
    Random.seed!(11)
    α = 1.4
    β = 1.0
    γ = 0.0
    η = 0.2
    dt = 0.1
    𝜋 = MixtureModel([Normal(-2, 0.5), Normal(2, 0.5)]) |> Density
    domain = -10 .. 10
    boundaries = PeriodicBox(-5 .. 5)
    u0 = [0.0, 0.0]
    tspan = 10000.00

    S = bFNS(; α, β, γ, η, u0, 𝜋, dt, tspan, boundaries, domain)
    sol = solve(S, EM(); dt) |> Timeseries |> eachcol |> first
    sol = rectify(sol, dims = 𝑡; tol = 1)
end

begin # * Plot
    f = FourPanel()

    ax = Axis(f[1, 1:2], xlabel = "t", ylabel = "x(t)")
    lines!(ax, sol[1:5000]) # Plot trajectory

    # Plot histogram
    ax = Axis(f[2, 1], xlabel = "x", ylabel = "Density")
    hist!(ax, sol; normalization = :pdf, bins = -3:0.1:3)
    lines!(ax, -3:0.1:3, 𝜋.(-3:0.1:3), color = :red, linewidth = 2)

    # Plot sampling accuracy
    ax = Axis(f[2, 2], xlabel = "t", ylabel = "Accuracy")
    ws = samplingaccuracy(sol, 𝜋, 1000:1000:10000; p = 100, domain = domain)
    lines!(ax, mean.(ws))

    f |> display
end
