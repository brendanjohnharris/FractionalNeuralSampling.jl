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
    α = 1.2
    β = 1.0
    γ = 0.0
    η = 0.2
    dt = 0.1
    𝜋 = MixtureModel([Normal(-2, 0.5), Normal(2, 0.5)]) |> Density
    domain = -10 .. 10
    boundaries = PeriodicBox(-5 .. 5)
    u0 = [0.0, 0.0]
    tspan = 5000.00

    S = bFNS(; α, β, γ, η, u0, 𝜋, dt, tspan, boundaries, domain)
    sol = solve(S, EM(); dt) |> Timeseries |> eachcol |> first
    sol = rectify(sol, dims = 𝑡; tol = 1)
    hist(sol; normalization = :pdf, bins = -3:0.1:3)
    lines!(-3:0.1:3, 𝜋.(-3:0.1:3), color = :red, linewidth = 2)
    current_figure()
end

begin # * Accuracy
end
