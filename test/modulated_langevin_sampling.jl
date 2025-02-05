begin
    using StatsBase
    using CairoMakie
    using Foresight
    using DifferentialEquations
    using FractionalNeuralSampling
    import FractionalNeuralSampling as FNS
    using Random
    using Statistics
    using StaticArraysCore
    using Test
    using Distributions
    using DiffEqNoiseProcess
    using StableDistributions
    using BenchmarkTools
    using Profile
    using LinearAlgebra
    using ForwardDiff
    using LogDensityProblems
    using DifferentiationInterface
    using BenchmarkTools
    using InteractiveUtils
    using Distances
    using Distributed
    using SpecialFunctions
    using StatsBase
    using TimeseriesTools
    import FractionalNeuralSampling: Density
    set_theme!(foresight(:physics))
end

begin # * Construct a multimodal, 2D distribution
    D = FNS.Density(MixtureModel(MvNormal, [([-2, -2], I(2) / 2), ([2, 2], I(2) / 2)]))
end
begin # * Construct a modulated langevin sampler
    u0 = [[0.0, 0.0] [0.00, 0.0]]
    tspan = (0.0, 1000.0)
    dt = 0.001

    S = FNS.LangevinSampler(; u0, tspan, β = 5.0, γ = 5.0, 𝜋 = D)
    X = solve(S, EM(); dt, saveat = 0.001)
end
begin
    f = TwoPanel()
    ax = Axis(f[1, 1])
    xs = -5:0.1:5
    XY = collect.(Iterators.product(xs, xs))
    heatmap!(xs, xs, D.(XY))
    x = X[:, 1, :]
    lines!(Point2f.(eachcol(x))[1:50000], linewidth = 2, color = (:black, 0.5))

    ax = Axis(f[1, 2])
    hexbin!(eachrow(x)..., bins = 100)
    f
end

# * Need to switch to more general 1st/2nd order formulation with gradient on gradient
# * prepare_gradients for ND input in Sampler simulation
