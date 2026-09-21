using FractionalNeuralSampling
using Distributions
using TimeseriesTools
using Random
using Test

begin # * Make Sampler
    dt = 0.001
    η = 0.1
    𝜋 = PotentialDensity{1}(_ -> 0.0) # Flat potential
    u0 = [0.0]
    tspan = 100.0
    S = OLE(; η, u0, 𝜋, tspan)
end

begin # * CaputoEM at β = 1 reduces exactly to EM
    Random.seed!(1234)
    alg = @inferred EM()
    sol = solve(S, alg; dt) |> Timeseries |> eachcol |> only

    Random.seed!(1234)
    sol2 = solve(S, CaputoEM(1.0, 1000); dt) |> Timeseries |> eachcol |> only
    @test sol == sol2
end

begin # * Stepping is type stable
    alg = @inferred EM()
    alg2 = @inferred CaputoEM(0.75f0, 1000)
    int = init(S, alg, dt = dt)
    int2 = init(S, alg2, dt = dt)
    @inferred StochasticDiffEqCore.perform_step!(int, int.cache)
    @inferred StochasticDiffEqCore.perform_step!(int2, int2.cache)
end

begin # * A coarser timestep gives a rougher path, not a different one
    Random.seed!(1234)
    noise = [[randn()] for n in 1:100000] |> cumsum # Must be the integral of the noise
    ts = range(S.tspan..., length = length(noise))
    W = DiffEqNoiseProcess.NoiseGrid(ts, noise)
    S2 = OLE(; η, u0, 𝜋, tspan, noise = W)

    dt2 = 0.01 # Below 0.01 the paths do diverge
    alg = @inferred CaputoEM(0.6, Int(100 ÷ dt2))
    sol2 = solve(S2, alg; dt = dt2) |> Timeseries |> eachcol |> only

    dt2 = 0.1
    alg = @inferred CaputoEM(0.6, Int(100 ÷ dt2))
    sol3 = solve(S2, alg; dt = dt2) |> Timeseries |> eachcol |> only

    ts = range(S2.tspan..., length = min(length(sol2), length(sol3)))
    @test cor(sol2[𝑡 = Near(ts)], sol3[𝑡 = Near(ts)]) > 0.95
end

begin # * 2D target, which takes the multivariate gradient path
    using LinearAlgebra
    Random.seed!(1234)
    dt = 0.01
    η = 0.1
    𝜋 = MixtureModel([MvNormal([-3.0, -3.0], I), MvNormal([3.0, 3.0], I)]) |>
        FractionalNeuralSampling.Density
    u0 = [0.0, 0.0]
    tspan = 10.0
    S = OLE(; η, u0, 𝜋, tspan)

    alg = @inferred CaputoEM(0.5, 1000)
    sol2 = @test_nowarn solve(S, alg; dt) |> Timeseries
    @test size(sol2, 2) == 2
end
