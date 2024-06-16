using CairoMakie
using Foresight
using DifferentialEquations
using FractionalNeuralSampling
import FractionalNeuralSampling as FNS
using Random
using Statistics
using StaticArraysCore
using Test
using StableDistributions
set_theme!(foresight(:physics))

@testset "LEVYPROCESS" begin
    import FractionalNeuralSampling.NoiseProcesses.LEVY_NOISE_DIST
    DIST = LEVY_NOISE_DIST{false}(2.0, 0.0, 1 / sqrt(2), 0.0)
    Random.seed!(42)
    rng = Random.default_rng()
    a = DIST(rng)
    Random.seed!(42)
    b = rand(rng, DIST.stable)
    @test a == b

    @test Base.return_types(DIST, (AbstractRNG,)) == [Float64]
    @test Base.return_types(DIST, (AbstractRNG, Matrix)) == [Matrix{Float64}]
    @test Base.return_types(DIST, (AbstractRNG, Type{Float64})) == [Float64]
    @test DIST(rng, randn(10, 10)) isa Matrix
    @test DIST(rng, Float64) isa Float64
    @test_throws MethodError DIST(rng, Float32)  # Method error on type mismatch
    x = StaticArraysCore.SMatrix{3, 3}(zeros(3, 3))
    Random.seed!(42)
    y = DIST(x, nothing, 0.01, nothing, nothing, nothing, rng)
    @test typeof(y) == typeof(x)

    DIST = LEVY_NOISE_DIST{true}(2.0, 0.0, 1 / sqrt(2), 0.0)
    x = zeros(10)
    DIST(rng, x)
    @test all(x .!= 0)
    @test length(unique(x)) == length(x)
    x = zeros(10, 10)
    Random.seed!(42)
    DIST(rng, x)
    @test all(x .!= 0)
    @test length(unique(x)) == length(x)

    z = zeros(3, 3)
    Random.seed!(42)
    DIST(z, nothing, 0.01, nothing, nothing, nothing, rng)
    @test all(z .== y)
end

@testset "FractionalNeuralSampling.jl" begin
    include("fractional_sampling.jl")
end

@testset "LevyProcess" begin
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    W = LevyProcess(2.0; rng)
    dt = 0.1
    W.dt = dt
    u = nothing
    p = nothing # for state-dependent distributions
    calculate_step!(W, dt, u, p)
    for i in 1:10
        accept_step!(W, dt, u, p)
    end
    Random.seed!(rng, 42)
    prob = NoiseProblem(LevyProcess(2.0; rng, reseed = false), (0.0, 1.0))
    sol = solve(prob; dt = 0.1)

    function f3(u, p, t, W)
        2u * sin(W)
    end
    Random.seed!(rng, 42)
    u0 = 1.00
    tspan = (0.0, 5.0)
    prob = RODEProblem(f3, u0, tspan; noise = LevyProcess(2.0))
    @time sol = solve(prob, RandomEM(), dt = 1 / 100)
    plot(sol)

    function f4(du, u, p, t, W)
        du[1] = 2u[1] * sin(W[1] - W[2])
        du[2] = -2u[2] * cos(W[1] + W[2])
    end
    u0 = [1.00; 1.00]
    tspan = (0.0, 5.0)
    prob = RODEProblem(f4, u0, tspan; noise = LevyProcess(2.0))
    @test_throws "BoundsError" solve(prob, RandomEM(), dt = 1 / 100)
    @test_throws "DomainError" NoiseProblem(LevyProcess(-1.0), (0.0, 1.0))

    function f3!(u0, u, p, t, W)
        u0[1] = 2u[1] * sin(W[1])
    end
    u0 = [1.00]
    tspan = (0.0, 5.0)
    L = LevyProcess!(2.0)
    prob = RODEProblem{true}(f3!, u0, tspan; noise = L)
    @time solve(prob, RandomEM(); dt = 1 / 100)
end

@testset "Brownian Noise" begin
    prob = NoiseProblem(LevyProcess(2.0), (0.0, 1.0))
    dt = 0.0001
    ensemble = EnsembleProblem(prob)
    sol = solve(prob, RandomEM(); dt)

    lines(sol.t, sol.u; linewidth = 2)
    @test std(diff(sol.u))≈sqrt(dt) rtol=1e-2
end

@testset "Levy Noise" begin
    L = LevyProcess(1.5)
    prob = NoiseProblem(L, (0.0, 1.0))
    dt = 1e-6
    sol = solve(prob, RandomEM(); dt)

    f = fit(Stable, diff(sol.u) ./ (dt^(1 / 1.5)))
    @test L.dist.stable.α≈f.α atol=1e-2
end

@testset "Ensemble" begin
    Random.seed!(42)
    L = LevyProcess(1.5)
    dt = 1e-3
    prob = NoiseProblem(L, (0.0, 1.0))
    ensemble = EnsembleProblem(prob)
    sol = @test_nowarn solve(ensemble, RandomEM(), EnsembleSerial(); trajectories = 5, dt)
    @test_nowarn solve(ensemble, RandomEM(), EnsembleDistributed(); trajectories = 5, dt)
    @test_nowarn solve(ensemble, RandomEM(), EnsembleThreads(); trajectories = 5, dt)
    f = Figure()
    ax = Axis(f[1, 1])
    [lines!(ax, s.t, s.u) for s in sol]
    display(f)
end
