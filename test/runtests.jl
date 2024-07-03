begin
    using CUDA
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
    using TimeseriesTools
    import FractionalNeuralSampling: Density
    set_theme!(foresight(:physics))
end

@testset "Autodiff" begin
    D = Normal(0, 0.5)
    f = x -> logpdf(D, only(x))
    ForwardDiff.gradient(f, [0.1])
    backend = AutoForwardDiff()
    @test gradlogdensity(FNS.Density{true}(D), 0.1:0.1:3) == gradlogpdf.([D], 0.1:0.1:3)
    a = @benchmark gradient($f, $backend, 0.1)
    b = @benchmark ForwardDiff.gradient($f, [0.1])
    c = @benchmark gradlogpdf($D, 0.1)
    @test a.allocs < 15
    @test b.allocs < 10
    @test c.allocs == c.memory == 0
    a = @benchmark gradlogdensity(FNS.Density{false}($D), 0.1)
    @test a.allocs == c.memory == 0
    b = @benchmark gradlogdensity(FNS.Density{true}($D), 0.1)
    @test b.allocs == b.memory == 0
    cl = @code_lowered FNS.Densities._gradlogdensity(FNS.Density{true}(D), 0.1)
    @test contains(string(cl.code), "AD_BACKEND")
end
@testset "Univariate DistributionDensity" begin
    d = Normal(0.0, 0.5)
    D = @test_nowarn FNS.Density(d)
    @test D isa FNS.Densities.UnivariateDistributionDensity
    @test D.doAd == false
    @test D(0.0) == 2 / sqrt(2π)
    @test D([0.0]) == 2 / sqrt(2π)
    @test LogDensityProblems.dimension(D) == 1
    @test all(LogDensityProblems.logdensity.([D], -1:0.1:1) .≈ log.(D.(-1:0.1:1)) .≈
              logpdf(D, -1:0.1:1))
    @inferred LogDensityProblems.logdensity(D, 0.0)
    @inferred LogDensityProblems.logdensity(D, 0)
    lines(-2:0.1:2, D.(-2:0.1:2))
    lines(-2:0.01:2, FNS.Densities.potential(D).(-2:0.01:2))
    @inferred FNS.Densities.gradlogdensity(D, 0.01)
    @inferred FNS.Densities.gradlogdensity(D, 0.01:0.01:5)
    @test FNS.Densities.gradlogdensity(D, 0.1:0.1:5) == gradlogpdf.([D], 0.1:0.1:5)

    d = Uniform(-0.5, 0.5)
    D = @test_nowarn FNS.Density(d)
    @test D(0.0) == 1
    @test LogDensityProblems.dimension(D) == 1
    @test all(LogDensityProblems.logdensity.([D], -1:0.1:1) .≈ log.(D.(-1:0.1:1)))
    @inferred LogDensityProblems.logdensity(D, 0.0)
    @inferred LogDensityProblems.logdensity(D, -0.6)
    @inferred LogDensityProblems.logdensity_and_gradient(D, -0.6)
    @inferred LogDensityProblems.logdensity_and_gradient(D, [-0.6])
    @inferred LogDensityProblems.logdensity_and_gradient(FNS.Density{true}(d), 0.5)

    if isinteractive()
        @benchmark FNS.Densities.gradlogdensity($D).(-1:0.01:1)
        @benchmark LogDensityProblems.logdensity_and_gradient.([$D], -1:0.01:1)
    end
    lines(-2:0.1:2, D.(-2:0.1:2))
    lines(-2:0.01:2, FNS.Densities.potential(D).(-2:0.01:2))
    lines(-2:0.01:2, FNS.Densities.gradlogdensity(D).(-2:0.01:2))

    D = @test_nowarn FNS.Density(Normal(0.0f0, 0.5f0))
    @test LogDensityProblems.logdensity(D, 0.0f0) isa Float32
    @test FNS.Densities.gradlogdensity(D, 0.0f0) isa Float32
end

@testset "Multivariate DistributionDensity" begin
    N = 3
    μs = randn(N)
    x = randn(N, 100)
    Σ = x * x'
    d = MvNormal(μs, Σ)
    D = @test_nowarn FNS.Density(d)
    @test D isa FNS.Densities.MultivariateDistributionDensity
    @test D.doAd == false
    p = randn(N)
    @test logdensity(D)(p) == logpdf(d, p)
    ps = eachcol(randn(N, 100))
    @test logdensity(D)(ps) == logpdf.([d], ps)
    @test gradlogdensity(D)(p) == gradlogpdf(d, p)
    @test gradlogdensity(D)(ps) == gradlogpdf.([d], ps)

    # * Ad
    D = @test_nowarn FNS.Density{true}(MvNormal(μs, Σ))
    @test D.doAd == true
    @test LogDensityProblems.logdensity(D, p) isa Float64
    @test logdensity(D)(p) == logpdf(d, p)
    @test logdensity(D)(ps) == logpdf.([d], ps)
    @test gradlogdensity(D)(p) ≈ gradlogpdf(d, p)
end

@testset "Mixture DistributionDensity" begin
    Nd = 3
    N = 10
    μs = [randn(Nd) for _ in 1:N]
    Σs = map(1:N) do i
        x = randn(Nd, 100)
        x * x'
    end
    d = MixtureModel([MvNormal(μs[i], Σs[i]) for i in 1:N])
    D = @test_nowarn FNS.Density(d)
    @test D isa FNS.Densities.AdDistributionDensity
    p = rand(distribution(D))
    @test logdensity(D)(p) == logpdf(d, p)
    ps = eachcol(randn(Nd, 100))
    @test logdensity(D)(ps) == logpdf.([d], ps)
    @test gradlogdensity(D)(p) isa Vector{Float64}

    d = MixtureModel([Normal(0, 1), Normal(0, 0.5)])
    D = @test_nowarn FNS.Density(d)
    @test D.doAd == true
    @test gradlogdensity(D)(0) == 0.0
end

@testset "AdDistributionDensity" begin
    D = FNS.Densities.Density{true}(Normal(0.0, 0.5))
    x = zeros(LogDensityProblems.dimension(D)) # ℓ is your log density
    @inferred LogDensityProblems.logdensity(D, x) # check inference, also see @code_warntype
    ds = FNS.Densities.distribution(D)
    g = gradlogpdf(ds, -0.1)
    @test g == FNS.Densities.gradlogdensity(D, -0.1)
    if isinteractive()
        @benchmark gradlogpdf($ds, -0.1)
        @benchmark FNS.Densities.gradlogdensity($D, -0.1)
        @benchmark pdf($ds, $x) # check performance and allocations
        @benchmark ($D)($x) # check performance and allocations
        @benchmark LogDensityProblems.logdensity($D, $x) # check performance and allocations
    end
    @test only(LogDensityProblems.logdensity(D, [0.1])) ==
          LogDensityProblems.logdensity(D, 0.1)
    @test_nowarn Distributions.gradlogpdf(D, 0.1)
    @inferred gradlogdensity(D, [0.1])
    @test gradlogdensity(D, [0.1]) == [-0.4]
    @test only.(LogDensityProblems.logdensity_and_gradient(D, [0.1])) ==
          LogDensityProblems.logdensity_and_gradient(D, 0.1)
end

@testset "Basic Samplers" begin
    u0 = [0.01]
    tspan = (0.0, 1.0)
    f = (x, y, z, w) -> x
    S1 = @test_nowarn Sampler(f; u0, tspan)
    S2 = @test_nowarn Sampler(f, u0, tspan)
    @test S1.f == S2.f
    @test typeof(Density(S1)) == typeof(Density(S2))
end
@testset "Langevin Sampler" begin
    u0 = [0.0, 0.0]
    tspan = (0.0, 100.0)
    S = FNS.LangevinSampler(; u0, tspan, β = 1.0, γ = 10.0)
    D = FNS.Density(Normal(0, 1))
    a = @benchmark Density($S) # Can this be made faster?
    @test a.allocs == a.memory == 0
    @test Density(S).distribution == D.distribution
    @test Density(S).doAd == D.doAd
    sol = @test_nowarn solve(S; dt = 0.0001, saveat = 0.01)
    x = first.(sol.u)
    plot(x)
    density(x)
    @test x == trajectory(S)
end
@testset "Box boundaries" begin
    box = ReflectingBox(-5 .. 5)
    # box = FNS.NoBoundary()
    u0 = [0.0 1.0]
    tspan = (0.0, 1000.0) # Must be a matrix; col 1 is position, col2 is momentum
    # 𝜋 = FNS.Density(Normal(0, 0.25))
    𝜋 = FNS.Density(Uniform(-5, 5)) # No potential here is pathalogical; no transient to momentum equilibrium
    S = FNS.LangevinSampler(; u0, tspan, β = 1.0, γ = 0.1, boundaries = box(), 𝜋)
    sol = @test_nowarn solve(S; dt = 0.001, saveat = 0.1)
    x = first.(sol.u)
    y = last.(sol.u)
    @test minimum(x) ≥ -5
    @test maximum(x) ≤ 5
    lines(sol.t, x)
    lines(sol.t, y) # Momentum is constant?
    density(x) # The boundaries interfere with the density if they are too close
    # @test x == trajectory(S)

    box = FNS.ReflectingBox(-1 .. 1)
    u0 = [0.0 0.0]
    tspan = (0.0, 100.0)
    𝜋 = FNS.Density(Normal(0, 1))
    S = FNS.LangevinSampler(; u0, tspan, β = 0.5, γ = 0.1, boundaries = box(), 𝜋)
    sol = @test_nowarn solve(S; dt = 0.001, saveat = 0.01)
    x = first.(sol.u)
    y = last.(sol.u)
    minimum(x)
    lines(sol.t, x; linewidth = 3)
    lines(sol.t, y, linewidth = 3)
    density(x)
    @test minimum(x) ≥ -1 - 0.02
    @test maximum(x) ≤ 1 + 0.02

    box = FNS.PeriodicBox(-1 .. 1)
    u0 = [0.0 1.0]
    tspan = (0.0, 10.0)
    𝜋 = FNS.Density(Normal(0, 1))
    S = FNS.LangevinSampler(; u0, tspan, β = 1, γ = 0.1, boundaries = box(), 𝜋)
    sol = @test_nowarn solve(S; dt = 0.001, saveat = 0.01)
    x = first.(sol.u)
    y = last.(sol.u)
    minimum(x)
    lines(sol.t, x; linewidth = 3)
    lines(sol.t, y, linewidth = 3)
    density(x)
    @test minimum(x) ≥ -1 - 0.02
    @test maximum(x) ≤ 1 + 0.02

    box = NoBoundary()
    u0 = [0.0f0 1.0f0]
    tspan = (0.0f0, 10000.0f0)
    𝜋 = FNS.Density{true}(Laplace(0.0f0, 1.0f0))
    S = FNS.LangevinSampler(; u0, tspan, β = 1.0f0, γ = 1.0f0, boundaries = box(), 𝜋)
    # @benchmark solve(S; dt = 0.001, saveat = 0.01)
    sol = @test_nowarn solve(S; dt = 0.001f0, saveat = 0.1f0)
    x = first.(sol.u)
    density(x)
    f = fit(Laplace, x)
    @test f.μ≈0.0f0 atol=1e-3
    @test f.θ≈0.5f0 atol=1e-1
end

@testset "Oscillations under flat potential?" begin
    u0 = [0.0, 0.0]
    tspan = (0.0, 100.0)

    # * Quadratic potential (gaussian pdf)
    𝜋 = Normal(0.0, 1.0) |> Density
    S = FNS.LangevinSampler(; u0, tspan, 𝜋, β = 1.0, γ = 0.1)
    sol = solve(S; dt = 0.0001, saveat = 0.01)
    x = Timeseries(sol.t, first.(sol.u))
    plot(x) # Oscillating? Yes.
    hill(collect(x))

    # * Flat potential (uniform pdf... kind of. Discontinuity sucks. Add callback...boundary conditions...to handle this)
    𝜋 = Uniform(-0.5, 0.5) |> Density
    S = FNS.LangevinSampler(; u0, tspan, 𝜋, β = 1.0, γ = 0.1, callbacks = ...)
    @test distribution(Density(S)) == distribution(𝜋)
    sol = solve(S; dt = 0.0001, saveat = 0.01)
    x = Timeseries(sol.t, first.(sol.u))
    plot(x) # Oscillating? No; divergent. Can't really handle delta gradient
    hill(collect(x))
end

@testset "LevyNoise" begin
    import FractionalNeuralSampling.NoiseProcesses.LevyNoise
    DIST = LevyNoise{false}(2.0, 0.0, 1 / sqrt(2), 0.0)
    Random.seed!(42)
    rng = Random.default_rng()
    a = DIST(rng)
    Random.seed!(42)
    b = DIST(rng)
    Random.seed!(42)
    c = rand(rng, FNS.NoiseProcesses.dist(DIST))
    @test a == b == c

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

    DIST = LevyNoise{true}(2.0, 0.0, 1 / sqrt(2), 0.0)
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
    @test all(z .* 0.01 .^ (1 / DIST.α) .== y)
end

@testset "Test that adaptive stepping is disabled for LevySamplers" begin end

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
    dt = 0.00001
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
    @test L.dist.α≈f.α atol=1e-2
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

@testset "Benchmark LevyNoise" begin
    import FractionalNeuralSampling.NoiseProcesses.LevyNoise
    import FractionalNeuralSampling.NoiseProcesses.LevyNoise!
    import DiffEqNoiseProcess.WHITE_NOISE_DIST as W
    import DiffEqNoiseProcess.INPLACE_WHITE_NOISE_DIST as W!
    L = LevyNoise(2.0, 0.0, 1 / sqrt(2), 0.0)
    L! = LevyNoise!(2.0, 0.0, 1 / sqrt(2), 0.0)
    rng = Random.default_rng()

    X = zeros(100, 100)
    _L = Stable(2.0, 0.0, 1 / sqrt(2), 0.0)
    a = @benchmark randn(size($X))
    c = @benchmark $L($rng, $X)
    @test a.memory≈c.memory atol=10
    @test a.allocs == c.allocs == 2

    a = @benchmark $W($X, 0.0, 0.01, 0.0, 0.0, 0.0, $rng)
    b = @benchmark $L($X, 0.0, 0.01, 0.0, 0.0, 0.0, $rng)
    c = @benchmark $W!($X, 0.0, 0.01, 0.0, 0.0, 0.0, $rng)
    d = @benchmark $L!($X, 0.0, 0.01, 0.0, 0.0, 0.0, $rng)
    @test c.allocs == d.allocs == 0
    @test c.memory≈d.memory atol=10
    @test a.allocs == b.allocs == 4
    @test a.memory≈b.memory atol=10

    a = @benchmark rand!($rng, Stable(2.0, 0.0, 1 / sqrt(2), 0.0), $X)
    a = @benchmark $L!($rng, $X)
    @test a.allocs == a.memory == 0

    @benchmark Stable(2.0, 0.0, 1 / sqrt(2), 0.0) # * Super cheap
end

if CUDA.functional(true)
    @testset "GPU Benchmark" begin
        using DiffEqGPU
        function f3!(u0, u, p, t, W)
            u0[1] = 2u[1] * sin(W[1])
        end
        u0 = [1.00]
        tspan = (0.0, 5.0)
        dt = 0.01
        L = LevyProcess!(2.0)
        prob = RODEProblem{true}(f3!, u0, tspan; noise = L)
        ensemble = EnsembleProblem(prob)
        @test_nowarn @benchmark solve($ensemble, RandomEM(), EnsembleSerial();
                                      trajectories = 5,
                                      dt = $dt)
        @test_throws "MethodError" solve(ensemble, RandomEM(),
                                         EnsembleGPUArray(CUDA.CUDABackend());
                                         trajectories = 5, dt)
    end
end
