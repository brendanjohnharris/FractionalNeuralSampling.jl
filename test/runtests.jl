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

@testset "Langevin sampler bias" begin
    u0 = [0.0001 0.0001]
    tspan = (0.0, 100.0)
    dt = 0.01
    D = FNS.Density(Normal(0, 1))
    S = FNS.LangevinSampler(; u0, tspan, β = 1.0, γ = 1.0, 𝜋 = D,
                            noise = WienerProcess(0.0, 0.0))

    W = @test_nowarn remake(S, p = S.p)
    @test_nowarn solve(W, EM(); dt, saveat = 0.01)
    @test W.p == S.p
    @test W.u0 == S.u0
    @test W.tspan == S.tspan
    @test W.f == S.f
    @test W.g == S.g

    @test_nowarn KLDivergence()(D, randn(100))

    sol = solve(S, EM(); dt)
    density(first.(sol.u))
    lines!(-2.5:0.1:2.5, D.(-2.5:0.1:2.5))
    current_figure()
    @test mean(first.(sol.u))≈0.0 atol=0.05
    @test std(first.(sol.u))≈1.0 atol=0.01

    βs = range(0, 5, length = 10)

    er = map(βs) do β
        P = remake(S, p = ((β, S.p[1][2:end]...), S.p[2:end]...))
        ensemble = EnsembleProblem(P)
        sol = solve(ensemble, EM(); dt, trajectories = 1000)
        ts = [s[1, :] for s in sol]
        er = evaluate.([KLDivergence()], [D], ts) |> mean
        s = std.(ts) |> mean
        return (er, s)
    end
    stds = last.(er)
    ers = first.(er)
    lines(βs, ers)
    lines(βs, stds)
end

@testset "Levy sampler bias" begin
    u0 = [0.0 0.00]
    tspan = (0.0, 500.0)
    dt = 0.01
    D = FNS.Density(MixtureModel(Normal, [(-2, 0.5), (2, 0.5)]))
    S = FNS.LevyFlightSampler(; u0, tspan, α = 1.2, β = 0.1, γ = 0.5, 𝜋 = D)

    W = @test_nowarn remake(S, p = S.p)
    @test_nowarn solve(W, EM(); dt, saveat = 0.01)
    @test W.p == S.p
    @test W.u0 == S.u0
    @test W.tspan == S.tspan
    @test W.f == S.f
    @test W.g == S.g

    sol = solve(S, EM(); dt)
    x = first.(sol.u)
    x = x[abs.(x) .< 6]
    density(x)
    lines!(-4:0.1:4, D.(-4:0.1:4))
    current_figure()
    @test evaluate(KLDivergence(), D, x) < 0.1

    βs = range(0, 2, length = 20)
    er = map(βs) do β
        P = remake(S, p = ((S.p[1][1], β, S.p[1][3]), S.p[2:end]...))
        ensemble = EnsembleProblem(P)
        sol = solve(ensemble, EM(); dt, trajectories = 10)
        ts = map(sol) do s
            x = s[1, :]
            x = x[abs.(x) .< 6]
        end
        er = evaluate.([KLDivergence()], [D], ts) |> mean
        return er
    end
    lines(βs, er)

    αs = range(1.1, 2.0, length = 20)
    er = map(αs) do α
        P = remake(S, p = ((α, S.p[1][2:end]...), S.p[2:end]...))
        ensemble = EnsembleProblem(P)
        sol = solve(ensemble, EM(), EnsembleThreads(); dt, trajectories = 100)
        ts = map(sol) do s
            x = s[1, :]
            x = x[abs.(x) .< 6]
        end
        er = evaluate.([KLDivergence()], [D], ts) |> mean
        return er
    end
    lines(αs, er)

    αs = range(1.25, 2.0, length = 52)
    βs = range(0, 2, length = 51)
    ps = Iterators.product(αs, βs) .|> collect
    push!.(ps, 1.0) # Add γ
    ers = pmap(ps) do p
        P = remake(S, p = (Tuple(p), S.p[2:end]...))
        ensemble = EnsembleProblem(P)
        sol = solve(ensemble, EM(); dt, trajectories = 1000)
        ts = [s[1, :] for s in sol]
        ts = map(sol) do s
            x = s[1, :]
            x = x[abs.(x) .< 6]
        end
        er = evaluate.([KLDivergence()], [D], ts) |> mean
        return er
    end
    fax = heatmap(αs, βs, (ers); axis = (; xlabel = "α", ylabel = "β"))
    Colorbar(fax.figure[1, 2], fax.plot; label = "KL Divergence")
    fax
    # heatmap(αs, βs, stds; axis = (; xlabel = "α", ylabel = "β"))
end

begin
    u0 = [-3.0 0.00]
    tspan = (0.0, 5000.0)
    dt = 0.01
    D = FNS.Density(MixtureModel(Laplace, [(-3, 0.3), (3, 0.3)]))
    # D = FNS.Density(Normal(-2, 0.5))
    S = FNS.LevyFlightSampler(; u0, tspan, α = 1.5, β = 0.0, γ = 0.1, 𝜋 = D)

    sol = solve(S, EM(); dt)
    x = first.(sol.u)
    x = x[abs.(x) .< 8]
    density(filter(!isnan, x))
    lines!(-4:0.1:4, D.(-4:0.1:4))
    current_figure()

    f = Figure(size = (800, 400))
    ax = Axis(f[1, 1])
    lines!((1:10000), x[1:10000])
    ax = Axis(f[1, 2])
    xx = Timeseries((1:length(x)), x)
    lines!((1:5000), autocor(centraldiff(xx), 1:5000))
    # spectrumplot(spectrum(xx, 1))
    f
end

begin # * Simple potential: power law iqr?
    u0 = [-0.001 0.00]
    tspan = (0.0, 1.0)
    dt = 0.00001
    D = FNS.Density(Normal(0, 1))
    S = FNS.LevyFlightSampler(; u0, tspan, α = 1.2, β = 0.0, γ = 0.1, 𝜋 = D)
    P = EnsembleProblem(S)
    sol = solve(P, EM(); dt, trajectories = 100)
    σ = mapslices(iqr, stack(getindex.(sol.u, 1, :)); dims = 2)[:] .^ 2

    mm = first([log10.(2:1000) ones(length(σ[2:1000]))] \ log10.(σ[2:1000]))

    @test 1.6 < mm < 1.7

    lines(σ[2:10000]; axis = (; xscale = log10, yscale = log10))
    lines!((2:10000) ./ 1e8)
    current_figure()
end

begin # * Unimodal vs bimodal comparison
    Random.seed!(43)
    f = Figure(size = (720, 360))
    u0 = [-0.001 0.00]
    tspan = (0.0, 1000.0)
    dt = 0.001
    idxs = range(start = 1, step = 200, length = 500)
    # ft = identity
    ft = x -> abs.((x[3:end] .- x[1:(end - 2)]) ./ 2)

    ax = Axis(f[1, 1]; xlabel = "t", ylabel = "v", title = "Unimodal")
    D = FNS.Density(Laplace(0, 0.5))
    S = FNS.LevyFlightSampler(; u0, tspan, α = 1.4, β = 2.0, γ = 0.5, 𝜋 = D)
    sol = solve(S, EM(); dt)
    x = sol[1, :][idxs]
    lines!(ax, ft(x), color = :cornflowerblue, linewidth = 2)

    ax = Axis(f[1, 2])
    hist!(ax, ft(x); direction = :x, bins = 50, color = :gray)
    hidedecorations!(ax)
    hidespines!(ax)
    tightlimits!(ax)

    u0 = [-0.201 0.00]
    ax = Axis(f[2, 1]; xlabel = "t", ylabel = "v", title = "Bimodal")
    D = FNS.Density(MixtureModel([Laplace(-1, 0.5), Laplace(1, 0.5)]))
    S = FNS.LevyFlightSampler(; u0, tspan, α = 1.4, β = 2.0, γ = 0.5, 𝜋 = D)
    sol = solve(S, EM(); dt)
    x = sol[1, :][idxs]

    lines!(ax, ft(x); color = :crimson, linewidth = 2)

    ax = f[2, 2] |> Axis
    hist!(ax, ft(x); direction = :x, bins = 50, color = :gray)
    hidedecorations!(ax)
    hidespines!(ax)
    tightlimits!(ax)

    colgap!(f.layout, 1, Relative(0))
    colsize!(f.layout, 2, Relative(0.2))
    linkyaxes!(contents(f.layout)...)
    f
end

begin # * Fixation simulation: heavy tailed msd??
    u0 = [-0.001 0.00]
    tspan = (0.0, 100.0)
    dt = 0.001
    D = FNS.Density(Laplace(0, 1))
    S = FNS.LevyWalkSampler(; u0, tspan, α = 2.0, β = 0.1, γ = 0.1, 𝜋 = D)

    sol = solve(S, EM(); dt)
    x = first.(sol.u)[1:50:end] # Need heavy oversampling to prevent blowout
    x = x[abs.(x) .< 8]
    density(filter(!isnan, x))
    lines!(-4:0.1:4, D.(-4:0.1:4))
    # current_axis().yscale = log10
    current_axis().limits = (nothing, (1e-3, nothing))
    current_figure()

    d = map(1:5000) do t
        mean((x[1:(end - t)] .- x[(t + 1):end]) .^ 2)
    end
    plot(d; axis = (; xscale = log10, yscale = log10))
    a = d[1:100]
    b = hcat(ones(length(a)), log.(1:length(a))) \ log.(a)
    lines!(1:100, exp(b[1]) * (1:100) .^ b[2]; color = :red)
    text!(1, 1; text = "α=$(b[2])")
    current_figure()

    lines(x[1:10:1000])

    c = collect(centraldiff(centraldiff(centraldiff(centraldiff(centraldiff(Timeseries(1:length(x),
                                                                                       x)))))))
    c = c[abs.(c) .< 0.01]
    hist(c, bins = 100, normalization = :pdf)
    f = fit(Stable, c)
    lines!(-0.005:0.00001:0.005, pdf.([f], -0.005:0.00001:0.005))
    current_figure()
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

@testset "Test that adaptive stepping is disabled for LevyFlightSamplers" begin end

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
