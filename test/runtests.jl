using Test
using TestItems
using TestItemRunner

@run_package_tests

@testitem "Aqua.jl" begin
    using Aqua
    Aqua.test_all(FractionalNeuralSampling; persistent_tasks = false)
end

@testsnippet Setup begin
    using RecursiveArrayTools
    using CairoMakie
    using Foresight
    using FractionalNeuralSampling
    import FractionalNeuralSampling: Density
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
    using FileIO
    using StatsBase
    using Autocorrelations
    using TimeseriesTools
    import TimeseriesTools.msdist
    using Normalization
    import FractionalNeuralSampling: Density
    set_theme!(foresight(:physics))
end

@testitem "Density" setup=[Setup] begin
    # Use a normal pdf
    f(x) = 1 / sqrt(2π) * exp(-x^2 / 2)
    f(x::AbstractVector{T}) where {T} = f(only(x))::T # Ensure univariate consistency
    D = @inferred Density{typeof(f), 1, false}(f) # 1D, no autodiff
    D = @inferred Density{1, false}(f) # 1D, no autodiff
    D = Density{1}(f, false) # Not type stable??
    @test_throws "MethodError" gradlogdensity(D, 0.0) # No autodiff, so errors

    D = @inferred Density{1}(f) # 1D, do autodiff
    d = @inferred D(0.0)
    logd = @inferred logdensity(D, 0.0)
    glogd = @inferred gradlogdensity(D, 0.0)
    @inferred gradlogdensity(D, [0])
    @test_throws "ArgumentError" gradlogdensity(D, [0.0, 0.0])
    @inferred gradlogdensity(D, [[0.0], [1.0]])

    @inferred potential(D, 0.0)
    @inferred gradpotential(D, 0.0)
    @test gradpotential(D) isa Function

    @inferred FractionalNeuralSampling.Densities.logdensity_and_gradient(D, 0.0)
end

@testitem "Langevin sampler bias" setup=[Setup] begin
    u0 = [0.0, 0.0]
    tspan = (0.0, 10000.0)
    dt = 0.01
    D = Density(Normal(0, 1))
    S = Langevin(; u0, tspan, β = 1.0, η = 1.0, 𝜋 = D)

    W = @test_nowarn remake(S, p = S.p)
    @test_nowarn solve(W, EM(); dt, saveat = 0.01)
    @test W.p == S.p
    @test W.u0 == S.u0
    @test W.tspan == S.tspan
    @test W.f == S.f
    @test W.g == S.g

    @test_nowarn KLDivergence()(D, randn(1000))
    sol = solve(S, EM(); dt)
    Makie.hist(first.(sol.u), bins = 50, normalization = :pdf)
    lines!(-2.5:0.1:2.5, D.(-2.5:0.1:2.5))
    current_figure()
    @test mean(first.(sol.u))≈0.0 atol=0.05
    @test std(first.(sol.u))≈1.0 atol=0.05

    if false
        tspan = (0.0, 100.0)
        S = Langevin(; u0, tspan, β = 1.0, η = 1.0, 𝜋 = D,
                     noise = WienerProcess(0.0, 0.0))
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
end

@testitem "Levy sampler bias" setup=[Setup] begin
    u0 = [0.0, 0.0]
    tspan = (0.0, 1000.0)
    dt = 0.01
    D = Density(MixtureModel(Normal, [(-2, 0.5), (2, 0.5)]))
    S = FractionalNeuralSampler(; u0, tspan, α = 1.4, β = 0.1, γ = 0.5, 𝜋 = D)

    W = @test_nowarn remake(S, p = S.p)
    @test_nowarn solve(W, EM(); dt, saveat = 0.01)
    @test W.p == S.p
    @test W.u0 == S.u0
    @test W.tspan == S.tspan
    @test W.f == S.f
    @test W.g == S.g

    sol = solve(S; dt)
    x = first.(sol.u)
    x = x[abs.(x) .< 6]
    density(x)
    lines!(-4:0.1:4, D.(-4:0.1:4))
    current_figure()
    @test evaluate(KLDivergence(), D, x) < 0.2

    if false
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
end

@testitem "Space-fractional neural sampling bias" setup=[Setup] begin
    u0 = [0.0, 0.0]
    tspan = (0.0, 5000.0)
    dt = 0.05
    D = Density(MixtureModel(Normal, [(-2, 0.5), (2, 0.5)]))
    d = -15 .. 15
    boundaries = PeriodicBox(-7 .. 7)
    S = sFNS(; u0, tspan, α = 1.5, β = 0.05, γ = 0.5, 𝜋 = D, domain = d, boundaries)

    W = @test_nowarn remake(S, p = S.p)
    @test_nowarn solve(W, EM(); dt, saveat = 0.01)
    @test W.p == S.p
    @test W.u0 == S.u0
    @test W.tspan == S.tspan
    @test W.f == S.f
    @test W.g == S.g

    sol = solve(S; dt)
    x = first.(sol.u)
    x = x[abs.(x) .< 6]
    hist(x; normalization = :pdf, bins = 50)
    lines!(-4:0.1:4, D.(-4:0.1:4); color = :crimson)
    current_figure()
    @test evaluate(KLDivergence(), D, x) < 0.05
end

@testitem "2D sFNS" setup=[Setup] begin
    u0 = ArrayPartition([0.0, 0.0], [0.0, 0.0])
    tspan = (0.0, 50.0)
    dt = 0.05
    D = Density(MixtureModel(MvNormal, [([-2, -2], I(2)), ([2, 2], I(2))]))
    d = (-7 .. 7, -7 .. 7)
    boundaries = PeriodicBox(-5 .. 5, -5 .. 5)
    S = sFNS(; u0, tspan, α = 1.5, β = 0.05, γ = 0.5, 𝜋 = D, domain = d, boundaries,
             approx_n_modes = 1000)

    W = @test_nowarn remake(S, p = S.p)
    @test_nowarn solve(W, EM(); dt, saveat = 0.01)
    @test W.p == S.p
    @test W.u0 == S.u0
    @test W.tspan == S.tspan
    @test W.f == S.f
    @test W.g == S.g

    sol = solve(S; dt)
    # ! Tests divergence against target distribution
end

@testitem "Recursive Arrays" setup=[Setup] begin
    # So for recursive arrays, we can use diagonal noise by setting the noise_rate prototype
    # to similar(u0)
    u0 = ArrayPartition([0.0], [0.0])
    tspan = (0.0, 500.0)
    dt = 0.01
    D = Density(MixtureModel(Normal, [(-2, 0.5), (2, 0.5)]))
    S = FractionalNeuralSampler(; u0, tspan, α = 1.2, β = 0.1, γ = 0.5, 𝜋 = D)

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
    @test evaluate(KLDivergence(), D, x) < 0.1
end

if false
    u0 = [-3.0 0.00]
    tspan = (0.0, 5000.0)
    dt = 0.01
    D = Density(MixtureModel(Laplace, [(-3, 0.3), (3, 0.3)]))
    # D = Density(Normal(-2, 0.5))
    S = FractionalNeuralSampler(; u0, tspan, α = 1.5, β = 0.0, γ = 0.1, 𝜋 = D)

    sol = solve(S, EM(); dt)
    x = first.(sol.u)
    x = x[abs.(x) .< 8]
    density(filter(!isnan, x))
    lines!(-4:0.1:4, D.(-4:0.1:4))
    current_figure()

    g = Figure(size = (800, 400))
    ax = Axis(g[1, 1])
    lines!((1:10000), x[1:10000])
    ax = Axis(g[1, 2])
    xx = Timeseries((1:length(x)), x)
    lines!((1:5000), autocor(centraldiff(xx), 1:5000))
    # spectrumplot(spectrum(xx, 1))
    g
end

if false # * Simple potential: power law iqr?
    u0 = [-0.001 0.00]
    tspan = (0.0, 1.0)
    dt = 0.00001
    D = Density(Normal(0, 1))
    S = FractionalNeuralSampler(; u0, tspan, α = 1.2, β = 0.0, γ = 0.1, 𝜋 = D)
    P = EnsembleProblem(S)
    sol = solve(P, EM(); dt, trajectories = 100)
    σ = mapslices(iqr, stack(getindex.(sol.u, 1, :)); dims = 2)[:] .^ 2

    mm = first([log10.(2:1000) ones(length(σ[2:1000]))] \ log10.(σ[2:1000]))

    @test 1.6 < mm < 1.7

    lines(σ[2:10000]; axis = (; xscale = log10, yscale = log10))
    lines!((2:10000) ./ 1e8)
    current_figure()
end

if false # * Unimodal vs bimodal comparison
    Random.seed!(43)
    g = Figure(size = (720, 360))
    u0 = [-0.001 0.00]
    tspan = (0.0, 1000.0)
    dt = 0.001
    idxs = range(start = 1, step = 200, length = 500)
    # ft = identity
    ft = x -> abs.((x[3:end] .- x[1:(end - 2)]) ./ 2)

    ax = Axis(g[1, 1]; xlabel = "t", ylabel = "v", title = "Unimodal")
    D = Density(Laplace(0, 0.5))
    S = FractionalNeuralSampler(; u0, tspan, α = 1.4, β = 2.0, γ = 0.5, 𝜋 = D)
    sol = solve(S, EM(); dt)
    x = sol[1, :][idxs]
    lines!(ax, ft(x), color = :cornflowerblue, linewidth = 2)

    ax = Axis(g[1, 2])
    hist!(ax, ft(x); direction = :x, bins = 50, color = :gray)
    hidedecorations!(ax)
    hidespines!(ax)
    tightlimits!(ax)

    u0 = [-0.201 0.00]
    ax = Axis(g[2, 1]; xlabel = "t", ylabel = "v", title = "Bimodal")
    D = Density(MixtureModel([Laplace(-1, 0.5), Laplace(1, 0.5)]))
    S = FractionalNeuralSampler(; u0, tspan, α = 1.4, β = 2.0, γ = 0.5, 𝜋 = D)
    sol = solve(S, EM(); dt)
    x = sol[1, :][idxs]

    lines!(ax, ft(x); color = :crimson, linewidth = 2)

    ax = g[2, 2] |> Axis
    hist!(ax, ft(x); direction = :x, bins = 50, color = :gray)
    hidedecorations!(ax)
    hidespines!(ax)
    tightlimits!(ax)

    colgap!(g.layout, 1, Relative(0))
    colsize!(g.layout, 2, Relative(0.2))
    linkyaxes!(contents(g.layout)...)
    g
end

if false # * Fixation simulation: heavy tailed msd??
    u0 = [-0.001 0.00]
    tspan = (0.0, 100.0)
    dt = 0.001
    D = Density(Laplace(0, 1))
    S = FHMC(; u0, tspan, α = 2.0, β = 0.1, γ = 0.1, 𝜋 = D)

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
    g = fit(Stable, c)
    lines!(-0.005:0.00001:0.005, pdf.([g], -0.005:0.00001:0.005))
    current_figure()
end

@testitem "Autodiff" setup=[Setup] begin
    D = Normal(0, 0.5)
    @inferred logpdf(D, 0.1)
    g(x::T) where {T <: Real} = logpdf(D, x)::T
    g(x::AbstractVector{T}) where {T <: Real} = logpdf(D, only(x))::T
    @inferred g(0.1)
    # @inferred ForwardDiff.gradient(g, [0.1])
    backend = AutoForwardDiff()
    @test gradlogdensity(Density(D, true)).(0.1:0.1:3) ==
          gradlogpdf.([D], 0.1:0.1:3)
    gr = gradient(g, backend, [0.1]) # @inferred
    @test gr isa Vector
    @test length(gr) == 1
    a = @benchmark gradient($g, $backend, [0.1])
    b = @benchmark ForwardDiff.gradient($g, [0.1])
    c = @benchmark gradlogpdf($D, 0.1)
    @test a.allocs < 15
    @test b.allocs < 10
    @test c.allocs == c.memory == 0
    @inferred gradlogdensity(Density(D, false), 0.1)
    a = @benchmark gradlogdensity(Density($D, false), 0.1)
    @test a.allocs == c.memory == 0
    @inferred gradlogdensity(Density(D, true), 0.1)
    b = @benchmark gradlogdensity(Density($D, true), 0.1)
    @test b.allocs < 15 # Slightly allocating

    cl = @code_lowered Densities._gradlogdensity(Density(D, true), 0.1)
    @test contains(string(cl.code), "AD_BACKEND")
end
@testitem "Univariate DistributionDensity" setup=[Setup] begin
    d = Normal(0.0, 0.5)
    D = @test_nowarn Density(d)
    @test D isa Densities.AbstractUnivariateDensity
    @test Densities.doautodiff(D) == false
    @test D(0.0) == 2 / sqrt(2π)
    @test D([0.0]) == 2 / sqrt(2π)
    @test LogDensityProblems.dimension(D) == 1
    @test all(map(LogDensityProblems.logdensity(D), -1:0.1:1) .≈ log.(D.(-1:0.1:1)) .≈
              logpdf(distribution(D), -1:0.1:1))
    @inferred LogDensityProblems.logdensity(D, 0.0)
    @inferred LogDensityProblems.logdensity(D, 0)
    lines(-2:0.1:2, D.(-2:0.1:2))
    lines(-2:0.01:2, Densities.potential(D).(-2:0.01:2))
    @inferred Densities.gradlogdensity(D, 0.01)
    @inferred map(Densities.gradlogdensity(D), 0.01:0.01:5)
    @test map(Densities.gradlogdensity(D), 0.1:0.1:5) == gradlogpdf.([d], 0.1:0.1:5)

    d = Uniform(-0.5, 0.5)
    D = @test_nowarn Density(d)
    @test D(0.0) == 1
    @test LogDensityProblems.dimension(D) == 1
    @test all(map(LogDensityProblems.logdensity(D), -1:0.1:1) .≈ log.(D.(-1:0.1:1)))
    @inferred LogDensityProblems.logdensity(D, 0.0)
    @inferred LogDensityProblems.logdensity(D, -0.6)
    @inferred LogDensityProblems.logdensity_and_gradient(D, -0.6)
    @inferred LogDensityProblems.logdensity_and_gradient(Density(d, true), 0.5)

    if isinteractive()
        @benchmark map(Densities.gradlogdensity($D), -1:0.01:1)
        @benchmark LogDensityProblems.logdensity_and_gradient.([$D], -1:0.01:1)
    end
    lines(-2:0.1:2, D.(-2:0.1:2))
    lines(-2:0.01:2, Densities.potential(D).(-2:0.01:2))
    lines(-2:0.01:2, Densities.gradlogdensity(D).(-2:0.01:2))

    D = @test_nowarn Density(Normal(0.0f0, 0.5f0))
    @test LogDensityProblems.logdensity(D, 0.0f0) isa Float32
    @test Densities.gradlogdensity(D, 0.0f0) isa Float32
end

@testitem "Multivariate DistributionDensity" setup=[Setup] begin
    N = 3
    μs = randn(N)
    x = randn(N, 100)
    Σ = x * x'
    d = MvNormal(μs, Σ)
    D = @test_nowarn Density(d)
    @test D isa Densities.DistributionDensity
    @test !(D isa Densities.AbstractUnivariateDensity)
    @test Densities.doautodiff(D) == false
    p = randn(N)
    @test logdensity(D)(p) == logpdf(d, p)
    ps = eachcol(randn(N, 100))
    @test logdensity(D)(ps) == logpdf.([d], ps)
    @test gradlogdensity(D)(p) == gradlogpdf(d, p)
    @test map(gradlogdensity(D), ps) == gradlogpdf.([d], ps)

    # * Ad
    D = @test_nowarn Density(MvNormal(μs, Σ), true)
    @test Densities.doautodiff(D) == true
    @test LogDensityProblems.logdensity(D, p) isa Float64
    @test logdensity(D)(p) == logpdf(d, p)
    @test logdensity(D)(ps) == logpdf.([d], ps)
    @test gradlogdensity(D)(p) ≈ gradlogpdf(d, p)
end

@testitem "Mixture DistributionDensity" setup=[Setup] begin
    Nd = 3
    N = 10
    μs = [randn(Nd) for _ in 1:N]
    Σs = map(1:N) do i
        x = randn(Nd, 100)
        x * x'
    end
    d = MixtureModel([MvNormal(μs[i], Σs[i]) for i in 1:N])
    D = @test_nowarn Density(d)
    @test D isa Densities.AdDensity
    p = rand(D) # Draw from the distribution
    @test logdensity(D)(p) == logpdf(d, p)
    ps = eachcol(randn(Nd, 100))
    @test logdensity(D)(ps) == logpdf.([d], ps)
    @test gradlogdensity(D)(p) isa Vector{Float64}

    d = MixtureModel([Normal(0, 1), Normal(0, 0.5)])
    D = @test_nowarn Density(d)
    @test Densities.doautodiff(D) == true
    @test gradlogdensity(D)(0.0) == 0.0
end

@testitem "AdDistributionDensity" setup=[Setup] begin
    D = @inferred Densities.Density(Normal(0.0, 0.5))
    D = @inferred Densities.Density{true}(Normal(0.0, 0.5))
    x = zeros(LogDensityProblems.dimension(D)) # ℓ is your log density
    @inferred LogDensityProblems.logdensity(D)(x) # check inference, also see @code_warntype
    ds = Densities.distribution(D)
    g = gradlogpdf(ds, -0.1)
    @test g == Densities.gradlogdensity(D, -0.1)
    if isinteractive()
        @benchmark gradlogpdf($ds, -0.1)
        @benchmark Densities.gradlogdensity($D, -0.1)
        @benchmark pdf($ds, $x) # check performance and allocations
        @benchmark ($D)($x) # check performance and allocations
        @benchmark LogDensityProblems.logdensity($D, $x) # check performance and allocations
    end
    @test only(LogDensityProblems.logdensity(D, [0.1])) ==
          LogDensityProblems.logdensity(D, 0.1)
    @test_nowarn Distributions.gradlogpdf(D)(0.1)
    @inferred gradlogdensity(D, [0.1])
    @test gradlogdensity(D, [0.1]) == [-0.4]
    @test only.(LogDensityProblems.logdensity_and_gradient(D, [0.1])) ==
          LogDensityProblems.logdensity_and_gradient(D, 0.1)
end

@testitem "Overdamped Langevin Sampler" setup=[Setup] begin
    u0 = [0.0]
    tspan = (0.0, 100.0)
    D = Density(Normal(0, 10.0))
    S = OLE(; u0, tspan, η = 1, 𝜋 = D)

    sol = @test_nowarn solve(S, EM(); dt = 0.001, saveat = 0.01)
    x = first.(sol.u)
    plot(x)
    density(x)

    # * Try setting parameters
    s = S(η = 0.1)
    @test s.p[1][:η] == 0.1

    s = S(; η = 10.0, tspan = 1000.0, 𝜋 = D)
    @test s.p[1][:η] == 10.0
    @test s.p[2] == D
    @test s.tspan == 1000.0
end

@testitem "Box boundaries" setup=[Setup] begin
    box = ReflectingBox(-5 .. 5)
    # box = NoBoundary()
    u0 = [0.0, 1.0]
    tspan = (0.0, 1000.0)
    # 𝜋 = Density(Normal(0, 0.25))
    𝜋 = Density(Uniform(-5, 5)) # No potential here is pathalogical; no transient to momentum equilibrium
    S = Langevin(; u0, tspan, β = 1.0, η = 0.1, boundaries = box(), 𝜋)
    sol = @test_nowarn solve(S; dt = 0.001, saveat = 0.1)
    x = first.(sol.u)
    y = last.(sol.u)
    @test minimum(x) ≥ -5 - 2e-2
    @test maximum(x) ≤ 5 + 2e-2
    lines(sol.t, x)
    lines(sol.t, y) # Momentum is constant?
    density(x) # The boundaries interfere with the density if they are too close
    # @test x == trajectory(S)

    box = ReflectingBox(-1 .. 1)
    u0 = [0.0, 0.0]
    tspan = (0.0, 100.0)
    𝜋 = Density(Normal(0, 1))
    S = Langevin(; u0, tspan, β = 0.5, η = 0.1, boundaries = box(), 𝜋)
    sol = @test_nowarn solve(S; dt = 0.001, saveat = 0.01)
    x = first.(sol.u)
    y = last.(sol.u)
    minimum(x)
    lines(sol.t, x; linewidth = 3)
    lines(sol.t, y, linewidth = 3)
    density(x)
    @test minimum(x) ≥ -1 - 0.05
    @test maximum(x) ≤ 1 + 0.05

    box = PeriodicBox(-1 .. 1)
    u0 = [0.0, 1.0]
    tspan = (0.0, 10.0)
    𝜋 = Density(Normal(0, 1))
    S = Langevin(; u0, tspan, β = 1, η = 0.1, boundaries = box(), 𝜋)
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
    u0 = [0.0f0, 1.0f0]
    tspan = (0.0f0, 10000.0f0)
    𝜋 = Density(Laplace(0.0f0, 1.0f0), true)
    S = Langevin(; u0, tspan, β = 1.0f0, η = 1.0f0, boundaries = box(), 𝜋)
    # @benchmark solve(S; dt = 0.001, saveat = 0.01)
    sol = @test_nowarn solve(S; dt = 0.01f0, saveat = 0.1f0)
    x = first.(sol.u)
    density(x)
    gg = fit(Laplace, x)
    @test gg.μ≈0.0f0 atol=5e-2
    @test gg.θ≈1.0f0 atol=1e-1
end

# @testitem "Oscillations under flat potential?" setup=[Setup] begin
# if false # !! Add callbacks for discontinuities
#     u0 = [0.0 0.0]
#     tspan = (0.0, 100.0)

#     # * Quadratic potential (gaussian pdf)
#     𝜋 = Normal(0.0, 1.0) |> Density
#     S = Langevin(; u0, tspan, 𝜋, β = 1.0, η = 0.1)
#     sol = solve(S; dt = 0.0001, saveat = 0.01)
#     x = Timeseries(sol.t, first.(sol.u))
#     plot(x) # Oscillating? Yes.
#     hill(collect(x))

#     # * Flat potential (uniform pdf... kind of. Discontinuity sucks. Add callback...boundary conditions...to handle this)
#     𝜋 = Uniform(-0.5, 0.5) |> Density
#     S = Langevin(; u0, tspan, 𝜋, β = 1.0, η = 0.1, callbacks = ...)
#     @test density(Density(S)) == density(𝜋)
#     sol = solve(S; dt = 0.0001, saveat = 0.01)
#     x = Timeseries(sol.t, first.(sol.u))
#     plot(x) # Oscillating? No; divergent. Can't really handle delta gradient
#     hill(collect(x))
# end

# @testitem "LevyNoise" setup=[Setup] begin
if false # ! Need to fix out-of-place noise
    import FractionalNeuralSampling.NoiseProcesses.LevyNoise
    DIST = LevyNoise{false}(2.0, 0.0, 1 / sqrt(2), 0.0)
    Random.seed!(42)
    rng = Random.default_rng()
    a = DIST(rng)
    Random.seed!(42)
    b = DIST(rng)
    Random.seed!(42)
    c = rand(rng, NoiseProcesses.dist(DIST))
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

# @testitem "Test that adaptive stepping is disabled for FractionalNeuralSamplers" setup=[Setup] begin end

# @testitem "FractionalNeuralSampling.jl" setup=[Setup] begin
#     include("fractional_sampling.jl")
# end

# @testitem "LevyProcess" setup=[Setup] begin
if false # ! Need to fix out-of-place noise
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

# @testitem "Brownian Noise" setup=[Setup] begin
if false # ! Need to fix out-of-place noise
    prob = NoiseProblem(LevyProcess(2.0), (0.0, 1.0))
    dt = 0.00001
    ensemble = EnsembleProblem(prob)
    sol = solve(prob, RandomEM(); dt)

    lines(sol.t, sol.u; linewidth = 2)
    @test std(diff(sol.u))≈sqrt(dt) rtol=1e-2
end

# @testitem "Levy Noise" setup=[Setup] begin
if false # ! Need to fix out-of-place noise
    L = LevyProcess(1.5)
    prob = NoiseProblem(L, (0.0, 1.0))
    dt = 1e-6
    sol = solve(prob, RandomEM(); dt)

    g = fit(Stable, diff(sol.u) ./ (dt^(1 / 1.5)))
    @test L.dist.α≈g.α atol=1e-2
end

# @testitem "Ensemble" setup=[Setup] begin
if false # ! Need to fix out-of-place noise
    Random.seed!(42)
    L = LevyProcess(1.5)
    dt = 1e-3
    prob = NoiseProblem(L, (0.0, 1.0))
    ensemble = EnsembleProblem(prob)
    sol = @test_nowarn solve(ensemble, RandomEM(), EnsembleSerial(); trajectories = 5, dt)
    @test_nowarn solve(ensemble, RandomEM(), EnsembleDistributed(); trajectories = 5, dt)
    @test_nowarn solve(ensemble, RandomEM(), EnsembleThreads(); trajectories = 5, dt)
    g = Figure()
    ax = Axis(g[1, 1])
    [lines!(ax, s.t, s.u) for s in sol]
    display(g)
end

# @testitem "Benchmark LevyNoise" setup=[Setup] begin
# if false # ! Need to fix out-of-place noise
#     import FractionalNeuralSampling.NoiseProcesses.LevyNoise
#     import FractionalNeuralSampling.NoiseProcesses.LevyNoise!
#     import DiffEqNoiseProcess.WHITE_NOISE_DIST as W
#     import DiffEqNoiseProcess.INPLACE_WHITE_NOISE_DIST as W!
#     L = LevyNoise(2.0, 0.0, 1 / sqrt(2), 0.0)
#     L! = LevyNoise!(2.0, 0.0, 1 / sqrt(2), 0.0)
#     rng = Random.default_rng()

#     X = zeros(100, 100)
#     _L = Stable(2.0, 0.0, 1 / sqrt(2), 0.0)
#     a = @benchmark randn(size($X))
#     c = @benchmark $L($rng, $X)
#     @test a.memory≈c.memory atol=10
#     @test a.allocs == c.allocs == 2

#     a = @benchmark $W($X, 0.0, 0.01, 0.0, 0.0, 0.0, $rng)
#     b = @benchmark $L($X, 0.0, 0.01, 0.0, 0.0, 0.0, $rng)
#     c = @benchmark $W!($X, 0.0, 0.01, 0.0, 0.0, 0.0, $rng)
#     d = @benchmark $L!($X, 0.0, 0.01, 0.0, 0.0, 0.0, $rng)
#     @test c.allocs == d.allocs == 0
#     @test c.memory≈d.memory atol=10
#     @test a.allocs == b.allocs == 4
#     @test a.memory≈b.memory atol=10

#     a = @benchmark rand!($rng, Stable(2.0, 0.0, 1 / sqrt(2), 0.0), $X)
#     a = @benchmark $L!($rng, $X)
#     @test a.allocs == a.memory == 0

#     @benchmark Stable(2.0, 0.0, 1 / sqrt(2), 0.0) # * Super cheap
# end

# if CUDA.functional(true)
#     @testitem "GPU Benchmark" setup=[Setup] begin
#         using DiffEqGPU
#         function f3!(u0, u, p, t, W)
#             u0[1] = 2u[1] * sin(W[1])
#         end
#         u0 = [1.00]
#         tspan = (0.0, 5.0)
#         dt = 0.01
#         L = LevyProcess!(2.0)
#         prob = RODEProblem{true}(f3!, u0, tspan; noise = L)
#         ensemble = EnsembleProblem(prob)
#         @test_nowarn @benchmark solve($ensemble, RandomEM(), EnsembleSerial();
#                                       trajectories = 5,
#                                       dt = $dt)
#         @test_throws "MethodError" solve(ensemble, RandomEM(),
#                                          EnsembleGPUArray(CUDA.CUDABackend());
#                                          trajectories = 5, dt)
#     end
# end

# @testitem "2D potential" setup=[Setup] begin
# if false
#     using DifferentialEquations
#     Δx = 5
#     d = MixtureModel([
#                          MvNormal([-Δx / 2, 0], [1 0; 0 1]),
#                          MvNormal([Δx / 2, 0], [1 0; 0 1])
#                      ])
#     D = Density(d)

#     αs = [2.0, 1.6, 1.2]
#     f = Figure(size = (900, 300))
#     gs = subdivide(f, 1, 3)
#     map(αs, gs) do α, g
#         L = FractionalNeuralSampler(;
#                                     u0 = [-Δx / 2 0 0 0],
#                                     tspan = 500.0,
#                                     α = α,
#                                     β = 0.2,
#                                     γ = 0.02,
#                                     𝜋 = D,
#                                     seed = 42)
#         sol = solve(L, EM(); dt = 0.001)
#         x, y = eachrow(sol[1:2, :])

#         xmax = maximum(abs.(extrema(vcat(x, y)))) * 1.5
#         xs = range(-xmax, xmax, length = 100)

#         ax = Axis(g[1, 1], title = "α = $α", aspect = DataAspect())
#         # heatmap!(ax, xs, xs, potential(D).(collect.(Iterators.product(xs, xs))), colormap=seethrough(:turbo))
#         heatmap!(ax, xs, xs, D.(collect.(Iterators.product(xs, xs))),
#                  colormap = seethrough(:turbo))
#         lines!(ax, x[1:10:end], y[1:10:end], color = (:black, 0.5), linewidth = 1)
#         hidedecorations!(ax)
#     end
#     f
# end

# @testitem "MSD check" setup=[Setup] begin
# begin
#     u0 = [0.0, 0.0]
#     tspan = (0.0, 10000.0)
#     dt = 0.1
#     D = Density(Normal(0.0, 1.0))
#     S = Samplers.FHMC(; u0, tspan, α = 1.9, β = 0.01, γ = 1.0, 𝜋 = D)
#     sol = solve(S, EM(); dt)

#     x = sol[1, :]
#     x = TimeseriesTools.Timeseries(sol.t, x)
#     x = rectify(x, dims = 𝑡, tol = 1)

#     msd = msdist(x)

#     begin
#         f = Figure(size = (1000, 300))
#         ax = Axis(f[1, 1])
#         lines!(ax, x[1:20:20000])

#         lu = (-4 * std(x), +4 * std(x))
#         axx = Axis(f[1, 2], limits = (lu, nothing))
#         xs = range(lu..., length = 1000)
#         hist!(axx, x; bins = 100, normalization = :pdf)
#         lines!(axx, xs, D.(xs); color = :red, linewidth = 2)

#         axxx = Axis(f[1, 3], xscale = log10, yscale = log10)
#         lines!(axxx, msd[𝑡 = dt .. 1000], label = nothing)

#         # * Fit a tail index to msd
#         y = msd[𝑡 = dt .. 1]
#         ts = logrange(extrema(times(y))..., length = 1000)
#         y = y[𝑡 = Near(ts)]
#         taus = times(y)
#         α, β = [log10.(taus) ones(length(y))] \ log10.(y)

#         # * Plot line of fitted tail
#         lines!(axxx, taus, 10 .^ (α * log10.(taus) .+ β); color = :red, linewidth = 2,
#                label = "Fit: α = $α")
#         axislegend(axxx, position = :lt)
#         display(f)
#     end
# end

# begin
#     x = rand(Stable(2.0, 0.0), 10000) |> cumsum
#     x = Timeseries(range(dt, length = length(x), step = dt), x)
#     lines(x)
#     msd = mad(x)
#     f = Figure()
#     ax = Axis(f[1, 1], xscale = log10, yscale = log10)
#     lines!(ax, msd[𝑡 = dt .. 100], label = nothing)

#     # * Fit a tail index to msd
#     y = msd[𝑡 = dt .. 1]
#     ts = logrange(extrema(times(y))..., length = 1000)
#     y = y[𝑡 = Near(ts)]
#     taus = times(y)
#     α, β = [log10.(taus) ones(length(y))] \ log10.(y)

#     # * Plot line of fitted tail
#     lines!(ax, taus, 10 .^ (α * log10.(taus) .+ β); color = :red, linewidth = 2,
#            label = "Fit: α = $α")
#     axislegend(ax, position = :lt)
#     display(f)
# end

# begin
#     using TimeseriesTools
#     using CairoMakie
#     using StableDistributions
#     using Statistics
#     function mad(x::AbstractVector{T}) where {T <: Real}
#         n = length(x)
#         lags = 1:(n - 1)
#         mads = zeros(T, n - 1)
#         Threads.@threads for lag in lags
#             displacements = abs.(x[(1 + lag):n] .- x[1:(n - lag)])
#             mads[lag] = mean(displacements)
#         end
#         return mads
#     end
#     function mad(x::UnivariateRegular)
#         mads = mad(parent(x))
#         lags = range(step(x), length = length(mads), step = step(x))
#         return Timeseries(lags, mads)
#     end
#     function mssd(x::AbstractVector{T}) where {T <: Real}
#         n = length(x)
#         lags = 1:(n - 1)
#         msds = zeros(T, n - 1)
#         Threads.@threads for lag in lags
#             displacements = (x[(1 + lag):n] .- x[1:(n - lag)]) .^ 2
#             msds[lag] = mean(displacements)
#         end
#         return msds
#     end
#     function mssd(x::UnivariateRegular)
#         mssds = mssd(parent(x))
#         lags = range(step(x), length = length(mssds), step = step(x))
#         return Timeseries(lags, mssds)
#     end
# end

# begin
#     x = rand(Stable(1.3, 0.0), 10000) |> cumsum
#     x = Timeseries(range(dt, length = length(x), step = dt), x)
#     lines(x)
#     msd = mad(x)
#     f = Figure()
#     ax = Axis(f[1, 1], xscale = log10, yscale = log10)
#     lines!(ax, msd[𝑡 = dt .. 100], label = nothing)

#     # * Fit a tail index to msd
#     y = msd[𝑡 = dt .. 1]
#     ts = logrange(extrema(times(y))..., length = 1000)
#     y = y[𝑡 = Near(ts)]
#     taus = times(y)
#     α, β = [log10.(taus) ones(length(y))] \ log10.(y)

#     # * Plot line of fitted tail
#     lines!(ax, taus, 10 .^ (α * log10.(taus) .+ β); color = :red, linewidth = 2,
#            label = "Fit: α = $α")
#     axislegend(ax, position = :lt)
#     display(f)
# end

# begin
#     global estimator = mad
#     repeats = 100
#     dt = 0.01 # Dummy timestep
#     αs = 1.1:0.05:2.0

#     ms = progressmap(αs) do α
#         map(1:repeats) do _
#             x = rand(Stable(α, 0.0), 10000) |> cumsum
#             x = Timeseries(range(dt, length = length(x), step = dt), x)
#             m = estimator(x)

#             # * Fit a tail index to msd
#             y = m[𝑡 = dt .. 1]
#             ts = logrange(extrema(times(y))..., length = 1000)
#             y = y[𝑡 = Near(ts)]
#             taus = times(y)
#             β, _ = [log10.(taus) ones(length(y))] \ log10.(y)
#             return β
#         end
#     end
#     σs = std.(ms) ./ 2
#     ms = mean.(ms)

#     f = Figure()
#     ax = Axis(f[1, 1], xlabel = "α", ylabel = "β", title = "$estimator")
#     band!(ax, αs, ms .- σs, ms .+ σs; color = (:black, 0.3))
#     lines!(ax, αs, ms, color = :black)
#     if estimator === mad
#         # * Plot a line of beta = 1/alpha
#         lines!(ax, αs, 1 ./ αs, color = :red, linestyle = :dash, label = "β = 1/α")
#     end
#     display(f)
# end

@testitem "CaputoEM" begin
    include("./Solvers/CaputoEM.jl")
end
@testitem "LFSM" begin
    include("./lfsn.jl")
end
