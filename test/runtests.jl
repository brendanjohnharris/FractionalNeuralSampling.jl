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
    import FFTW
    FFTW.set_num_threads(1) # Threaded FFTW segfaults (JuliaMath/FFTW.jl#236)
end

@testitem "Density" setup = [Setup] begin
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

@testitem "Langevin sampler bias" setup = [Setup] begin
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
    @test mean(first.(sol.u)) ≈ 0.0 atol = 0.05
    @test std(first.(sol.u)) ≈ 1.0 atol = 0.05

    if false
        tspan = (0.0, 100.0)
        S = Langevin(;
            u0, tspan, β = 1.0, η = 1.0, 𝜋 = D,
            noise = WienerProcess(0.0, 0.0)
        )
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
    end
end

@testitem "Levy sampler bias" setup = [Setup] begin
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
    @test evaluate(KLDivergence(), D, x) < 0.2

end

@testitem "Space-fractional neural sampling bias" setup = [Setup] begin
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
    @test evaluate(KLDivergence(), D, x) < 0.05
end

@testitem "2D sFNS" setup = [Setup] begin
    u0 = ArrayPartition([0.0, 0.0], [0.0, 0.0])
    tspan = (0.0, 50.0)
    dt = 0.05
    D = Density(MixtureModel(MvNormal, [([-2, -2], I(2)), ([2, 2], I(2))]))
    d = (-7 .. 7, -7 .. 7)
    boundaries = PeriodicBox(-5 .. 5, -5 .. 5)
    S = sFNS(;
        u0, tspan, α = 1.5, β = 0.05, γ = 0.5, 𝜋 = D, domain = d, boundaries,
        approx_n_modes = 1000
    )

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

@testitem "Recursive Arrays" setup = [Setup] begin
    # So for recursive arrays, we can use diagonal noise by setting the noise_rate prototype
    # to similar(u0)
    u0 = ArrayPartition([0.0], [0.0])
    # At tspan = 500 the KL estimate ranged over 0.005--0.27 across runs, straddling the
    # 0.1 threshold below, so this test failed about half the time (and kept CI red on
    # whichever platform lost the toss). Four times the trajectory brings the spread to
    # 0.004--0.024; the seed then makes it reproducible for a given Julia version.
    tspan = (0.0, 2000.0)
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

    Random.seed!(42)
    sol = solve(S, EM(); dt)
    x = first.(sol.u)
    x = x[abs.(x) .< 6]
    @test evaluate(KLDivergence(), D, x) < 0.1
end

@testitem "Autodiff" setup = [Setup] begin
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
@testitem "Univariate DistributionDensity" setup = [Setup] begin
    d = Normal(0.0, 0.5)
    D = @test_nowarn Density(d)
    @test D isa Densities.AbstractUnivariateDensity
    @test Densities.doautodiff(D) == false
    @test D(0.0) == 2 / sqrt(2π)
    @test D([0.0]) == 2 / sqrt(2π)
    @test LogDensityProblems.dimension(D) == 1
    @test all(
        map(LogDensityProblems.logdensity(D), -1:0.1:1) .≈ log.(D.(-1:0.1:1)) .≈
            logpdf(distribution(D), -1:0.1:1)
    )
    @inferred LogDensityProblems.logdensity(D, 0.0)
    @inferred LogDensityProblems.logdensity(D, 0)
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

    D = @test_nowarn Density(Normal(0.0f0, 0.5f0))
    @test LogDensityProblems.logdensity(D, 0.0f0) isa Float32
    @test Densities.gradlogdensity(D, 0.0f0) isa Float32
end

@testitem "Multivariate DistributionDensity" setup = [Setup] begin
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
    @test gradlogdensity(D)(collect(ps)) == gradlogpdf.([d], ps) # Collection of positions
    @test gradlogdensity(D)([view(p, 1:N)]) == [gradlogpdf(d, p)] # Views, as in solvers

    # * Ad
    D = @test_nowarn Density(MvNormal(μs, Σ), true)
    @test Densities.doautodiff(D) == true
    @test LogDensityProblems.logdensity(D, p) isa Float64
    @test logdensity(D)(p) == logpdf(d, p)
    @test logdensity(D)(ps) == logpdf.([d], ps)
    @test gradlogdensity(D)(p) ≈ gradlogpdf(d, p)
end

@testitem "Mixture DistributionDensity" setup = [Setup] begin
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

@testitem "AdDistributionDensity" setup = [Setup] begin
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

@testitem "Overdamped Langevin Sampler" setup = [Setup] begin
    u0 = [0.0]
    tspan = (0.0, 100.0)
    D = Density(Normal(0, 10.0))
    S = OLE(; u0, tspan, η = 1, 𝜋 = D)

    sol = @test_nowarn solve(S, EM(); dt = 0.001, saveat = 0.01)
    x = first.(sol.u)

    # * Try setting parameters
    s = S(η = 0.1)
    @test s.p[1][:η] == 0.1

    s = S(; η = 10.0, tspan = 1000.0, 𝜋 = D)
    @test s.p[1][:η] == 10.0
    @test s.p[2] == D
    @test s.tspan == 1000.0
end

@testitem "Box boundaries" setup = [Setup] begin
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
    @test minimum(x) ≥ -5 - 2.0e-2
    @test maximum(x) ≤ 5 + 2.0e-2
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
    gg = fit(Laplace, x)
    @test gg.μ ≈ 0.0f0 atol = 5.0e-2
    @test gg.θ ≈ 1.0f0 atol = 1.0e-1
end


@testitem "CaputoEM" begin
    include("./Solvers/CaputoEM.jl")
end
@testitem "MultiCaputoEM" begin
    include("./Solvers/MultiCaputoEM.jl")
end
@testitem "PositionalCaputoEM" begin
    include("./Solvers/PositionalCaputoEM.jl")
end
@testitem "LFSM" begin
    include("./lfsn.jl")
end
@testitem "tFOLE" begin
    include("./Samplers/tFOLE.jl")
end
@testitem "sFOLE" begin
    include("./Samplers/sFOLE.jl")
end
@testitem "bFOLE" begin
    include("./Samplers/bFOLE.jl")
end
@testitem "bFNS" begin
    include("./Samplers/bFNS.jl")
end
@testitem "FHMC" begin
    include("./Samplers/FHMC.jl")
end
@testitem "Adaptive samplers" begin
    include("./Samplers/Adaptive.jl")
end

@testitem "Spectral transforms run serially" setup = [Setup] begin
    # FFTW segfaults on the in-place plans used here when multithreaded
    # (JuliaMath/FFTW.jl#236), so `serial_fftw` drops it to one thread per transform and
    # restores the caller's count. Without it these two calls take the process down
    n = FFTW.get_num_threads()
    try
        FFTW.set_num_threads(2)
        @test FFTW.get_num_threads() == 2
        @test_nowarn lfsn(1000, 1.5, 0.6)
        @test_nowarn sFOLE(;
            tspan = (0.0, 1.0), η = 0.5, α = 1.5,
            𝜋 = Density(Normal(0, 1)), domain = -10 .. 10
        )
        @test FFTW.get_num_threads() == 2 # Restored, not clamped
    finally
        FFTW.set_num_threads(n)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Regression tests for the v0.3.0 cleanup
# ─────────────────────────────────────────────────────────────────────────────

@testitem "Sampler defaults" setup = [Setup] begin
    # Every sampler with a default target used to throw a MethodError, since
    # `default_density` already returns a `Density` and was being wrapped again
    tspan = (0.0, 1.0)
    @test_nowarn FNS(; tspan, α = 1.5, β = 0.1, γ = 0.5)
    @test_nowarn FHMC(; tspan, α = 1.5, β = 0.1, γ = 0.5)
    @test_nowarn Langevin(; tspan, β = 1.0, η = 1.0)
    @test_nowarn OLE(; tspan, η = 1.0)

    # ... and the default target has the dimension the sampler's order implies
    @test dimension(Density(FNS(; tspan, α = 1.5, β = 0.1, γ = 0.5))) == 1
    S = FNS(; u0 = zeros(4), tspan, α = 1.5, β = 0.1, γ = 0.5)
    @test dimension(Density(S)) == 2
    @test dimension(Density(OLE(; u0 = [0.0], tspan, η = 1.0))) == 1

    # FHMC defaulted u0 to a 1x2 matrix (a missing comma), and dropped `boundaries`
    # unless they were already a callback
    @test FHMC(; tspan, α = 1.5, β = 0.1, γ = 0.5).u0 isa AbstractVector
    @test_nowarn FHMC(;
        tspan, α = 1.5, β = 0.1, γ = 0.5,
        boundaries = ReflectingBox(-5 .. 5)
    )

    # Every sampler supplies a default algorithm, so `solve(S)` works
    for S in (
            FNS(; tspan, α = 1.5, β = 0.1, γ = 0.5),
            FHMC(; tspan, α = 1.5, β = 0.1, γ = 0.5),
            Langevin(; tspan, β = 1.0, η = 1.0),
            OLE(; tspan, η = 1.0),
        )
        @test_nowarn solve(S; dt = 0.01)
    end
end

@testitem "Fractional solvers are fixed-step" setup = [Setup] begin
    import FractionalNeuralSampling.Solvers: isadaptive
    for alg in (CaputoEM(0.6, 10), MultiCaputoEM([0.6], 10), PositionalCaputoEM(0.6, 10))
        @test isadaptive(alg) == false
    end
    # The L1 weights assume a uniform grid, so asking for adaptivity must error
    S = OLE(; u0 = [0.0], tspan = (0.0, 1.0), η = 1.0, 𝜋 = Density(Normal(0, 1)))
    @test_throws "Fixed timestep" solve(S, CaputoEM(0.6, 10); dt = 0.01, adaptive = true)
end

@testitem "Boundaries" setup = [Setup] begin
    import FractionalNeuralSampling.Boundaries: isoutside, boxdist, _corners

    # `isoutside` is the allocation-free form of the old `boxdist(...) < 0`
    box = PeriodicBox(-2 .. 3, -1 .. 1)
    mn, mx = _corners(box)
    for p in ([0.0, 0.0], [-2.0, -1.0], [3.0, 1.0], [-2.1, 0.0], [0.0, 1.1], [9.0, 9.0])
        @test isoutside(p, mn, mx) == (boxdist(p, mn, mx) < 0)
    end
    @test [0.0, 0.0] ∈ box
    @test [9.0, 9.0] ∉ box

    # Corner fields are concrete, so the per-step condition is type stable
    for b in (ReflectingBox(-5 .. 5), PeriodicBox(-1 .. 1), ReentrantBox(-1.0 => 1.0))
        @test isconcretetype(fieldtype(typeof(b), 1))
    end
    @test ReflectingBox(-5 .. 5.0).min_corner === (-5.0,) # promoted, not Any

    # `domain` and `in` read through `_corners`, so they work for ReentrantBox too,
    # which has no min_corner/max_corner fields
    @test FractionalNeuralSampling.domain(PeriodicBox(-2 .. 3)) == [-2 .. 3]
    @test FractionalNeuralSampling.domain(ReentrantBox(-1.0 => 1.0)) == [-1.0 .. 1.0]
end

@testitem "Window" setup = [Setup] begin
    import FractionalNeuralSampling: Window, roll!

    w = Window([1.0, 2.0, 3.0])
    @test collect(w) == [1.0, 2.0, 3.0] # index 1 oldest, end newest after pushes
    push!(w, 4.0)
    @test w[end] == 4.0

    # Elements must be distinct objects: `fill` would alias one array into every slot,
    # which the in-place history roll would then corrupt
    v = Window([0.0], 3)
    @test length(unique(objectid.(v.data))) == 3

    # `roll!` returns the dropped slot for overwriting, matching `push!`
    a = Window([0.0], 3)
    b = Window([0.0], 3)
    for i in 1:5
        push!(a, [float(i)])
        roll!(b) .= [float(i)]
    end
    @test collect(a) == collect(b)
end

@testitem "lfsn robustness" setup = [Setup] begin
    import FFTW
    FFTW.set_num_threads(1) # Threaded FFTW segfaults (JuliaMath/FFTW.jl#236)

    # The FFT padding used to be sized before `m` was rounded up to even, so a small
    # odd `m` left the tail of the result uninitialised
    for m in (1, 2, 3, 127, 128)
        x = lfsn(500, 1.5, 0.6; m)
        @test length(x) == 500
        @test all(isfinite, x)
    end
    @test length(lfsn(1000, 1.5, 0.6; M = 10)) == 1000

    # dt scales the increments by dt^H
    Random.seed!(1)
    a = lfsn(2000, 2.0, 0.5; dt = 1)
    Random.seed!(1)
    b = lfsn(2000, 2.0, 0.5; dt = 0.25)
    @test b ≈ a .* 0.25^0.5
end

@testitem "Noise processes" setup = [Setup] begin
    import FractionalNeuralSampling.NoiseProcesses: LevyNoise

    # Out-of-place Lévy noise has no sampling method, so it must fail at construction
    # rather than at solve time
    @test_throws ArgumentError LevyProcess(1.5)
    @test_nowarn LevyProcess!(1.5; W0 = [0.0])

    @test fieldtype(LevyNoise{true, Float64}, :ND) === Int # was ::Integer, an abstract field
    L = LevyNoise{true}(1.5, 0.0, 1.0, 0.0, 2)
    x = zeros(4)
    Random.seed!(1)
    L(Random.default_rng(), x)
    @test all(!iszero, x)
    @test_throws Exception LevyNoise{true}(2.5) # α outside (0, 2]
end

@testitem "Density constructors" setup = [Setup] begin
    # For a distribution the type parameter is `doAd`, not the dimension; passing a
    # dimension used to build a density that matched no gradient method
    @test_throws ArgumentError Density{1}(Normal(0.0, 1.0))
    @test Densities.doautodiff(Density{true}(Normal(0.0, 1.0))) == true
    @test Densities.doautodiff(Density(Normal(0.0, 1.0))) == false # analytic gradlogpdf

    # `Density{N}(f)` still means the dimension for a plain function
    f(x) = exp(-only(x)^2 / 2) / sqrt(2π)
    @test dimension(Density{1}(f)) == 1
end

@testitem "Noise generators accept a tuple tspan" setup = [Setup] begin
    import FFTW
    FFTW.set_num_threads(1)
    # These divided `tspan` by `dt`, which fails for the tuple form the README documents
    @test_nowarn tFOLE(;
        tspan = (0.0, 1.0), dt = 0.01, η = 0.1, β = 0.8,
        𝜋 = Density(Normal(0, 1))
    )
    @test_nowarn tFOLE(;
        tspan = 1.0, dt = 0.01, η = 0.1, β = 0.8,
        𝜋 = Density(Normal(0, 1))
    )
end

@testitem "Per-step allocations" setup = [Setup] begin
    using StochasticDiffEqCore
    import FractionalNeuralSampling.Boundaries: getcondition

    # The boundary condition runs every step. It used to materialise the per-axis edge
    # distances (272 bytes a call); it now tests for a crossing in place. Taking `c` and
    # `u` as arguments keeps the measurement itself free of closure boxing.
    function condition_calls(c, u, n)
        for _ in 1:n
            c(u, 0.0, nothing)
        end
        return nothing
    end
    for box in (PeriodicBox(-3 .. 3), PeriodicBox(-3 .. 3, -3 .. 3), ReflectingBox(-3 .. 3))
        c = getcondition(box)
        u = zeros(2 * length(FractionalNeuralSampling.domain(box)))
        condition_calls(c, u, 1) # compile
        @test (@allocated condition_calls(c, u, 100)) == 0
    end

    # A step must not allocate its history element or re-prepare its gradient.
    # Measured on this machine: OLE/EM 5440 -> 2240, OLE/CaputoEM 7040 -> 2560.
    function steps(S, alg; dt = 0.01, n = 20)
        int = init(S, alg; dt)
        StochasticDiffEqCore.perform_step!(int, int.cache) # compile
        return @allocated for _ in 1:n
            StochasticDiffEqCore.perform_step!(int, int.cache)
        end
    end

    𝜋 = Density(Normal(0.0, 1.0))
    u0 = [0.0]
    tspan = (0.0, 10.0)
    @test steps(OLE(; u0, tspan, η = 1.0, 𝜋), EM()) < 4000
    @test steps(OLE(; u0, tspan, η = 1.0, 𝜋), CaputoEM(0.6, 50)) < 4500
end

@testitem "Sampler kwargs survive remake" setup = [Setup] begin
    # `DiffEqBase` reads `values(prob.kwargs)` and needs a `NamedTuple` back, so the field
    # has to stay a `Base.Pairs`. Since DiffEqBase v7.21 `_erase_problem_callback_types`
    # rebuilds it as a plain `NamedTuple`, whose `values` is a `Tuple`, and every `solve`
    # then fails in `merge_problem_kwargs`.
    S = OLE(; u0 = [0.0], tspan = (0.0, 1.0), η = 1.0, 𝜋 = Density(Normal(0, 1)))
    @test S.kwargs isa Base.Pairs
    @test values(S.kwargs) isa NamedTuple

    W = remake(S, p = S.p)
    @test W.kwargs isa Base.Pairs

    # the path the solver itself takes: `_erase_problem_callback_types` rebuilds the
    # field from a NamedTuple, via this positional constructor
    R = Sampler{true}(
        S.f, S.g, S.u0, S.tspan, S.p, S.noise,
        (; callback = S.kwargs[:callback], alg = S.kwargs[:alg]),
        S.noise_rate_prototype, S.seed
    )
    @test R.kwargs isa Base.Pairs
    @test values(R.kwargs) isa NamedTuple
    @test_nowarn solve(R, EM(); dt = 0.01)
end

@testitem "Fractional Laplacian operator" setup = [Setup] begin
    using ApproxFun
    import FractionalNeuralSampling: Power
    import FractionalNeuralSampling.Samplers: space_fractional_deriv

    # At α = 2 the fractional Laplacian is the identity, so the drift reduces to ∇𝜋
    S, D, 𝒟 = space_fractional_deriv(Val(1); α = 2.0, domain = -5.0 .. 5.0)
    𝜋(x) = exp(-x^2 / 2) / sqrt(2π)
    𝜋s = Fun(𝜋, S, 500)
    xs = -3:0.5:3
    @test (𝒟 * 𝜋s).(xs) ≈ 𝜋s.(xs) atol = 1.0e-6
    @test (D * 𝒟 * 𝜋s).(xs) ≈ (D * 𝜋s).(xs) atol = 1.0e-6
    # ∇𝜋 of a standard normal is -x 𝜋(x)
    @test (D * 𝜋s).(xs) ≈ -xs .* 𝜋.(xs) atol = 1.0e-6

    # Power is diagonal and leaves the zero mode alone for negative powers
    Δ = FractionalNeuralSampling.Samplers.maybeLaplacian(S)
    P = Power(-Δ, -0.5)
    @test ApproxFun.bandwidths(P) == (0, 0)
    @test P[1, 2] == 0
end
