using FractionalNeuralSampling
using Distributions
using IntervalSets
using Random
using Statistics
using Test

kernel(x) = exp(-only(x)^2 / 2)

begin # * The adaptation kernel needs a domain, which only `boundaries` supplies
    𝜋 = Density(Normal(0.0, 1.0))
    @test_throws ArgumentError AdaptiveWalkSampler(
        kernel, 64;
        tspan = (0.0, 1.0), γ = 0.5, τ_r = 1.0, τ_d = 10.0, 𝜋
    )
    @test_throws ArgumentError AdaptiveLevySampler(
        kernel, 64;
        tspan = (0.0, 1.0), α = 1.5, γ = 0.5, τ_r = 1.0, τ_d = 10.0, 𝜋
    )
end

begin # * Construction
    𝜋 = Density(Normal(0.0, 1.0))
    boundaries = PeriodicBox(-5 .. 5)
    S = @test_nowarn AdaptiveWalkSampler(
        kernel, 64;
        tspan = (0.0, 1.0), γ = 0.5, τ_r = 1.0, τ_d = 10.0, 𝜋, boundaries
    )
    @test S.u0 == [0.0]
    @test parameters(S).γ == 0.5
    @test parameters(S).dim == 1
    @test all(iszero, parameters(S).a_K) # The kernel starts empty
end

begin # * The kernel accumulates while the sampler runs, and only when τ_r is finite
    𝜋 = Density(Normal(0.0, 1.0))
    boundaries = PeriodicBox(-5 .. 5)
    mk(τ_r) = AdaptiveWalkSampler(
        kernel, 64;
        tspan = (0.0, 5.0), γ = 0.5, τ_r, τ_d = 100.0, 𝜋, boundaries
    )

    S = mk(Inf) # Infinite deposition time: nothing is ever laid down
    solve(S, EM(); dt = 0.01)
    @test all(iszero, parameters(S).a_K)

    S = mk(1.0)
    solve(S, EM(); dt = 0.01)
    @test any(!iszero, parameters(S).a_K)
    @test all(isfinite, parameters(S).a_K)
end

begin # * An empty kernel leaves the adaptive walk equal to the overdamped Langevin
    𝜋 = Density(Normal(0.0, 1.0))
    γ = 0.5
    A = AdaptiveWalkSampler(
        kernel, 64;
        tspan = (0.0, 1.0), γ, τ_r = Inf, τ_d = Inf, 𝜋,
        boundaries = PeriodicBox(-5 .. 5)
    )
    O = OLE(; tspan = (0.0, 1.0), η = γ, 𝜋)

    du_a, du_o = zeros(1), zeros(1)
    for x in -2.0:0.5:2.0
        A.f.f(du_a, [x], A.p, 0.0)
        O.f.f(du_o, [x], O.p, 0.0)
        @test only(du_a) ≈ only(du_o) # Drift: -γ∇V = γ∇log𝜋
        A.g(du_a, [x], A.p, 0.0)
        O.g(du_o, [x], O.p, 0.0)
        @test only(du_a) ≈ only(du_o) # Diffusion: √2γ
    end
end

begin # * At α = 2 the Lévy prefactor Γ(α-1)/Γ(α/2)² is one, leaving the same drift
    𝜋 = Density(Normal(0.0, 1.0))
    γ = 0.5
    A = AdaptiveLevySampler(
        kernel, 64;
        tspan = (0.0, 1.0), α = 2.0, γ, τ_r = Inf, τ_d = Inf, 𝜋,
        boundaries = PeriodicBox(-5 .. 5)
    )
    O = OLE(; tspan = (0.0, 1.0), η = γ, 𝜋)

    du_a, du_o = zeros(1), zeros(1)
    for x in -2.0:0.5:2.0
        A.f.f(du_a, [x], A.p, 0.0)
        O.f.f(du_o, [x], O.p, 0.0)
        @test only(du_a) ≈ only(du_o)
    end
end

begin # * Both samplers solve, and the box keeps them confined
    𝜋 = Density(Normal(0.0, 1.0))
    boundaries = PeriodicBox(-5 .. 5)
    args = (; tspan = (0.0, 50.0), γ = 0.5, τ_r = 1.0, τ_d = 100.0, 𝜋, boundaries)

    x = only.(solve(AdaptiveWalkSampler(kernel, 64; args...), EM(); dt = 0.01).u)
    @test all(isfinite, x)
    @test minimum(x) ≥ -5 - 0.5 # Gaussian steps overshoot the wall by very little
    @test maximum(x) ≤ 5 + 0.5

    x = only.(solve(AdaptiveLevySampler(kernel, 64; α = 1.5, args...), EM(); dt = 0.01).u)
    @test all(isfinite, x)
    # A Lévy jump is wrapped on the step after it lands, so a sample can sit outside the
    # box. The overshoot cannot accumulate, so the walker stays within one box width
    @test maximum(abs, x) < 15
    @test mean(abs.(x) .<= 5) > 0.8
end
