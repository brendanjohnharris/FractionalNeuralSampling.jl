using FractionalNeuralSampling
using Distributions
using IntervalSets
using Random
using Statistics
using Test

begin # * Construction
    𝜋 = Density(Normal(0.0, 1.0))
    S = @test_nowarn bFOLE(;
        tspan = (0.0, 1.0), dt = 0.01, η = 0.5, α = 1.5, β = 0.8, 𝜋,
        domain = -10 .. 10
    )
    @test S.u0 == [0.0]
    @test parameters(S).α == 1.5
    @test parameters(S).β == 0.8
    @test S.kwargs[:alg] isa CaputoEM
    @test S.kwargs[:alg].β == 0.8 # The solver order tracks the order of the noise
end

begin # * A first-order sampler carries position only
    𝜋 = Density(Normal(0.0, 1.0))
    @test_throws ArgumentError bFOLE(;
        tspan = (0.0, 1.0), dt = 0.01, η = 0.5, α = 1.5, β = 0.8, 𝜋,
        domain = -10 .. 10, u0 = [0.0, 0.0]
    )
end

begin # * The space-fractional drift is shared with `sFOLE`; only noise and solver differ
    𝜋 = Density(Normal(0.0, 1.0))
    args = (; η = 0.7, α = 1.5, 𝜋, domain = -10 .. 10, λ = 1.0e-4, approx_n_modes = 1000)
    B = bFOLE(; tspan = (0.0, 1.0), dt = 0.01, β = 0.8, args...)
    S = sFOLE(; tspan = (0.0, 1.0), args...)

    du_b, du_s = zeros(1), zeros(1)
    for x in -2.0:0.5:2.0
        B.f.f(du_b, [x], B.p, 0.0)
        S.f.f(du_s, [x], S.p, 0.0)
        @test only(du_b) ≈ only(du_s)
    end
end

begin # * Solves with the algorithm supplied at construction
    𝜋 = Density(MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)]))
    S = bFOLE(;
        tspan = (0.0, 100.0), dt = 0.01, η = 0.5, α = 1.5, β = 0.8, 𝜋,
        domain = -15 .. 15, boundaries = PeriodicBox(-7 .. 7), seed = 42
    )
    sol = @test_nowarn solve(S; dt = 0.01)
    x = only.(sol.u)
    @test all(isfinite, x)
    @test length(x) > 100
end

begin # * The seed fixes the noise, so two samplers with one seed give one path
    𝜋 = Density(Normal(0.0, 1.0))
    args = (; tspan = (0.0, 10.0), dt = 0.01, η = 0.5, α = 1.5, β = 0.8, 𝜋,
        domain = -10 .. 10)
    a = solve(bFOLE(; args..., seed = 7); dt = 0.01)
    b = solve(bFOLE(; args..., seed = 7); dt = 0.01)
    c = solve(bFOLE(; args..., seed = 8); dt = 0.01)
    @test a.u == b.u
    @test a.u != c.u
end

begin # * remake keeps the sampler solvable
    𝜋 = Density(Normal(0.0, 1.0))
    S = bFOLE(;
        tspan = (0.0, 1.0), dt = 0.01, η = 0.5, α = 1.5, β = 0.8, 𝜋,
        domain = -10 .. 10
    )
    W = @test_nowarn remake(S, p = S.p)
    @test W.kwargs isa Base.Pairs
    @test_nowarn solve(W; dt = 0.01)
end
