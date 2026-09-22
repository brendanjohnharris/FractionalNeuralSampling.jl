using FractionalNeuralSampling
using Distributions
using Distances
using StatsBase
using IntervalSets
using Random
using Statistics
using Test

begin # * Construction
    𝜋 = Density(Normal(0.0, 1.0))
    S = @test_nowarn sFOLE(; tspan = (0.0, 1.0), η = 0.5, α = 1.5, 𝜋, domain = -10 .. 10)
    @test S.u0 == [0.0]
    @test parameters(S).η == 0.5
    @test parameters(S).α == 1.5
    @test S.kwargs[:alg] isa EM
end

begin # * A first-order sampler carries position only
    𝜋 = Density(Normal(0.0, 1.0))
    @test_throws ArgumentError sFOLE(;
        tspan = (0.0, 1.0), η = 0.5, α = 1.5, 𝜋,
        domain = -10 .. 10, u0 = [0.0, 0.0]
    )
end

begin # * At α = 2 the fractional Laplacian is the identity, so the drift is that of `OLE`
    # ∇𝒟𝜋/𝜋 reduces to ∇𝜋/𝜋 = ∇log𝜋; λ regularises the quotient, so it is off here
    𝜋 = Density(Normal(0.0, 1.0))
    η = 0.7
    S = sFOLE(;
        tspan = (0.0, 1.0), η, α = 2.0, 𝜋, domain = -10 .. 10,
        λ = 0.0, approx_n_modes = 2000
    )
    O = OLE(; tspan = (0.0, 1.0), η, 𝜋)

    du_s, du_o = zeros(1), zeros(1)
    for x in -2.0:0.5:2.0
        S.f.f(du_s, [x], S.p, 0.0)
        O.f.f(du_o, [x], O.p, 0.0)
        @test only(du_s) ≈ only(du_o) atol = 1.0e-6
    end
end

begin # * Solving a bimodal target recovers it
    Random.seed!(42)
    𝜋 = Density(MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)]))
    S = sFOLE(;
        tspan = (0.0, 2000.0), η = 0.5, α = 1.5, 𝜋, domain = -15 .. 15,
        boundaries = PeriodicBox(-7 .. 7)
    )
    sol = @test_nowarn solve(S; dt = 0.01)
    x = only.(sol.u)
    @test all(isfinite, x)
    x = x[abs.(x) .< 6]
    @test 0.3 < mean(x .< 0) < 0.7 # Both modes are visited, not just the nearer one
    @test evaluate(KLDivergence(), 𝜋, x) < 0.2
end

begin # * remake keeps the sampler solvable
    𝜋 = Density(Normal(0.0, 1.0))
    S = sFOLE(; tspan = (0.0, 1.0), η = 0.5, α = 1.5, 𝜋, domain = -10 .. 10)
    W = @test_nowarn remake(S, p = S.p)
    @test W.kwargs isa Base.Pairs
    @test_nowarn solve(W; dt = 0.01)
end
