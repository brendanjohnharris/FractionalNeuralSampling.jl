using FractionalNeuralSampling
using Distributions
using IntervalSets
using Random
using Statistics
using Test

begin # * Construction
    𝜋 = Density(Normal(0.0, 1.0))
    S = @test_nowarn bFNS(;
        tspan = (0.0, 10.0), dt = 0.01, α = 1.5, β = 0.8, γ = 0.5, η = 0.1, 𝜋,
        domain = -10 .. 10
    )
    @test S.u0 == [0.0, 0.0] # Second order: position and momentum
    @test parameters(S).α == 1.5
    @test parameters(S).β == 0.8
    @test S.kwargs[:alg] isa PositionalCaputoEM
    @test S.kwargs[:alg].β1 == 0.8 # Only the position is fractional in time
end

begin # * A second-order sampler needs 2d state for a d-dimensional target
    𝜋 = Density(Normal(0.0, 1.0))
    @test_throws ArgumentError bFNS(;
        tspan = (0.0, 10.0), dt = 0.01, α = 1.5, β = 0.8, γ = 0.5, η = 0.1, 𝜋,
        domain = -10 .. 10, u0 = [0.0]
    )
end

begin # * A scalar tspan is accepted, having once been divided by dt
    𝜋 = Density(Normal(0.0, 1.0))
    S = @test_nowarn bFNS(;
        tspan = 10.0, dt = 0.01, α = 1.5, β = 0.8, γ = 0.5, η = 0.1, 𝜋,
        domain = -10 .. 10
    )
    @test S.tspan == (0.0, 10.0)
end

begin # * The momentum has neither drift nor noise when γ = 0, so it never moves
    # `bfns_f!` gives dv = γb, `bfns_g!` gives dv = 0, and `gen_lfsm_fns` zeroes the
    # momentum column of the noise grid
    𝜋 = Density(Normal(0.0, 1.0))
    S = bFNS(;
        tspan = (0.0, 20.0), dt = 0.01, α = 1.5, β = 0.8, γ = 0.0, η = 0.5, 𝜋,
        domain = -10 .. 10, seed = 42
    )
    sol = solve(S; dt = 0.01)
    v = last.(sol.u)
    x = first.(sol.u)
    @test all(iszero, v)
    @test all(isfinite, x)
    @test !all(iszero, x) # ... while the position does move, so the test is not vacuous
end

begin # * Solves with the algorithm supplied at construction
    𝜋 = Density(MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)]))
    S = bFNS(;
        tspan = (0.0, 100.0), dt = 0.01, α = 1.5, β = 0.8, γ = 0.5, η = 0.1, 𝜋,
        domain = -15 .. 15, boundaries = PeriodicBox(-7 .. 7), seed = 42
    )
    sol = @test_nowarn solve(S; dt = 0.01)
    x = first.(sol.u)
    @test all(isfinite, x)
    @test length(x) > 100
end

begin # * The seed fixes the noise, so two samplers with one seed give one path
    𝜋 = Density(Normal(0.0, 1.0))
    args = (; tspan = (0.0, 10.0), dt = 0.01, α = 1.5, β = 0.8, γ = 0.5, η = 0.1, 𝜋,
        domain = -10 .. 10)
    a = solve(bFNS(; args..., seed = 7); dt = 0.01)
    b = solve(bFNS(; args..., seed = 7); dt = 0.01)
    c = solve(bFNS(; args..., seed = 8); dt = 0.01)
    @test a.u == b.u
    @test a.u != c.u
end

begin # * remake keeps the sampler solvable
    𝜋 = Density(Normal(0.0, 1.0))
    S = bFNS(;
        tspan = (0.0, 10.0), dt = 0.01, α = 1.5, β = 0.8, γ = 0.5, η = 0.1, 𝜋,
        domain = -10 .. 10
    )
    W = @test_nowarn remake(S, p = S.p)
    @test W.kwargs isa Base.Pairs
    @test_nowarn solve(W; dt = 0.01)
end
