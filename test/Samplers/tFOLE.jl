using FractionalNeuralSampling
using Distributions
using Random
using Statistics
using Test

begin # * Construction
    𝜋 = Density(Normal(0.0, 1.0))
    S = @test_nowarn tFOLE(; tspan = (0.0, 1.0), dt = 0.01, η = 0.5, β = 0.8, 𝜋)
    @test S.u0 == [0.0]
    @test parameters(S).η == 0.5
    @test parameters(S).β == 0.8
    @test S.kwargs[:alg] isa CaputoEM
    @test S.kwargs[:alg].β == 0.8 # The solver order tracks the order of the noise
end

begin # * A first-order sampler carries position only
    𝜋 = Density(Normal(0.0, 1.0))
    @test_throws ArgumentError tFOLE(;
        tspan = (0.0, 1.0), dt = 0.01, η = 0.5, β = 0.8, 𝜋,
        u0 = [0.0, 0.0]
    )
end

begin # * Solves with the algorithm supplied at construction
    𝜋 = Density(Normal(0.0, 1.0))
    S = tFOLE(; tspan = (0.0, 20.0), dt = 0.01, η = 2.0, β = 0.7, 𝜋, seed = 42)
    sol = @test_nowarn solve(S; dt = 0.01)
    x = only.(sol.u)
    @test all(isfinite, x)
    @test length(x) > 100
end

begin # * The seed fixes the noise, so two samplers with one seed give one path
    𝜋 = Density(Normal(0.0, 1.0))
    args = (; tspan = (0.0, 10.0), dt = 0.01, η = 1.0, β = 0.8, 𝜋)
    a = solve(tFOLE(; args..., seed = 7); dt = 0.01)
    b = solve(tFOLE(; args..., seed = 7); dt = 0.01)
    c = solve(tFOLE(; args..., seed = 8); dt = 0.01)
    @test a.u == b.u
    @test a.u != c.u
end

begin # * At β = 1 the sampler recovers its target, for any noise strength
    # `tfole_g!` is √η where `ole_g!` is √(2η), because `gen_fbm` draws α = 2 stable
    # increments carrying variance 2dt rather than the dt of a Wiener process. Getting this
    # wrong is invisible at a single η: the old coefficient η gave 𝜋^(1/η), which is the
    # target at η = 1 and nowhere else, so every strength below is needed
    𝜋 = Density(Normal(0.0, 1.0))
    for η in (0.5, 2.0, 4.0) # `local`, since these names are globals elsewhere in the file
        local x = only.(solve(tFOLE(; tspan = (0.0, 3000.0), dt = 0.01, η, β = 1.0, 𝜋,
            seed = 42); dt = 0.01).u)
        @test all(isfinite, x)
        @test mean(x) ≈ 0.0 atol = 0.15
        @test std(x) ≈ 1.0 atol = 0.12
    end
end

begin # * The target's own width is recovered, not just a standard normal's
    𝜋 = Density(Normal(0.0, 2.0))
    x = only.(solve(tFOLE(; tspan = (0.0, 3000.0), dt = 0.01, η = 2.0, β = 1.0, 𝜋,
        seed = 7); dt = 0.01).u)
    @test std(x) ≈ 2.0 atol = 0.25
end

begin # * A fractional order slows the approach but leaves the stationary law alone
    𝜋 = Density(Normal(0.0, 1.0))
    x = only.(solve(tFOLE(; tspan = (0.0, 3000.0), dt = 0.01, η = 2.0, β = 0.8, 𝜋,
        seed = 42); dt = 0.01).u)
    @test all(isfinite, x)
    @test mean(x) ≈ 0.0 atol = 0.2
    @test std(x) ≈ 1.0 atol = 0.15
end

begin # * remake keeps the sampler solvable
    𝜋 = Density(Normal(0.0, 1.0))
    S = tFOLE(; tspan = (0.0, 1.0), dt = 0.01, η = 1.0, β = 0.8, 𝜋)
    W = @test_nowarn remake(S, p = S.p)
    @test W.kwargs isa Base.Pairs
    @test_nowarn solve(W; dt = 0.01)
end
