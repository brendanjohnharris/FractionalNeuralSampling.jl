using FractionalNeuralSampling
using Distributions
using IntervalSets
using Random
using SpecialFunctions
import DiffEqNoiseProcess # NoiseGrid; not reexported by the package
using Statistics
using Test

begin # * Construction
    𝜋 = Density(Normal(0.0, 1.0))
    S = @test_nowarn FHMC(; tspan = (0.0, 1.0), α = 1.5, β = 0.1, γ = 0.5, 𝜋)
    @test S.u0 == [0.0, 0.0] # Second order: position and momentum
    @test parameters(S).α == 1.5
    @test parameters(S).β == 0.1
    @test parameters(S).γ == 0.5
    @test S.kwargs[:alg] isa EM
    @test FractionalHamiltonianMonteCarlo === FHMC
end

begin # * A second-order sampler needs 2d state for a d-dimensional target
    𝜋 = Density(Normal(0.0, 1.0))
    @test_throws ArgumentError FHMC(;
        tspan = (0.0, 1.0), α = 1.5, β = 0.1, γ = 0.5, 𝜋,
        u0 = [0.0]
    )
end

begin # * Without damping or noise the walker is ballistic, at the Lévy-corrected speed
    # dx = c_α β v and dv = -c_α β ∇V - γv, so a flat potential and γ = 0 leave v fixed
    𝜋 = PotentialDensity{1}(_ -> 0.0)
    α, β, T = 1.5, 0.7, 2.0
    c_α = gamma(α + 1) / gamma(α / 2 + 1)^2
    S = FHMC(; tspan = (0.0, T), α, β, γ = 0.0, 𝜋, u0 = [0.0, 1.0])
    sol = solve(S; dt = 1.0e-4)
    x, v = sol.u[end]
    @test v ≈ 1.0 atol = 1.0e-10 # No damping and no noise
    @test x ≈ c_α * β * T rtol = 1.0e-6
end

begin # * Damping alone relaxes the momentum exponentially
    # `fractional_hmc_g!` puts γ^(1/α) on the momentum, so damping cannot be isolated by
    # parameters alone; a zero noise grid leaves dv = -γv, whose solution is v₀exp(-γt)
    𝜋 = PotentialDensity{1}(_ -> 0.0)
    γ, T = 1.5, 3.0
    ts = range(0.0, T, length = 101)
    W = DiffEqNoiseProcess.NoiseGrid(ts, [zeros(2) for _ in ts])
    S = FHMC(; tspan = (0.0, T), α = 2.0, β = 0.0, γ, 𝜋, u0 = [0.0, 1.0], noise = W)
    x, v = solve(S; dt = 1.0e-4).u[end]
    @test x == 0.0 # β = 0 decouples the position
    @test v ≈ exp(-γ * T) rtol = 1.0e-3
end

begin # * Boundaries are honoured whether given as a box or as a callback
    𝜋 = Density(Normal(0.0, 1.0))
    box = ReflectingBox(-2 .. 2)
    for boundaries in (box, box())
        local S = FHMC(;
            tspan = (0.0, 200.0), α = 1.8, β = 0.5, γ = 0.5, 𝜋, boundaries,
            u0 = [0.0, 0.0]
        )
        local x = first.(solve(S; dt = 0.001).u) # A global `x` exists in this file
        @test minimum(x) ≥ -2 - 0.05
        @test maximum(x) ≤ 2 + 0.05
    end
end

begin # * At α = 2 the sampler recovers a Gaussian target
    Random.seed!(42)
    𝜋 = Density(Normal(0.0, 1.0))
    S = FHMC(;
        tspan = (0.0, 2000.0), α = 2.0, β = 1.0, γ = 1.0, 𝜋,
        boundaries = ReflectingBox(-5 .. 5) # Truncates the target by ~1e-6
    )
    x = first.(solve(S; dt = 0.01).u)
    @test all(isfinite, x)
    @test mean(x) ≈ 0.0 atol = 0.2
    @test std(x) ≈ 1.0 atol = 0.15
    @test mean(abs.(x) .< 1) ≈ 0.68 atol = 0.06
end

begin # * remake keeps the sampler solvable
    𝜋 = Density(Normal(0.0, 1.0))
    S = FHMC(; tspan = (0.0, 1.0), α = 1.5, β = 0.1, γ = 0.5, 𝜋)
    W = @test_nowarn remake(S, p = S.p)
    @test W.kwargs isa Base.Pairs
    @test_nowarn solve(W; dt = 0.01)

    S2 = S(; γ = 1.0) # Samplers are callable for parameter updates
    @test parameters(S2).γ == 1.0
    @test parameters(S).γ == 0.5
end
