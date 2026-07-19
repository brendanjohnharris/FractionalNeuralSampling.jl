using FractionalNeuralSampling
using Distributions
using LinearAlgebra
using Random
using Test

import FractionalNeuralSampling.Solvers: full_cache, jac_iter, rand_cache,
    ratenoise_cache

begin # * 2D overdamped sampler; target factorises so x₂ evolves independently of x₁
    dt = 0.01
    η = 0.1
    𝜋 = MvNormal(zeros(2), I(2)) |> FractionalNeuralSampling.Density # Gradient -x factorises
    u0 = [0.1, -0.1]
    tspan = 10.0
    S = OLE(; η, u0, 𝜋, tspan)
end

begin # * Argument checks
    @test_throws ArgumentError PositionalCaputoEM(0.0, 10)
    @test_throws ArgumentError PositionalCaputoEM(1.5, 10)
end

begin # * β₁ = 1 reduces to standard EM
    Random.seed!(1234)
    sol_em = solve(S, EM(); dt)
    Random.seed!(1234)
    sol_pos = solve(S, PositionalCaputoEM(1.0, 100); dt)
    @test sol_em.u == sol_pos.u
    @test SciMLBase.successful_retcode(sol_pos)
end

begin # * Fractional β₁: x₁ deviates from EM while x₂ stays on the EM path
    Random.seed!(1234)
    sol_em = solve(S, EM(); dt)
    Random.seed!(1234)
    sol_pos = solve(S, PositionalCaputoEM(0.6, 100); dt)
    x1_em, x1_pos = getindex.(sol_em.u, 1), getindex.(sol_pos.u, 1)
    x2_em, x2_pos = getindex.(sol_em.u, 2), getindex.(sol_pos.u, 2)
    @test x1_em != x1_pos
    @test x2_em ≈ x2_pos
    @test all(all(isfinite, u) for u in sol_pos.u)
end

begin # * Integrator internals and cache interface
    int = StochasticDiffEq.init(S, PositionalCaputoEM(0.75, 100); dt)
    c = int.cache
    @test c isa FractionalNeuralSampling.Solvers.PositionalCaputoEMCache
    @test full_cache(c) == (c.u, c.uhist1, c.weights1, c.correction1, c.tmp, c.rtmp1)
    @test jac_iter(c) === ()
    @test rand_cache(c) === ()
    @test ratenoise_cache(c) === (c.rtmp2,)
    @test length(c.weights1) == 100
    @inferred StochasticDiffEq.perform_step!(int, int.cache)
end
