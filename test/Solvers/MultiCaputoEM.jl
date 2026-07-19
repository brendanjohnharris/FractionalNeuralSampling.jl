using FractionalNeuralSampling
using Distributions
using LinearAlgebra
using Random
using Test

import FractionalNeuralSampling.Solvers: full_cache, jac_iter, rand_cache,
                                         ratenoise_cache

begin # * 2D overdamped sampler with independent Gaussian target
    dt = 0.01
    η = 0.1
    𝜋 = MvNormal(zeros(2), I(2)) |> FractionalNeuralSampling.Density # Analytic gradlogpdf path
    u0 = [0.1, -0.1]
    tspan = 10.0
    S = OLE(; η, u0, 𝜋, tspan)
end

begin # * Constructors and argument checks
    alg = MultiCaputoEM(0.8, 2, 100) # Uniform-β convenience constructor
    @test alg.β == [0.8, 0.8]
    @test_throws ArgumentError MultiCaputoEM([1.5, 1.0], 10)
    @test_throws ArgumentError MultiCaputoEM([0.0, 1.0], 10)
    @test_throws DimensionMismatch solve(S, MultiCaputoEM([1.0], 100); dt) # β-u0 mismatch
end

begin # * β = 1 for all variables reduces to standard EM
    Random.seed!(1234)
    sol_em = solve(S, EM(); dt)
    Random.seed!(1234)
    sol_multi = solve(S, MultiCaputoEM([1.0, 1.0], 100); dt)
    @test sol_em.u == sol_multi.u
    @test SciMLBase.successful_retcode(sol_multi)
end

begin # * Fractional orders alter the trajectory but remain finite
    Random.seed!(1234)
    sol_frac = solve(S, MultiCaputoEM([0.6, 0.9], 100); dt)
    @test all(all(isfinite, u) for u in sol_frac.u)
    @test length(sol_frac.u) == length(sol_frac.t)
    Random.seed!(1234)
    sol_em = solve(S, EM(); dt)
    @test sol_frac.u != sol_em.u
end

begin # * Integrator internals and cache interface
    int = StochasticDiffEq.init(S, MultiCaputoEM([0.9, 0.9], 100); dt)
    c = int.cache
    @test c isa FractionalNeuralSampling.Solvers.MultiCaputoEMCache
    @test full_cache(c) == (c.u, c.uhist, c.weights, c.correction, c.tmp, c.rtmp1)
    @test jac_iter(c) === ()
    @test rand_cache(c) === ()
    @test ratenoise_cache(c) === (c.rtmp2,)
    @test size(c.weights) == (2, 100) # nvars × nhist
    @inferred StochasticDiffEq.perform_step!(int, int.cache)
end
