using FractionalNeuralSampling
using Distributions
using CairoMakie
using TimeseriesTools
using Random
using Foresight
using Test
using BenchmarkTools
set_theme!(foresight(:physics))

begin # * Make Sampler
    dt = 0.001
    η = 0.1
    # 𝜋 = MixtureModel([Normal(-3, 1), Normal(3, 1)]) |> FractionalNeuralSampling.Density
    𝜋 = PotentialDensity{1}(_ -> 0.0)
    u0 = [0.0]
    tspan = 100.00
    S = OLE(; η, u0, 𝜋, tspan)
end

begin # * Standard EM
    Random.seed!(1234)
    alg = @inferred EM()
    sol = solve(S, alg; dt) |> Timeseries |> eachcol |> only
    f = TwoPanel()
    ax = Axis(f[1, 1], xlabel = "t", ylabel = "x(t)")
    lines!(ax, sol, linewidth = 3)
    display(f)
end

begin # * Same fractional EM Ok.
    Random.seed!(1234)
    _sol2 = solve(S, FractionalEM(1.0, 1000); dt)
    sol2 = _sol2 |> Timeseries |> eachcol |> only
    ax = Axis(f[1, 2], xlabel = "t", ylabel = "x(t)")
    lines!(ax, sol2, linewidth = 3)
    display(f)

    @test sol == sol2
end

begin # * Stepping cost
    alg = @inferred EM()
    alg2 = @inferred FractionalEM(0.75f0, 1000)

    int = StochasticDiffEq.init(S, alg, dt = dt)
    int2 = StochasticDiffEq.init(S, alg2, dt = dt)
    @inferred StochasticDiffEq.perform_step!(int, int.cache)
    @inferred StochasticDiffEq.perform_step!(int2, int2.cache)

    # * Benchmark
    @info "EM"
    @benchmark StochasticDiffEq.perform_step!($int, $int.cache)

    @info "FractionalEM"
    @benchmark StochasticDiffEq.perform_step!($int2, $int2.cache)
end

begin # * Check finer timestep using NoiseGrid
    Random.seed!(1234)
    noise = [[randn()] for n in 1:100000] |> cumsum # Must be the integral of the noise
    ts = range(S.tspan..., length = length(noise))
    W = StochasticDiffEq.NoiseGrid(ts, noise)
    S2 = OLE(; η, u0, 𝜋, tspan, noise = W)

    dt2 = 0.01  # Vary and see we get rougher path, but not different overall. Dont drop it below 0.01
    alg = @inferred FractionalEM(0.6, Int(100 ÷ dt2))
    _sol2 = solve(S2, alg; dt = dt2)
    sol2 = _sol2 |> Timeseries |> eachcol |> only

    dt2 = 0.1  # Vary and see we get rougher path, but not different overall. Dont drop it below 0.01
    alg = @inferred FractionalEM(0.6, Int(100 ÷ dt2))
    _sol3 = solve(S2, alg; dt = dt2)
    sol3 = _sol3 |> Timeseries |> eachcol |> only

    ts = range(S2.tspan..., length = min(length(sol2), length(sol3)))
    @test cor(sol2[𝑡 = Near(ts)], sol3[𝑡 = Near(ts)]) > 0.95

    f = TwoPanel()
    ax = Axis(f[1, 1], xlabel = "t", ylabel = "x(t)")
    lines!(ax, sol2, linewidth = 3)

    ax = Axis(f[1, 2], xlabel = "t", ylabel = "x(t)")
    lines!(ax, sol3, linewidth = 3)
    display(f)
end

begin # * Plot power spectrum
    s = spectrum(rectify(sol, dims = 𝑡; tol = 1), 1)[10:end]
    s2 = spectrum(rectify(sol2, dims = 𝑡; tol = 1), 1)[10:end]

    plotspectrum(s)
    plotspectrum!(current_axis(), s2)
    display(current_figure())
end

begin # * 2D example
    using LinearAlgebra
    Random.seed!(1234)
    dt = 0.01
    η = 0.1
    𝜋 = MixtureModel([MvNormal([-3.0, -3.0], I), MvNormal([3.0, 3.0], I)]) |>
        FractionalNeuralSampling.Density
    u0 = [0.0, 0.0]
    tspan = 10.0
    S = OLE(; η, u0, 𝜋, tspan)

    alg = @inferred FractionalEM(0.5, 1000)
    _sol2 = solve(S, alg; dt)
    sol2 = _sol2 |> Timeseries

    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "y")
    lines!(ax, eachcol(sol2)..., linewidth = 1)
    display(f)
end
