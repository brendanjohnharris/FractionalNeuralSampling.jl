using FractionalNeuralSampling
using Distributions
using CairoMakie
using TimeseriesTools
using Random

begin # * Make Sampler
    dt = 0.1
    η = 0.1
    𝜋 = MixtureModel([Normal(-3, 1), Normal(3, 1)]) |> FractionalNeuralSampling.Density
    u0 = [0.0]
    tspan = 1000.00
    S = OLE(; η, u0, 𝜋, tspan)
end

begin # * Standard EM
    Random.seed!(1234)
    sol = solve(S, EM(); dt) |> Timeseries |> eachcol |> first
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "t", ylabel = "x(t)")
    lines!(ax, sol)
    display(f)
end

begin # * Same fractional EM Ok.
    Random.seed!(1234)
    sol2 = solve(S, FractionalEM(); dt) |> Timeseries |> eachcol |> first
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "t", ylabel = "x(t)")
    lines!(ax, sol)
    display(f)

    sol == sol2
end
