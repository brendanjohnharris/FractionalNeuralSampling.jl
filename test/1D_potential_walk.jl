using CairoMakie
using Foresight
using DifferentialEquations
using FractionalNeuralSampling
using Distributions
using LinearAlgebra

import FractionalNeuralSampling: Density
set_theme!(foresight(:physics))

begin
    Δx = 5
    Ng = 2
    centers = range(-(Ng - 1) * Δx / 2, (Ng - 1) * Δx / 2, length = Ng)
    d = MixtureModel([Normal(c, 1) for c in centers])
    D = Density(d)

    α = 1.6
    f = Figure()
    L = LevyWalkSampler(;
                        u0 = [-Δx / 2 0],
                        tspan = 500.0,
                        α = α,
                        β = 0.5,
                        γ = 2.0,
                        𝜋 = D,
                        seed = 42)
    sol = solve(L, EM(); dt = 0.001)
    x = sol[1, :]

    xmax = maximum(abs.(extrema(vcat(x, y)))) * 4
    xs = range(-xmax, xmax, length = 100)

    ax = Axis(f[1, 1], title = "α = $α")
    # heatmap!(ax, xs, xs, potential(D).(collect.(Iterators.product(xs, xs))), colormap=seethrough(:turbo))
    lines!(ax, xs, D.(xs))
    lines!(ax, x[1:10:end], D.(x[1:10:end]))
    hidedecorations!(ax)
    f
end
begin
    lines(sol.t, x)
end
