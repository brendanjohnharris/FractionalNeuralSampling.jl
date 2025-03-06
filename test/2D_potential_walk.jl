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
    Ng = 3
    phis = range(0, stop = 2π, length = Ng + 1)[1:Ng]
    centers = exp.(2π * im * phis)
    d = MixtureModel([MvNormal([real(c), imag(c)], I(2)) for c in centers])
    D = Density(d)

    αs = [2.0, 1.6, 1.2]
    f = Figure(size = (900, 300))
    gs = subdivide(f, 1, 3)
    map(αs, gs) do α, g
        L = LevyWalkSampler(;
                            u0 = [-Δx / 2 0 0 0],
                            tspan = 500.0,
                            α = α,
                            β = 0.1,
                            γ = 1.0,
                            𝜋 = D,
                            seed = 42)
        sol = solve(L, EM(); dt = 0.001)
        x, y = eachrow(sol[1:2, :])

        xmax = maximum(abs.(extrema(vcat(x, y)))) * 1.5
        xs = range(-xmax, xmax, length = 100)

        ax = Axis(g[1, 1], title = "α = $α", aspect = DataAspect())
        # heatmap!(ax, xs, xs, potential(D).(collect.(Iterators.product(xs, xs))), colormap=seethrough(:turbo))
        heatmap!(ax, xs, xs, D.(collect.(Iterators.product(xs, xs))),
                 colormap = seethrough(:turbo))
        lines!(ax, x[1:10:end], y[1:10:end], color = (:black, 0.5), linewidth = 1)
        hidedecorations!(ax)
    end
    f
end
