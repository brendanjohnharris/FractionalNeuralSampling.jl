using CairoMakie
using Foresight
using DifferentialEquations
using FractionalNeuralSampling
import FractionalNeuralSampling: Density
set_theme!(foresight(:physics))

begin
    Δx = 5
    d = MixtureModel([
                         MvNormal([-Δx / 2, 0], [1 0; 0 1]),
                         MvNormal([Δx / 2, 0], [1 0; 0 1])
                     ])
    D = Density(d)

    αs = [2.0, 1.6, 1.2]
    f = Figure(size = (900, 300))
    gs = subdivide(f, 1, 3)
    map(αs, gs) do α, g
        L = LevyFlightSampler(;
                              u0 = [-Δx / 2 0 0 0],
                              tspan = 500.0,
                              α = α,
                              β = 0.2,
                              γ = 0.02,
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
