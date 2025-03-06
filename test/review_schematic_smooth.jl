using TestEnv;
TestEnv.activate();
using CairoMakie
using Foresight
using DifferentialEquations
using FractionalNeuralSampling
using Distributions
using LinearAlgebra
using Interpolations
using FileIO
using Normalization
using IntervalSets
using CairoMakie
using CairoMakie.Colors
# using GLMakie
# using GLMakie.Colors
import FractionalNeuralSampling: Density
set_theme!(foresight(:physics))

function peaks(xy)
    x, y = xy
    return 3 * (1 - x)^2 * exp(-x^2 - (y + 1)^2) -
           10 * (x / 5 - x^3 - y^5) * exp(-x^2 - y^2) -
           (1 / 3) * exp(-(x + 1)^2 - y^2)
end
function peaks_dist(xy)
    x, y = xy
    negV = -3 * (1 - x)^2 * exp(-x^2 - (y + 1)^2) +
           10 * ((x / 5) - x^3 - y^5) * exp(-x^2 - y^2) +
           (1 / 3) * exp(-(x + 1)^2 - y^2)

    return exp(negV)
end
begin
    d = peaks_dist
    # d = MixtureModel([MvNormal(μ, I(2) * σ) for (μ, σ) in zip(centers, widths)])
    xmax = 10
end
begin
    xs = range(-xmax, xmax, length = 100)
    D = Density(peaks_dist, d; dimension = 2)

    f = Figure()#size = (900, 300))
    αs = [2.0, 1.6, 1.2]
    gs = subdivide(f, 1, 3)
    # map(αs, gs) do α, g
    α = 1.6
    g = first(gs)
    begin
        xmax = 6.0
        cmax = 7.5 # Controls what range of the z axis is plotted

        surf = (-) ∘ D
        subd = 10
        box = ReflectingBox(-xmax .. xmax, -xmax .. xmax)
        @info "Starting simulation"
        L = LevyWalkSampler(;
                            u0 = [0 0 0 0],
                            tspan = 1000.0,
                            α = α,
                            β = 0.2,
                            γ = 5.0,
                            𝜋 = D,
                            seed = 42,
                            boundaries = box())
        sol = solve(L, EM(); dt = 0.001)
        trans = length(sol) ÷ 2
        xs = range(-xmax, xmax, length = 1000)
        x, y = eachrow(sol[1:2, :])

        zs = surf.(collect.(Iterators.product(xs, xs)))
        z = map(surf ∘ collect, zip(x[trans:subd:end], y[trans:subd:end]))

        ax = Axis3(g[1, 1], title = "α = $α")# ,                   limits = (nothing, nothing, (minimum(z) - 1, maximum(z))),                   aspect = (5.0, 5.0, 2 / 3))
        # heatmap!(ax, xs, xs, potential(D).(collect.(Iterators.product(xs, xs))),
        # colormap=seethrough(:turbo))
        # cmax = maximum(abs.(extrema(zs)))
        # idxs = xs .∈ [-cxmax .. cxmax] # Idxs of zs within the original domain
        crange = [minimum(zs), cmax]
        crange[1] = crange[1] .- 0.5 * diff(crange) |> only
        surface!(ax, xs, xs, zs, colormap = :turbo, colorrange = extrema(zs[idxs, idxs]))
        # reverse(seethrough(:turbo, -1))
    end
    begin
        subd = 10
        lines!(ax, x[trans:subd:end], y[trans:subd:end], z, color = (:black, 0.5),
               linewidth = 2)
    end
    # save("test/review_schematic.pdf", f)
    f
end
