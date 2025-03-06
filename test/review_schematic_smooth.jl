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
import FractionalNeuralSampling: Density
set_theme!(foresight(:physics))

function peaks(x, y) # A quasi-potential
    return 3 * (1 - x)^2 * exp(-x^2 - (y + 1)^2) -
           10 * (x / 5 - x^3 - y^5) * exp(-x^2 - y^2) -
           (1 / 3) * exp(-(x + 1)^2 - y^2)
end
peaks(xy) = peaks(xy...)
function radialfade(xs, ys, zs; colormap, sigma)
    xys = Iterators.product(xs, ys)
    colors = cgrad(colormap, extrema(zs))
    zhat = zs .- minimum(zs)
    zhat ./= maximum(zhat)
    colors = [colors[i] for i in zhat]
    radii = norm.(xys)
    radii ./= maximum(radii)
    alphas = exp.(-radii .^ 2 / sigma^2)
    colors = map(colors, alphas) do c, a
        RGBA(c.r, c.g, c.b, a)
    end
    return colors
end
begin
    d = peaks
    D = PotentialDensity{2}(d)
    surf = potential(D)

    xmax = 6
    box = ReflectingBox(-xmax .. xmax, -xmax .. xmax)
    L = LevyWalkSampler(;
                        u0 = [0 0 0 0],
                        tspan = 1600.0,
                        α = 1.5,
                        β = 0.009,
                        γ = 0.2,
                        𝜋 = D,
                        seed = 28,
                        boundaries = box())
    # L = LevyFlightSampler(;
    #                       u0 = [0 0 0 0],
    #                       tspan = 1000.0,
    #                       α = 1.5,
    #                       β = 0.015,
    #                       γ = 0.007,
    #                       𝜋 = D,
    #                       seed = 62,
    #                       boundaries = box())

    sol = solve(L, EM(); dt = 0.01)
    xs = ys = range(-xmax, xmax, length = 100)
    x, y = eachrow(sol[1:2, :])

    zs = surf.(collect.(Iterators.product(xs, xs)))

    f = Figure(size = (800, 600))
    gs = subdivide(f, 1, 3)
    g = first(gs)
    ax = Axis3(g[1, 1], aspect = (1, 1, 0.3))

    color = radialfade(xs, ys, zs, colormap = :turbo, sigma = 0.3)
    surface!(ax, xs, xs, zs; color)

    subd = 10
    z = map(surf ∘ collect, zip(x[1:subd:end], y[1:subd:end]))
    lines!(ax, x[1:subd:end], y[1:subd:end], z, color = (:black, 0.5),
           linewidth = 2)

    ax.yreversed = true
    hidedecorations!(ax)
    ax.azimuth = 2.0
    ax.elevation = 0.8
    save("test/review_schematic.pdf", f)
    f
end
