using CairoMakie
using Foresight
using DifferentialEquations
using FractionalNeuralSampling
using Distributions
using LinearAlgebra
using TimeseriesTools
using ApproxFun

import FractionalNeuralSampling: Density, AdaptiveWalkSampler

begin # * Try construction
    σ = 2.0
    n_modes = 100
    kernel = x -> exp(-(norm(x)) / (2 * σ^2))
    # kernel = x -> exp(-sqrt(norm(x)) / σ)

    if false # 1d
        boundaries = PeriodicBox(-10 .. 10)
        D = MixtureModel([Normal(-5, 0.5), Normal(0, 0.5), Normal(5, 0.5)])
        D = Density(D)
        u0 = [-3.0]
    else # 2d
        boundaries = PeriodicBox(-10 .. 10, -10 .. 10)

        Δx = 3 # Controls spacing between wells
        Ng = 3
        phis = range(0, stop = 2π, length = Ng + 1)[1:Ng]
        centers = Δx .* exp.(im * phis)
        d = MixtureModel([MvNormal([real(c), imag(c)], I(2)) for c in centers])
        D = Density(d)

        u0 = [-2.0, -2.0]
    end

    τ_d = 10.0
    A = AdaptiveWalkSampler(kernel, 1000;
                            tspan = 50.0,
                            γ = 0.1,
                            τ_d,
                            τ_r = τ_d / 40,
                            u0,
                            boundaries,
                            𝜋 = D)

    sol = solve(A, EM(); dt = 0.01)
end

begin # * Plot
    f = Figure(size = (800, 400))
    ax = Axis(f[1, 1]; xlabel = "Time", ylabel = "Position")
    lines!(ax, sol.t, sol[1, :], color = :blue)
    f
end

if length(u0) == 2
    ax = Axis(f[1, 2]; xlabel = "Position x", ylabel = "Position y")

    xs = range(-10, 10, length = 100)
    ys = range(-10, 10, length = 100)
    xys = Iterators.product(xs, ys)
    zs = map((D), map(collect, xys))
    heatmap!(ax, xs, ys, zs, colormap = seethrough(:turbo))

    lines!(ax, sol[1, :], sol[2, :], color = :black, alpha = 0.5, linewidth = 0.5)

    f
end

begin # * Test matrix exponential stuff
    function plotfun(fun::Fun, space)
        bounds = domain(space)
        bounds = bounds.domains
        bounds = Interval.(bounds)
        xs = [range(b[1], b[2], length = 100) for b in extrema.(bounds)]
        xys = Iterators.product(xs...)
        zs = map(fun, map(collect, xys))
        f = Figure(size = (400, 400))
        ax = Axis(f[1, 1]; xlabel = "Position x", ylabel = "Position y", aspect = 1)
        heatmap!(ax, xs[1], xs[2], zs, colormap = reverse(seethrough(:turbo)))
        return f
    end

    kernel = x -> exp(-norm(x)^2)
    sp = prod(Fourier.([-10 .. 10, -10 .. 10]))
    approx_n_modes = 1000
    k = Fun(kernel, sp, approx_n_modes)
    kc = Fun(sp, CuArray(k.coefficients))

    D = Derivative(sp, [0, 1])
    @time D * kc
    f = plotfun(k, sp)
end

function mexp(A, n = 20)
    terms = map(i -> A^i / factorial(i), 0:n)
    reduce(+, terms)
end
begin
    Dx = Derivative([1, 0])
    Dy = Derivative(sp, [0, 1])
    C = Convolution(k)

    # re-transform a shifted version
    x0 = [4.0, 4.0]
    ps = points(sp, approx_n_modes) # It's fucked but the num of point shift like nobody is watching in 2D; maybe open a pr?
    plan = ApproxFunBase.plan_transform(sp, length(ps))

    a_k̂ = plan * map(kernel ∘ Base.Fix2(-, x0), ps)
    k̂ = Fun(sp, a_k̂)
    f = plotfun(k̂, sp)
end

function fourierns(n)
    nmax = ceil(Int, (n - 1) / 2)
    ns = 1:nmax
    ns = hcat(ns, ns)'[:]
    prepend!(ns, 0)
    ns = ns[1:n]
    return ns
end
function fourierfreqs(space::TensorSpace, n)
end
function fourierfreqs(space::Space, n)
end

N = 264
N^2 + N - N / 2 = 264

15^2
