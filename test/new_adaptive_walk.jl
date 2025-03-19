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
    σ = 2.0 # Width of adaptive kernel
    n_modes = 100
    kernel = x -> exp(-(norm(x)^2) / (2 * σ^2))
    # kernel = x -> exp(-sqrt(norm(x)) / σ)

    begin # 2d
        boundaries = PeriodicBox(-8 .. 8, -8 .. 8)

        s = 0.4
        Δx = 4 # Controls spacing between wells
        Ng = 3
        phis = range(0, stop = 2π, length = Ng + 1)[1:Ng]
        centers = Δx .* exp.(im * phis)
        d = MixtureModel([MvNormal([real(c), imag(c)], s^2 .* I(2)) for c in centers])
        D = Density(d)

        u0 = [-2.0, -2.0]
    end

    τ_d = 50.0
    A = AdaptiveLevySampler(kernel, 500;
                            tspan = 100.0,
                            α = 1.4,
                            γ = 0.3,
                            τ_d,
                            τ_r = τ_d / 100,
                            u0,
                            boundaries,
                            𝜋 = D)

    sol = solve(A, EM(); dt = 0.1)
    @info "Adaptive walk solved"
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

begin # * Plot adaptive potential
    is = round.(Int, range(1, length(sol), length = 1000))
    xs, ys = FractionalNeuralSampling.gridaxes(boundaries, 100)
    xys = FractionalNeuralSampling.grid(boundaries, 100)

    sp = A.p[1][:sp]
    Ks = [fill(Float32(NaN), size(xys)) for i in 1:length(is)]
    Threads.@threads for n in eachindex(is)
        i = is[n]
        s = sol[i]
        F = Fun(sp, s.x[end])
        Ks[n] .= map(F, xys)
    end
    clims = extrema(Iterators.flatten(Ks))
end
begin
    begin # * Animation. On the left, the distribution and trajectory. On the right, the adaptive potential
        tail = 100
        xy = Observable(Point2f.(fill(sol[1].x[1], tail)))
        color = Observable(zeros(length(xy[])))
        K = Observable(first(Ks))

        limits = FractionalNeuralSampling.domain(boundaries) .|> extrema |> Tuple

        f = Figure(size = (400, 400))
        ax = Axis(f[1, 1]; xlabel = "Position x", ylabel = "Position y", limits,
                  xaxisposition = :top, xticksvisible = true, yticksvisible = true,
                  xticklabelsvisible = false, yticklabelsvisible = false, xtickalign = 1,
                  ytickalign = 1, xticksmirrored = true, yticksmirrored = true,
                  xgridvisible = false, ygridvisible = false)
        heatmap!(ax, xs, ys, K, colormap = seethrough(:turbo), colorrange = clims)
        contour!(ax, xs, ys, map((D), xys), colormap = seethrough(:turbo))
        lines!(ax, xy; color, colorrange = (0, 1), colormap = seethrough(:binary))
        # f
    end
    dosides = true
    if dosides # * Add an axis below showing the x trace
        xtrace = getindex.(sol.u, 1)
        ytrace = getindex.(sol.u, 2)
        xt = Observable([Point2f([first(sol.t), first(xtrace)])])
        yt = Observable([Point2f([first(ytrace), first(sol.t)])])

        ax = Axis(f[2, 1]; xlabel = "Time", limits = (extrema(sol.t), extrema(xtrace)))
        colsize!(f.layout, 1, Relative(0.8))
        rowsize!(f.layout, 2, Relative(0.2))
        lines!(ax, xt, color = :black)
        hidedecorations!(ax)

        ax = Axis(f[1, 2]; ylabel = "Time", limits = (extrema(ytrace), extrema(sol.t)))
        colsize!(f.layout, 2, Relative(0.2))
        rowsize!(f.layout, 1, Relative(0.8))
        colgap!(f.layout, 1, Relative(0.0))
        rowgap!(f.layout, 1, Relative(0.0))
        lines!(ax, yt, color = :black)
        hidedecorations!(ax)
    end
    begin# * Animate
        tau = tail / 10
        w(i0) = i -> exp(-(i0 - i) / tau)
        ws = map(Float32 ∘ w(tail), 1:tail)
        color[] = ws
        record(f, "adaptive_walk.mp4", enumerate(is); framerate = 30) do (n, i)
            xy[][1:(end - 1)] .= xy[][2:end]
            xy[][end] = Point2f(sol[i].x[1])

            xt[] = push!(xt[], Point2f(sol.t[i], xtrace[i]))
            yt[] = push!(yt[], Point2f(ytrace[i], sol.t[i]))

            K[] = Ks[n]
        end
    end
end
# begin # * Test matrix exponential stuff
#     function plotfun(fun::Fun, space)
#         bounds = domain(space)
#         bounds = bounds.domains
#         bounds = Interval.(bounds)
#         xs = [range(b[1], b[2], length = 100) for b in extrema.(bounds)]
#         xys = Iterators.product(xs...)
#         zs = map(fun, map(collect, xys))
#         f = Figure(size = (400, 400))
#         ax = Axis(f[1, 1]; xlabel = "Position x", ylabel = "Position y", aspect = 1)
#         heatmap!(ax, xs[1], xs[2], zs, colormap = reverse(seethrough(:turbo)))
#         return f
#     end

#     kernel = x -> exp(-norm(x)^2)
#     sp = prod(Fourier.([-10 .. 10, -10 .. 10]))
#     approx_n_modes = 1000
#     k = Fun(kernel, sp, approx_n_modes)
#     kc = Fun(sp, CuArray(k.coefficients))

#     D = Derivative(sp, [0, 1])
#     @time D * kc
#     f = plotfun(k, sp)
# end

# function mexp(A, n = 20)
#     terms = map(i -> A^i / factorial(i), 0:n)
#     reduce(+, terms)
# end
# begin
#     Dx = Derivative([1, 0])
#     Dy = Derivative(sp, [0, 1])
#     C = Convolution(k)

#     # re-transform a shifted version
#     x0 = [4.0, 4.0]
#     ps = points(sp, approx_n_modes) # It's fucked but the num of point shift like nobody is watching in 2D; maybe open a pr?
#     plan = ApproxFunBase.plan_transform(sp, length(ps))

#     a_k̂ = plan * map(kernel ∘ Base.Fix2(-, x0), ps)
#     k̂ = Fun(sp, a_k̂)
#     f = plotfun(k̂, sp)
# end

# function fourierns(n)
#     nmax = ceil(Int, (n - 1) / 2)
#     ns = 1:nmax
#     ns = hcat(ns, ns)'[:]
#     prepend!(ns, 0)
#     ns = ns[1:n]
#     return ns
# end
# function fourierfreqs(space::TensorSpace, n)
# end
# function fourierfreqs(space::Space, n)
# end

# N = 264
# N^2 + N - N / 2 = 264

# 15^2
