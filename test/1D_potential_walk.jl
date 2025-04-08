using CairoMakie
using Foresight
using DifferentialEquations
using FractionalNeuralSampling
using Distributions
using LinearAlgebra
using TimeseriesTools

import FractionalNeuralSampling: Density
set_theme!(foresight(:physics))

begin
    Δx = 5
    Ng = 2
    centers = range(-(Ng - 1) * Δx / 2, (Ng - 1) * Δx / 2, length = Ng)
    d = MixtureModel([Normal(c, 1) for c in centers])
    D = Density(d)

    boundaries = PeriodicBox(-3 .. -2)
    α = 1.6
    f = Figure()
    L = LevyWalkSampler(;
                        u0 = [-Δx / 2 0],
                        tspan = 500.0,
                        α = α,
                        β = 0.5,
                        γ = 2.0,
                        𝜋 = D,
                        seed = 42,
                        boundaries = boundaries())
    sol = solve(L, EM(); dt = 0.001)
    x = sol[1, :]

    xmax = maximum(abs.(extrema(x))) * 4
    xs = range(-xmax, xmax, length = 100)

    ax = Axis(f[1, 1], title = "α = $α")
    # heatmap!(ax, xs, xs, potential(D).(collect.(Iterators.product(xs, xs))), colormap=seethrough(:turbo))
    lines!(ax, xs, D.(xs))
    lines!(ax, x[1:10:end], D.(x[1:10:end]))
    hidedecorations!(ax)
    f
end
begin
    x = TimeseriesTools.TimeSeries(sol.t, x)
    x = rectify(x, dims = 𝑡, tol = 1)
    scatterlines(x)
    domain = FractionalNeuralSampling.Boundaries.domain(boundaries)
    hlines!(extrema(domain |> only) |> collect, linestyle = :dash, color = :red)
    current_figure()
end

begin # * Power spectrum
    u0 = [-Δx / 2, 0]
    L = LevyWalkSampler(;
                        u0 = u0,
                        tspan = 10000.0,
                        α = 1.6, #α,
                        β = 2.0,
                        γ = 3.0,
                        𝜋 = D,
                        seed = 42,
                        noise_rate_prototype = similar(u0))
    sol = solve(L, EM(); dt = 0.01)
    x = sol[1, :]
    x = TimeseriesTools.TimeSeries(sol.t, x)
    x = rectify(x, dims = 𝑡, tol = 1)
    x = x .- mean(x)
end
begin
    s = spectrum(x, 0.001)
    plot(s, linewidth = 2)

    # * Fit a line to the tail
    s = s[𝑓(1 .. Inf)]
    logsample = exp10.(range(log10(minimum(freqs(s))), log10(maximum(freqs(s))),
                             length = 1000))
    s = s[𝑓 = Near(logsample)]
    a = log.(freqs(s))
    b = log.(s |> collect)
    a = [ones(length(a)) a]
    b, m = a \ b
    lines!(freqs(s), exp.(m * log.(freqs(s)) .+ b), color = (:black, 0.5))
    text!(10, 0.1, text = "α = $(round(m, sigdigits=2))", color = (:black, 0.5))

    # * add a time series inset top
    axx = Axis(current_figure()[1, 1],
               width = Relative(0.5),
               height = Relative(0.3),
               halign = 0.05,
               valign = 0.05)
    hidedecorations!(axx)
    lines!(axx, x[1:5000], color = (:black, 0.5), linewidth = 1)

    current_figure()
end
