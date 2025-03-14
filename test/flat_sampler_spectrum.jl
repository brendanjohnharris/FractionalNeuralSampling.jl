using CairoMakie
using Foresight
using DifferentialEquations
using FractionalNeuralSampling
using Distributions
using LinearAlgebra
using TimeseriesTools
using RecursiveArrayTools

import FractionalNeuralSampling: Density
set_theme!(foresight(:physics))

begin
    V(x) = 0.0
    D = PotentialDensity{1}(V)

    ff = Figure()
    u0 = VectorOfArray([-Δx / 2, 0.0])
    L = LangevinSampler(;
                        u0,
                        tspan = 1000.0,
                        β = 0.5,
                        γ = 2.0,
                        𝜋 = D,
                        seed = 50,
                        noise_rate_prototype = [0.0, 0.0])
    sol = solve(L, EM(); dt = 0.001)
    x = sol[1, :]

    xmax = maximum(abs.(extrema(x))) * 4
    xs = range(-xmax, xmax, length = 100)

    # ax = Axis(ff[1, 1])
    # # heatmap!(ax, xs, xs, potential(D).(collect.(Iterators.product(xs, xs))), colormap=seethrough(:turbo))
    # lines!(ax, xs, D.(xs))
    # lines!(ax, x[1:10:end], D.(x[1:10:end]))
    # hidedecorations!(ax)
    # ff
end
begin
    lines(sol.t, x)
end
begin
    y = TimeseriesTools.TimeSeries(sol.t, x)
    y = rectify(y, dims = 𝑡, tol = 1)
    y = y .- mean(y)
    s = spectrum(y, 0.01)
    s = s[2:end]

    fmin = minimum(freqs(s))
    fmax = maximum(freqs(s))
    idxs_even = exp10.(range(log10(fmin), log10(fmax), length = 1000))
    s = s[𝑓 = Near(idxs_even)]
    _x = [ones(length(s)) freqs(s) .|> log10]
    _y = parent(s) .|> log10
    b, m = _x \ _y
    plot(s)
    lines!(freqs(s), 10 .^ (m * log10.(freqs(s)) .+ b), color = :red)
    current_figure()
end
