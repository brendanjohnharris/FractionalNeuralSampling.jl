using CairoMakie
using Foresight
using DifferentialEquations
using FractionalNeuralSampling
using Distributions
using LinearAlgebra
using RecursiveArrayTools

import FractionalNeuralSampling: Density
set_theme!(foresight(:physics))

begin
    Δx = 5
    Ng = 1
    centers = range(-(Ng - 1) * Δx / 2, (Ng - 1) * Δx / 2, length = Ng)
    d = MixtureModel([Normal(c, 1) for c in centers])
    D = Density(d)

    f = Figure()
    u0 = ArrayPartition([0.5], [0.0])
    u0 = [0.0, 0.0]
    noise_rate_prototype = similar(u0)
    L = LangevinSampler(;
                        u0,
                        tspan = 1000.0,
                        β = 0.0,
                        γ = 1.0,
                        𝜋 = D,
                        seed = 50,
                        noise_rate_prototype,
                        noise = WienerProcess!(0.0, u0))
    # noise = NoiseProcesses.LevyProcess!(2.0; ND = 2,
    #                                     W0 = zero(u0)))
    sol = solve(L, EM(); dt = 0.001)
    x = sol[1, :]
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

begin
    using StochasticDiffEq
    α = 1
    β = 1
    u₀ = 1 / 2
    f(u, p, t) = α * u
    g(u, p, t) = β * u
    dt = 1 // 2^(4)
    tspan = (0.0, 1.0)
    prob = SDEProblem(f, g, u₀, (0.0, 1.0))
    sol = solve(prob, EM(); dt)
end
