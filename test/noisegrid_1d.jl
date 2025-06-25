using FractionalNeuralSampling # For Sampler
using Distributions # For MixtureModel
using StochasticDiffEq # For EM solver
using RecursiveArrayTools # For ArrayPartition
using CairoMakie
using StableDistributions

begin # * Set up distribution
    Δx = 5
    d = MixtureModel([
                         Normal(-Δx / 2, 1),
                         Normal(Δx / 2, 1)
                     ])
    𝜋 = Density(d)
end

begin # * Set up sampler
    u0 = [-Δx / 2]
    tspan = 500.0
    γ = 0.5

    L = OverdampedLangevinSampler(; u0, tspan, γ, 𝜋, seed = 42)
end

begin # * Solve
    dt = 0.001

    sol = solve(L, EM(); dt)
    x = sol[1, :]
    lines(x; linewidth = 1, color = :blue)
end

function dist_and_fit!(ax, x; dist = Stable, kwargs...)
    shared_keys = intersect(keys(kwargs), [:linecolor, :color, :linewidth])
    shared_args = Dict(shared_keys .=> getindex.([kwargs], shared_keys))
    hist!(ax, x; normalization = :pdf, kwargs...)
    d = fit(dist, x)

    xs = range(extrema(x)..., length = 1000)
    lines!(ax, xs, pdf(d, xs); shared_args...)

    ax.title = "α = $(round(d.α, sigdigits = 2)), β = $(round(d.β, sigdigits = 2)), μ = $(round(d.μ, sigdigits = 2)), σ = $(round(d.σ, sigdigits = 2))
    Variance = $(round(2 * d.σ^2, sigdigits = 2))"
    return nothing
end
begin # * Step size distribution of the integrated trajectory
    f = Figure()
    ax = Axis(f[1, 1])
    dist_and_fit!(ax, diff(x); bins = 100)

    ax.title = "$(ax.title[])
    (expected $(round(2*sol.prob.p[1] * dt, sigdigits = 2)))"
    f |> display
end
begin # * Underlying trajectory of the noise process
    f = Figure()
    ax = Axis(f[1, 1])
    dist_and_fit!(ax, diff(sol.W[1, :]); bins = 100)

    ax.title = "$(ax.title[])
    (expected $(round(dt, sigdigits = 2)))"
    f |> display
end
