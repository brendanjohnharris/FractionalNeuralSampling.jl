using FractionalNeuralSampling # For Sampler
using Distributions # For MixtureModel
using StochasticDiffEq # For EM solver
using RecursiveArrayTools # For ArrayPartition
using CairoMakie
using StableDistributions

begin # * Set up distribution
    Δx = 5
    d = MixtureModel([
                         MvNormal([-Δx / 2, 0], [1 0; 0 1]),
                         MvNormal([Δx / 2, 0], [1 0; 0 1])
                     ])
    𝜋 = Density(d)
end

begin # * Set up sampler
    u0 = [-Δx / 2, 0.0]
    tspan = 500.0
    γ = 1.0

    L = OverdampedLangevinSampler(; u0, tspan, γ, 𝜋, seed = 42)
end

begin # * Solve
    dt = 0.001

    sol = solve(L, EM(); dt)
    x, y = eachrow(sol[1:2, :])
    lines(x, y; linewidth = 1, color = :blue)
end

function dist_and_fit!(ax, x; dist = Stable, kwargs...)
    shared_keys = intersect(keys(kwargs), [:linecolor, :color, :linewidth])
    shared_args = Dict(shared_keys .=> getindex.([kwargs], shared_keys))
    hist!(ax, x; normalization = :pdf, kwargs...)
    d = fit(dist, x)

    xs = range(extrema(x)..., length = 1000)
    lines!(ax, xs, pdf(d, xs); shared_args...)

    v = 2 * d.σ^2
    ax.title = "α = $(round(d.α, sigdigits = 2)), β = $(round(d.β, sigdigits = 2)), μ = $(round(d.μ, sigdigits = 2)), σ = $(round(d.σ, sigdigits = 2))
    Variance = $(round(v, sigdigits = 2)) (expected $(round(sol.prob.p[1] * dt, sigdigits = 2)))"
    return nothing
end
begin # * Compare distributions. Variance is 2*sigma^2
    f = Figure()
    ax = Axis(f[1, 1])
    dist_and_fit!(ax, diff(x); bins = 100)
    f |> display
end
begin # * Compare distributions. Variance is 2*sigma^2
    f = Figure()
    ax = Axis(f[1, 1])
    dist_and_fit!(ax, diff(sol.W[1, :]); bins = 100)
    f |> display
end
