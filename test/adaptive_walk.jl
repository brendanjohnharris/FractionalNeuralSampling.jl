using CairoMakie
using Foresight
using DifferentialEquations
using FractionalNeuralSampling
using Distributions
using LinearAlgebra
using TimeseriesTools

import FractionalNeuralSampling: Density, divide_dims
set_theme!(foresight(:physics))

domain = Boundaries.domain(boundaries)
domain = prod(Chebyshev.(domain))
x, y = Fun(domain)
xt, yt = [0.0, 0.0]
τ = 1
σ = 1
K = exp.(-((x)^2 + (y)^2) / (2 * σ^2)) / τ
Dx = Derivative(domain, [1, 0])
Dy = Derivative(domain, [0, 1])
∇xK = Dx * K # For a given xt, yt = [0.0, 0.0]
∇yK = Dy * K # For a given

s∇xK = recenter(∇xK, [x[1], x[2]])
s∇yK = recenter(∇yK, [x[1], x[2]])

function adaptive_f!(du, u, p, t)
    (γ,), 𝜋 = p
    x, v = divide_dims(u, length(u) ÷ 2)
    ∇V = .-gradlogdensity(𝜋)(x)
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= -∇V + [s∇xK s∇yK]
end
function adaptive_g!(du, u, p, t)
    (γ,), 𝜋 = p
    x, v = divide_dims(u, length(u) ÷ 2)
    dx, du = divide_dims(du, length(du) ÷ 2)
    dx .= sqrt(2.0 * γ)
end

function AdaptiveSampler(;
                         tspan, γ, u0 = [0.0 0.0 0.0 0.0],
                         boundaries = nothing,
                         noise_rate_prototype = zeros(size(u0)),
                         𝜋 = Density(default_density(first(u0))),
                         noise = WienerProcess!(0.0,
                                                zeros(length(u0), length(u0))),
                         kwargs...)
    Sampler(adaptive_f!, adaptive_g!; kwargs..., callback = boundaries, u0,
            noise_rate_prototype, noise,
            tspan, p = ((γ,), 𝜋))
end

begin # * Simulate and plot
    boundaries = PeriodicBox(-5 .. 5, -5 .. 5)
    D = FractionalNeuralSampling.Density(MvNormal([0.0, 0.0], I(2)))
    S = AdaptiveSampler(; tspan = 10.0,
                        γ = 1.0,
                        𝜋 = D,
                        boundaries = boundaries(),
                        seed = 42)
    sol = solve(S, EM(); dt = 0.01)
    x = sol[1, :]
    y = sol[2, :]
    t = sol.t
end
begin
    f = Figure()
    ax = Axis(f[1, 1])
    lines!(ax, t, x)
    lines!(ax, t, y)
    f
end

begin # * Choose a basis for the adaptive potential. Gives periodic approximation of the derivative
    ∇xK(0.0, 0.4)
    ∇yK(0.0, 0.5)
end
begin # * To get the gradient relative to the walkers position
    function recenter(f, x)
        compose(f, Base.Fix2(-, x))
    end
    s∇xK = recenter(∇xK, [0.6, 0.6])
    s∇xK([5.1, 0.6])
end
