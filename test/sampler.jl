using FractionalNeuralSampling
using DifferentialEquations
using CairoMakie
using ForwardDiff
using SpecialFunctions

# A sampler has the following components:
# 1. a noise process (e.g. levy noise, brownian noise)
# 2. a tspan and dt

function diffusion_sampler!(du, u, p, t, W)
    β, γ, b = p
    x, v = u
    du[1] = γ * b(x) + β * v + γ^(1 // 2) * W[1] # W is a Wiener process, so α = 2
    du[2] = β * b(x)
end

u0 = [0.01, 0]
tspan = (0.0, 5000.0)

π(x; k = 6) = exp(-(x)^k) / (2 * gamma((k + 1) / k))
# π(x) = (exp(-(x - 3)^2) + exp(-(x + 3)^2)) ./ sqrt(4 * pi)
∂ = ForwardDiff.derivative
function b(x)
    ∂(x -> log(π(x)), x) # propto -x for a normal distribution
    # -x
end
p = (0.5, 1.0, b)
prob = RODEProblem(diffusion_sampler!, u0, tspan, p; rand_prototype = [0.0])
@time sol = solve(prob, RandomEM(), dt = 1 / 100)
lines(sol.t, sol[1, :]; linewidth = 1)
hist(sol[1, :])
