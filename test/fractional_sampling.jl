import FractionalNeuralSampling as FNS
using FractionalNeuralSampling
using CairoMakie
using DifferentialEquations

function f3(u, p, t, W)
    2u * sin(W)
end
u0 = 1.00
tspan = (0.0, 5.0)
prob = RODEProblem(f3, u0, tspan)
sol = solve(prob, RandomEM(), dt = 1 / 100)
plot(sol)

function f(du, u, p, t, W)
    du[1] = 2u[1] * sin(W[1] - W[2])
    du[2] = -2u[2] * cos(W[1] + W[2])
end
u0 = [1.00; 1.00]
tspan = (0.0, 5.0)
prob = RODEProblem(f, u0, tspan)
sol = solve(prob, RandomEM(), dt = 1 / 100)
plot(sol)
