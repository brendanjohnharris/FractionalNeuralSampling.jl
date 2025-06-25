using DifferentialEquations, FractionalDiffEq, CairoMakie
import FractionalDiffEq: FODESystem

h = 0.005
alpha = [0.9, 1]
x0 = [-1.0, 0.0]
tspan = [0, 100]
function Duffing!(du, u, p, t)
    α = -0.5
    du[1] = u[2]
    du[2] = -α * u[2] + (u[1] - u[1]^3)
end
prob = FODEProblem(Duffing!, alpha, x0, tspan)
sol = solve(prob, GL(), dt = h)
lines(sol[1, 1:10:end])
