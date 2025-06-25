using DomainSets
using ModelingToolkit

begin
    @parameters t
    @variables x(..)
    Di = Differential(t)
    Ix = Integral(t in DomainSets.ClosedInterval(0, t))
    eq = Di(x(t)) + 2 * x(t) + 5 * Ix(x(t)) ~ 1
    bcs = [x(0.0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]
    @named pde_system = PDESystem(eq, bcs, domains, [t], [x(t)])
    pde_system |> ModelingToolkit.latexify
end

chain = Lux.Chain(Lux.Dense(1, 15, Lux.σ), Lux.Dense(15, 1))

strategy_ = QuadratureTraining()
discretization = PhysicsInformedNN(chain,
                                   strategy_)
prob = NeuralPDE.discretize(pde_system, discretization)
callback = function (p, l)
    println("Current loss is: $l")
    return false
end
res = Optimization.solve(prob, BFGS(); maxiters = 100)
