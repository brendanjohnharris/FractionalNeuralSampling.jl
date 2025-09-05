using FractionalNeuralSampling
using StochasticDiffEq
using Distributions

begin # * Define the sampler
    𝜋 = MixtureModel([Normal(-3, 1), Normal(3, 1)]) |> Density
    η = 1
    u0 = [0.0]
    tspan = 1000.00
    S = OLE(;
            η,
            u0,
            𝜋,
            tspan)
end

begin # * Run a simulation
    sol = solve(S, EM(); dt = 0.01)
end

begin # * Extract distribution divergences from time-shifted subsets of data, with a certain overlap (buffer). Get result as a function of time
end

begin # * Extract driving energy for same buffers
end
