module Solvers
using UnPack
import ..FractionalNeuralSampling: Window
import ..Boundaries: wrap_integrator_cache!
import StochasticDiffEq: StochasticDiffEqAlgorithm,
                         StochasticDiffEqMutableCache,
                         alg_cache, full_cache, jac_iter, rand_cache, ratenoise_cache,
                         perform_step!, is_split_step, is_diagonal_noise,
                         alg_compatible, DiffEqBase, SVector,
                         @cache, @muladd, @..

import SpecialFunctions: gamma
using LinearAlgebra

export CaputoEM, MultiCaputoEM, PositionalCaputoEM

abstract type FractionalAlgorithm <: StochasticDiffEqAlgorithm end

include("Solvers/CaputoEM.jl")
include("Solvers/MultiCaputoEM.jl")
include("Solvers/PositionalCaputoEM.jl")

end
