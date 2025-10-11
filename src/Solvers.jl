module Solvers
import ..FractionalNeuralSampling: Window
import StochasticDiffEq: StochasticDiffEqAlgorithm,
                         StochasticDiffEqMutableCache,
                         alg_cache, full_cache, jac_iter, rand_cache, ratenoise_cache,
                         perform_step!, is_split_step, is_diagonal_noise,
                         alg_compatible, DiffEqBase, SVector,
                         @cache, @muladd, @unpack, @..

import SpecialFunctions: gamma
using LinearAlgebra

export CaputoEM

abstract type FractionalAlgorithm <: StochasticDiffEqAlgorithm end

include("Solvers/CaputoEM.jl")

end
