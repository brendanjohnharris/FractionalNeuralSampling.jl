module Solvers
using UnPack
import ..FractionalNeuralSampling: Window, roll!
import ..Boundaries: wrap_integrator_cache!
# Internals moved into the OrdinaryDiffEq.jl monorepo in the v7 wave; import from owners
import StochasticDiffEqCore: StochasticDiffEqAlgorithm,
    StochasticDiffEqMutableCache,
    alg_cache, jac_iter, perform_step!,
    is_split_step, alg_compatible, isadaptive
import DiffEqBase: DiffEqBase, full_cache, rand_cache, ratenoise_cache,
    is_diagonal_noise, @..
import MuladdMacro: @muladd

import SpecialFunctions: gamma
using LinearAlgebra

export CaputoEM, MultiCaputoEM, PositionalCaputoEM

abstract type FractionalAlgorithm <: StochasticDiffEqAlgorithm end

# The L1 weights and the Γ(2-β)Δt^(β-1) correction are built once, for a uniform grid, so
# these are fixed-step methods. `StochasticDiffEqAlgorithm` already defaults to `false`;
# stated here so that the assumption survives a change of supertype.
isadaptive(::FractionalAlgorithm) = false

include("Solvers/CaputoEM.jl")
include("Solvers/MultiCaputoEM.jl")
include("Solvers/PositionalCaputoEM.jl")

end
