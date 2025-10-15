module FractionalNeuralSampling
using Reexport
using Preferences
using DifferentiationInterface
using RecursiveArrayTools
using ComponentArrays
using ForwardDiff
@reexport using SciMLBase
@reexport using StochasticDiffEq

function set_ad_backend!(new_backend::Union{DifferentiationInterface.AbstractADType,
                                            AbstractString})
    @set_preferences!("ad_backend"=>string(new_backend))
    @info("New autodiff backend set; restart your Julia session for this change to take effect!")
end
const AD_BACKEND = eval(Meta.parse(@load_preference("ad_backend",
                                                    "AutoForwardDiff()")))

"""
Divide a vector into views of length ND. Works with regular vectors, with optimizations for
ArrayPartitions and ComponentArrays, assuming each 'partition' has the same length ND
"""
function divide_dims(rand_vec::AbstractVector, ND)
    [view(rand_vec, ((i - 1) * ND + 1):(i * ND)) for i in 1:(length(rand_vec) ÷ ND)]
end
function divide_dims(rand_vec::ArrayPartition, ND) # ND unused, could check against size of randvec but might be slow
    return rand_vec.x # Assume each partition is one variable of length ND
end
function divide_dims(rand_vec::ComponentArray, ND) # ND unused
    return map(Base.Fix1(view, rand_vec), ComponentArrays.valkeys(rand_vec))
end

include("PowerOperator.jl")
include("Probabilities.jl")
include("NoiseProcesses.jl")
include("Densities.jl")
include("Boundaries.jl")
include("Window.jl")
include("Solvers.jl")
include("Samplers.jl")

@reexport using .NoiseProcesses
@reexport using .Densities
@reexport using .Boundaries
@reexport using .Samplers
@reexport using .Solvers

# * Extension placeholders
function samplingpower end
function samplingaccuracy end
function samplingefficiency end
function _samplingaccuracy end

export samplingpower, samplingaccuracy, samplingefficiency
end
