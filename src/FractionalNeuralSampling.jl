module FractionalNeuralSampling
using Reexport
using Preferences
using DifferentiationInterface

function set_ad_backend!(new_backend::Union{DifferentiationInterface.AbstractADType,
                                            AbstractString})
    @set_preferences!("ad_backend"=>string(new_backend))
    @info("New autodiff backend set; restart your Julia session for this change to take effect!")
end
const AD_BACKEND = eval(Meta.parse(@load_preference("ad_backend",
                                                    "AutoForwardDiff()")))

function divide_dims(rand_vec, ND) # Divide a vector into views of length ND
    [view(rand_vec, ((i - 1) * ND + 1):(i * ND)) for i in 1:(length(rand_vec) ÷ ND)]
end

include("Probabilities.jl")
include("NoiseProcesses.jl")
include("Densities.jl")
include("Boundaries.jl")
include("Samplers.jl")

@reexport using .NoiseProcesses
@reexport using .Densities
@reexport using .Boundaries
@reexport using .Samplers
end
