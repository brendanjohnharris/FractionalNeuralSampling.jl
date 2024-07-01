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

include("Probabilities.jl")
include("NoiseProcesses.jl")
include("Densities.jl")
include("Samplers.jl")

@reexport using .NoiseProcesses
@reexport using .Samplers
@reexport using .Densities
end
