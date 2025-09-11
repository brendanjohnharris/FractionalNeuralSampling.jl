module TimeseriesToolsExt
using TimeseriesTools
using FractionalNeuralSampling
import FractionalNeuralSampling: samplingpower, samplingaccuracy

samplingpower(x::RegularTimeseries) = samplingpower(x, step(x))

function samplingaccuracy(x::RegularTimeseries, 𝜋::AbstractDensity, τs::AbstractVector;
                          dt = step(x), kwargs...)
    y = samplingaccuracy(parent(x), 𝜋, τs; kwargs...)
    return ToolsArray(y, 𝑡(τs .* step(x)))
end

end
