module TimeseriesToolsExt
using TimeseriesTools
using FractionalNeuralSampling
import FractionalNeuralSampling: samplingpower, samplingaccuracy, samplingefficiency,
                                 _samplingaccuracy

samplingpower(x::RegularTimeseries) = samplingpower(x, step(x))

"""
Taus in unit steps
"""
function samplingaccuracy(x::RegularTimeseries, 𝜋::AbstractDensity, τs::AbstractVector;
                          kwargs...)
    y = samplingaccuracy(parent(x), 𝜋, τs; kwargs...)
    return ToolsArray(y, 𝑡(τs .* samplingperiod(x)))
end

"""
Taus in time units
"""
function samplingefficiency(x::RegularTimeseries, 𝜋::AbstractDensity,
                            τs::AbstractVector = samplingperiod(x) .* (2:100);
                            kwargs...)
    dt = samplingperiod(x)
    _τs = round.(Int, τs ./ dt)
    Main.@infiltrate
    ΔI = _samplingaccuracy(x, 𝜋, _τs; kwargs...)
    vd = samplingpower(x, dt)
    se = map(x -> exp.(-x ./ vd), ΔI)
    return ToolsArray(se, 𝑡(τs))
end

end
