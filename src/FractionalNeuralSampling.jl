module FractionalNeuralSampling
using Reexport
using Preferences
import FFTW
using DifferentiationInterface
using RecursiveArrayTools
using ComponentArrays
using ForwardDiff
@reexport using SciMLBase
@reexport using StochasticDiffEqLowOrder # Reexports StochasticDiffEqCore; `EM` is the only upstream solver used

"""
    set_ad_backend!(backend)

Set the automatic differentiation backend used for density gradients, as a
`DifferentiationInterface` type or a string naming one.

The choice is stored as a preference, so it survives across sessions, and is read when the
package is loaded; a change therefore takes effect only after a restart. Defaults to
`AutoForwardDiff()`.

```julia
FractionalNeuralSampling.set_ad_backend!("AutoEnzyme()")
```
"""
function set_ad_backend!(
        new_backend::Union{
            DifferentiationInterface.AbstractADType,
            AbstractString,
        }
    )
    @set_preferences!("ad_backend" => string(new_backend))
    return @info("New autodiff backend set; restart your Julia session for this change to take effect!")
end
const AD_BACKEND = eval(
    Meta.parse(
        @load_preference(
            "ad_backend",
            "AutoForwardDiff()"
        )
    )
)

"""
    serial_fftw(f)

Run `f` with FFTW restricted to a single thread, restoring the previous count afterwards.

FFTW segfaults on the in-place plans used here when it is multithreaded
(JuliaMath/FFTW.jl#236), taking the session down rather than throwing. Every spectral
transform in the package is planned and executed inside this, so the default multithreaded
FFTW a user arrives with is safe. A plan carries its thread count from construction, so
transforms applied later (the adaptation kernel's, each step) stay serial too.
"""
function serial_fftw(f)
    n = FFTW.get_num_threads()
    n == 1 && return f()
    FFTW.set_num_threads(1)
    return try
        f()
    finally
        FFTW.set_num_threads(n)
    end
end

"""
Divide a vector into views of length ND. Works with regular vectors, with optimizations for
ArrayPartitions and ComponentArrays, assuming each 'partition' has the same length ND
"""
function divide_dims(rand_vec::AbstractVector, ND)
    return [view(rand_vec, ((i - 1) * ND + 1):(i * ND)) for i in 1:(length(rand_vec) ÷ ND)]
end
function divide_dims(rand_vec::ArrayPartition, ND) # ND unused, could check against size of randvec but might be slow
    return rand_vec.x # Assume each partition is one variable of length ND
end
function divide_dims(rand_vec::ComponentArray, ND) # ND unused
    return map(Base.Fix1(view, rand_vec), ComponentArrays.valkeys(rand_vec))
end

"""
The first of the views returned by [`divide_dims`](@ref), without materialising the rest
"""
first_dims(u::AbstractVector, ND) = view(u, 1:ND)
first_dims(u::ArrayPartition, ND) = first(u.x)
first_dims(u::ComponentArray, ND) = view(u, first(ComponentArrays.valkeys(u)))

include("PowerOperator.jl")
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
import .Boundaries: domain

# * Extension placeholders (defined in TimeseriesToolsExt)
"""
    samplingpower(x, dt; p = 2)
    samplingpower(x::RegularTimeseries; p = 2)

The p-variation of the increments of `x` per unit time; at `p = 2` a rate of quadratic
variation.

Measures how much ground a sampler covers rather than how well it covers it, so it
separates a trajectory built from many small steps from one built from a few large jumps.

Provided by the `TimeseriesTools` extension, so load `TimeseriesTools` to use it.
"""
function samplingpower end

"""
    samplingaccuracy(x, 𝜋::AbstractDensity; domain = nothing)
    samplingaccuracy(x, 𝜋::AbstractDensity, τs::AbstractVector; p = 0, domain = nothing)

The Wasserstein-1 distance between the samples `x` and the target `𝜋`, computed by
comparing sorted samples against the quantiles of `𝜋`.

Given `τs`, `x` is first cut into windows of each length τ and the distance is returned per
window, which shows how the estimate converges with sample size; `p` sets the overlap
between windows, and is zero by default. `domain` restricts the comparison to samples
inside it, which matters where a sampler makes excursions far outside the support of `𝜋`.

Provided by the `TimeseriesTools` extension, so load `TimeseriesTools` to use it. With a
`RegularTimeseries`, `τs` is given in unit steps and the result is returned over time.
"""
function samplingaccuracy end
function _samplingaccuracy end

export samplingpower, samplingaccuracy
end
