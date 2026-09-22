module NoiseProcesses
using StableDistributions
using SciMLBase
using DiffEqNoiseProcess
using Random
using LinearAlgebra
using StaticArraysCore
import ..FractionalNeuralSampling: divide_dims, serial_fftw
export LevyProcess, LevyProcess!

struct LevyNoise{inplace, T}
    α::T
    β::T
    σ::T
    μ::T
    ND::Int # ? The number of dimensions to the noise process
end
function LevyNoise{inplace}(α, β = 0.0, σ = 1, μ = 0.0, ND = 1) where {inplace}
    Stable(α, β, σ, μ) # Validate the parameters up front
    return LevyNoise{inplace, typeof(α)}(α, β, σ, μ, ND)
end
function LevyNoise(args...)
    return LevyNoise{false}(args...)
end
function LevyNoise!(args...)
    return LevyNoise{true}(args...)
end

dist(L::LevyNoise) = Stable(L.α, L.β, L.σ, L.μ)

@inline function (L::LevyNoise{true})(rng::AbstractRNG, rand_vec::AbstractVector)
    rand_vecs = divide_dims(rand_vec, L.ND) # * Add ND noise independently to each column (each variable)
    d = dist(L)
    return foreach(rand_vecs) do x
        randn!(rng, x) # * Choose a point from a spherical distribution
        x ./= norm(x) # * Normalize the vector
        x .*= rand(rng, d) # * Take a levy step in the chosen direction
    end
end
@inline function (L::LevyNoise{true})(rng::AbstractRNG, rand_mat::AbstractMatrix)
    rand_vec = view(rand_mat, diagind(rand_mat))
    return L(rng, rand_vec)
end

function (L!::LevyNoise{true})(rand_mat, W, dt, u, p, t, rng)
    L!(rng, rand_mat)
    return @fastmath rand_mat .*= abs(dt)^(1 / L!.α)
end

"""
Out-of-place Lévy noise is not implemented: `LevyNoise{false}` has no sampling method, so
the process would only fail once solved. Use [`LevyProcess!`](@ref).
"""
function LevyProcess(
        α, β = 0.0, σ = 1; μ = 0.0, t0 = 0.0, W0 = 0.0, Z0 = nothing,
        ND = 1,
        kwargs...
    )
    return throw(ArgumentError("Out-of-place Lévy noise is not implemented; use `LevyProcess!`"))
end
"""
    LevyProcess!(α, β = 0.0, σ = 1; μ = 0.0, t0 = 0.0, W0 = [0.0], ND = 1, kwargs...)

An α-stable (Lévy) noise process, usable wherever a `DiffEqNoiseProcess` is expected.

Increments are drawn isotropically: a direction uniform on the sphere, and a magnitude from
`Stable(α, β, σ, μ)`. The increment over a step is scaled by ``|dt|^{1/α}``, which makes
the process self-similar with exponent 1/α. At α = 2 the increments are Gaussian and the
process is a rescaled Brownian motion.

# Arguments
- `α`: stability, in (0, 2]; smaller α gives heavier tails
- `β`: skewness
- `σ`: scale
- `μ`: location
- `W0`: initial value
- `ND`: number of dimensions to which noise is added independently

The out-of-place [`LevyProcess`](@ref) is not implemented.
"""
function LevyProcess!(
        α, β = 0.0, σ = 1; μ = 0.0, t0 = 0.0, W0 = [0.0],
        Z0 = nothing, ND = 1,
        kwargs...
    )
    return NoiseProcess{true}(t0, W0, Z0, LevyNoise{true}(α, β, σ, μ, ND), nothing; kwargs...)
end

LEVYPROCESS = NoiseProcess{
    A, B, C, D, E, F,
    G,
} where {A, B, C, D, E, F, G <: LevyNoise}
LevyProblem = RODEProblem{
    A, B, C, D,
    E,
} where {A, B, C, D, E <: LEVYPROCESS}

# ! Will want to throw an error to solve if anything other than EM() is used.

include("LFSM.jl")
end # module
