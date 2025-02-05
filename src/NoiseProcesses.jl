module NoiseProcesses
using StableDistributions
using SciMLBase
using DiffEqNoiseProcess
using Random
using LinearAlgebra
using StaticArraysCore
import ..FractionalNeuralSampling.divide_dims
export LevyProcess, LevyProcess!

struct LevyNoise{inplace, T}
    α::T
    β::T
    σ::T
    μ::T
    ND::Integer # ? The number of dimensions to the noise process
end
function LevyNoise{inplace}(α, β = 0.0, σ = 1 / sqrt(2), μ = 0.0, ND = 1) where {inplace}
    Stable(α, β, σ, μ) # Check values
    LevyNoise{inplace, typeof(α)}(α, β, σ, μ, ND)
end
function LevyNoise(args...)
    LevyNoise{false}(args...)
end
function LevyNoise!(args...)
    LevyNoise{true}(args...)
end

dist(L::LevyNoise) = Stable(L.α, L.β, L.σ, L.μ)

# @inline function (L::LevyNoise{false, T})(rng::AbstractRNG)::T where {T}
#     rand(rng, dist(L))
# end

# @inline function (L::LevyNoise{false, T})(rng::AbstractRNG,
#                                           proto::AbstractArray{T})::AbstractArray{T} where {T <:
#                                                                                             Number}
#     rand(rng, dist(L), size(proto))
# end
# @inline function (L::LevyNoise{false, T})(rng::AbstractRNG,
#                                           proto::Type{S})::S where {T <:
#                                                                     Number,
#                                                                     S <:
#                                                                     StaticArraysCore.StaticArray
#                                                                     }
#     S(rand(rng, dist(L), size(S)))
# end
# @inline function (L::LevyNoise{false, T})(rng::AbstractRNG,
#                                           proto::S)::S where {T <: Number,
#                                                               S <:
#                                                               StaticArraysCore.StaticArray}
#     L(rng, S)
# end
# @inline function (L::LevyNoise{false, T})(rng::AbstractRNG,
#                                           proto::Type{T})::T where {T}
#     rand(rng, dist(L))
# end
@inline function (L::LevyNoise{true})(rng::AbstractRNG, rand_mat::AbstractMatrix)
    rand_vec = view(rand_mat, diagind(rand_mat))
    rand_vecs = divide_dims(rand_vec, L.ND) # * Add ND noise independently to each column (each variable)
    map(rand_vecs) do x
        randn!(rng, x) # * Choose a point from a spherical distribution
        x ./= norm(x) # * Normalize the vector
        x .*= rand(rng, dist(L)) # * Take a levy step in the chosen direction
    end
end
# @inline function (L::LevyNoise{true})(rng::AbstractRNG, rand_vec)
#     rand_vec .= Base.Broadcast.Broadcasted(_ -> rand(rng, dist(L)), ())
# end
# # This fallback works for GPUs because it doesn't assume we can pass an RNG
# @inline function (L::LevyNoise{true})(rng::AbstractRNG,
#                                       rand_vec::DiffEqNoiseProcess.GPUArraysCore.AbstractGPUArray)
#     rand!(rng, dist(L), rand_vec)
# end

# function (L::LevyNoise{false, T})(dW::T, W, dt, u, p, t, rng) where {T}
#     @fastmath L(rng) * abs(dt)^(1 / L.α)
# end
# function (L::LevyNoise{false, T})(dW::AbstractArray{T}, W, dt, u, p, t, rng) where {T}
#     @fastmath L(rng, dW) * abs(dt)^(1 / L.α)
# end
# function (L::LevyNoise{false, T})(dW::StaticArraysCore.SArray{N, T}, W, dt, u, p, t,
#                                   rng) where {N, T}
#     @fastmath L(rng, dW) * abs(dt)^(1 / L.α)
# end
function (L!::LevyNoise{true})(rand_mat, W, dt, u, p, t, rng)
    L!(rng, rand_mat)
    @fastmath rand_mat .*= abs(dt)^(1 / L!.α)
end

function LevyProcess(α, β = 0.0, σ = 1 / sqrt(2); μ = 0.0, t0 = 0.0, W0 = 0.0, Z0 = nothing,
                     ND = 1,
                     kwargs...)
    NoiseProcess{false}(t0, W0, Z0, LevyNoise{false}(α, β, σ, μ, ND), nothing; kwargs...)
end
function LevyProcess!(α, β = 0.0, σ = 1 / sqrt(2); μ = 0.0, t0 = 0.0, W0 = [0.0],
                      Z0 = nothing, ND = 1,
                      kwargs...)
    NoiseProcess{true}(t0, W0, Z0, LevyNoise{true}(α, β, σ, μ, ND), nothing; kwargs...)
end

LEVYPROCESS = NoiseProcess{A, B, C, D, E, F,
                           G} where {A, B, C, D, E, F, G <: LevyNoise}
LevyProblem = RODEProblem{A, B, C, D,
                          E} where {A, B, C, D, E <: LEVYPROCESS}

# ! Will want to throw an error to solve if anything other than RandomEM() is used.

end # module
