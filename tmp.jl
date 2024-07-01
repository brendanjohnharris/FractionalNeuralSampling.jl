module NoiseProcesses
using StableDistributions
using SciMLBase
using DiffEqNoiseProcess
using Random
using StaticArraysCore
export LevyProcess, LevyProcess!, LevyNoise, LevyNoise!

struct LevyNoise{T <: Real}
    stable::Stable{T}
end
struct LevyNoise!{T <: Real}
    stable::Stable{T}
end
const LevyNoises = Union{LevyNoise, LevyNoise!}

LevyNoise(α, β, σ, μ) = LevyNoise(Stable(α, β, σ, μ))
LevyNoise!(α, β, σ, μ) = LevyNoise!(Stable(α, β, σ, μ))

@inline function (L::LevyNoise{T})(rng::AbstractRNG)::T where {T}
    rand(rng, L.stable)
end

@inline function (L::LevyNoise{T})(rng::AbstractRNG,
                                   proto::AbstractArray{T})::AbstractArray{T} where {T <:
                                                                                     Number}
    rand(rng, L.stable, size(proto))
end
@inline function (L::LevyNoise{T})(rng::AbstractRNG,
                                   proto::Type{S})::S where {T <:
                                                             Number,
                                                             S <:
                                                             StaticArraysCore.StaticArray
                                                             }
    S(rand(rng, L.stable, size(S)))
end
@inline function (L::LevyNoise{T})(rng::AbstractRNG,
                                   proto::Type{T})::T where {T}
    rand(rng, L.stable)
end
@inline function (L::LevyNoise!)(rng::AbstractRNG, rand_vec::AbstractArray)
    rand!(rng, L.stable, rand_vec)
end
@inline function (L::LevyNoise!)(rng::AbstractRNG, rand_vec)
    rand_vec .= Base.Broadcast.Broadcasted(_ -> rand(rng, L.stable), ())
end
# This fallback works for GPUs because it doesn't assume we can pass an RNG
@inline function (L::LevyNoise!)(rng::AbstractRNG,
                                 rand_vec::DiffEqNoiseProcess.GPUArraysCore.AbstractGPUArray)
    rand!(rng, L.stable, rand_vec)
end

function (L::LevyNoise{T})(dW::T, W, dt, u, p, t, rng) where {T}
    @fastmath L(rng) * abs(dt)^(1 / L.stable.α)
end
function (L::LevyNoise{T})(dW::AbstractArray{T}, W, dt, u, p, t, rng) where {T}
    @fastmath L(rng, dW) * abs(dt)^(1 / L.stable.α)
end
function (L::LevyNoise{T})(dW::StaticArraysCore.SArray{N, T}, W, dt, u, p, t,
                           rng) where {N, T}
    @fastmath L(rng, dW) * abs(dt)^(1 / L.stable.α)
end
function (L!::LevyNoise!)(rand_vec, W, dt, u, p, t, rng)
    L!(rng, rand_vec)
    sqrtabsdt = @fastmath abs(dt)^(1 / L!.stable.α)
    rand_vec .*= sqrtabsdt
end
function LevyProcess(α, β = 0.0, σ = 1 / sqrt(2), μ = 0.0, t0 = 0.0, W0 = 0.0, Z0 = nothing;
                     kwargs...)
    NoiseProcess{false}(t0, W0, Z0, LevyNoise(α, β, σ, μ), nothing; kwargs...)
end
function LevyProcess!(α, β = 0.0, σ = 1 / sqrt(2), μ = 0.0, t0 = 0.0, W0 = [0.0]L,
                      Z0 = nothing;
                      kwargs...)
    NoiseProcess{true}(t0, W0, Z0, LevyNoise!(α, β, σ, μ), nothing; kwargs...)
end

LEVYPROCESS = NoiseProcess{A, B, C, D, E, F,
                           G} where {A, B, C, D, E, F, G <: LevyNoises}
LevyProblem = RODEProblem{A, B, C, D,
                          E} where {A, B, C, D, E <: LEVYPROCESS}

# ! Will want to throw an error to solve if anything other than RandomEM() is used.

end # module
