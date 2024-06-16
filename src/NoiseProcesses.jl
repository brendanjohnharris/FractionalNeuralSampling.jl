module NoiseProcesses
using StableDistributions
using SciMLBase
using DiffEqNoiseProcess
using Random
using StaticArraysCore
export LevyProcess, LevyProcess!

struct LEVY_NOISE_DIST{inplace, T}
    stable::Stable{T}
end

function LEVY_NOISE_DIST{iip}(α, β, σ, μ) where {iip}
    LEVY_NOISE_DIST{iip, Float64}(Stable(α, β, σ, μ))
end

@inline function (L::LEVY_NOISE_DIST{false, T})(rng::AbstractRNG)::T where {T}
    rand(rng, L.stable)
end

@inline function (L::LEVY_NOISE_DIST{false, T})(rng::AbstractRNG,
                                                proto::AbstractArray{T})::AbstractArray{T} where {T <:
                                                                                                  Number}
    rand(rng, L.stable, size(proto))
end
@inline function (L::LEVY_NOISE_DIST{false, T})(rng::AbstractRNG,
                                                proto::Type{S})::S where {T <:
                                                                          Number,
                                                                          S <:
                                                                          StaticArraysCore.StaticArray
                                                                          }
    S(rand(rng, L.stable, size(S)))
end
@inline function (L::LEVY_NOISE_DIST{false, T})(rng::AbstractRNG,
                                                proto::Type{T})::T where {T}
    rand(rng, L.stable)
end
@inline function (L::LEVY_NOISE_DIST{true})(rng::AbstractRNG, rand_vec::AbstractArray)
    rand!(rng, L.stable, rand_vec)
end
@inline function (L::LEVY_NOISE_DIST{true})(rng::AbstractRNG, rand_vec)
    rand_vec .= Base.Broadcast.Broadcasted(_ -> rand(rng, L.stable), ())
end
# This fallback works for GPUs because it doesn't assume we can pass an RNG
@inline function (L::LEVY_NOISE_DIST{true})(rng::AbstractRNG,
                                            rand_vec::DiffEqNoiseProcess.GPUArraysCore.AbstractGPUArray)
    rand!(rng, L.stable, rand_vec)
end

function (L::LEVY_NOISE_DIST{false})(dW, W, dt, u, p, t, rng)
    if dW isa AbstractArray && !(dW isa StaticArraysCore.SArray)
        return @fastmath L(rng, dW) * abs(dt)^(1 / L.stable.α)
    else
        return @fastmath L(rng, typeof(dW)) *
                         abs(dt)^(1 / L.stable.α)
    end
end
function (L!::LEVY_NOISE_DIST{true})(rand_vec, W, dt, u, p, t, rng)
    L!(rng, rand_vec)
    sqrtabsdt = @fastmath abs(dt)^(1 / L!.stable.α)
    rand_vec .*= sqrtabsdt
end
function LevyProcess(α, β = 0.0, σ = 1 / sqrt(2), μ = 0.0, t0 = 0.0, W0 = 0.0, Z0 = nothing;
                     kwargs...)
    NoiseProcess{false}(t0, W0, Z0, LEVY_NOISE_DIST{false}(α, β, σ, μ), nothing; kwargs...)
end
function LevyProcess!(α, β = 0.0, σ = 1 / sqrt(2), μ = 0.0, t0 = 0.0, W0 = [0.0]L,
                      Z0 = nothing;
                      kwargs...)
    NoiseProcess{true}(t0, W0, Z0, LEVY_NOISE_DIST{true}(α, β, σ, μ), nothing; kwargs...)
end

LEVYPROCESS = NoiseProcess{A, B, C, D, E, F,
                           G} where {A, B, C, D, E, F, G <: LEVY_NOISE_DIST}
LevyProblem = RODEProblem{A, B, C, D,
                          E} where {A, B, C, D, E <: LEVYPROCESS}

# ! Will want to throw an error to solve if anything other than RandomEM() is used.

end # module
