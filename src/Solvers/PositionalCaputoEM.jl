"""
Caputo-EM for the first variable; plain EM for the rest.

System:
    D^{β1}_t x₁(t) = f₁(x) + g₁(x) ξ(t),      0 < β₁ ≤ 1
    d xᵢ(t)        = fᵢ(x) dt + gᵢ(x) dW(t),  i = 2,…,n
Uses L1 EM-based approximation for the Caputo term of x₁ only.
"""
struct PositionalCaputoEM{T} <: FractionalAlgorithm
    β1::T
    nhist::Int
    function PositionalCaputoEM{T}(β1::T, nhist::Int) where {T}
        (0 < β1 <= 1) || throw(ArgumentError("β₁ must be in (0, 1]"))
        new{T}(β1, nhist)
    end
end

function PositionalCaputoEM(β1::T, nhist::Int) where {T <: AbstractFloat}
    PositionalCaputoEM{T}(β1, nhist)
end

struct PositionalCaputoEMCache{T, uType <: AbstractArray{<:T}, rateType, rateNoiseType} <:
       StochasticDiffEqMutableCache
    u::uType                   # Current state
    uhist1::Window{T}          # History of Δx₁ only (length nhist)
    weights1::Vector{T}        # L1 weights for β₁ (length nhist)
    correction1::T             # Caputo correction factor for x₁
    tmp::uType
    rtmp1::rateType
    rtmp2::rateNoiseType
end

nhist(A::PositionalCaputoEM) = A.nhist
nhist(C::PositionalCaputoEMCache) = length(C.uhist1)

function full_cache(c::PositionalCaputoEMCache)
    # Return the same shape of tuple as other algs typically do
    tuple(c.u, c.uhist1, c.weights1, c.correction1, c.tmp, c.rtmp1)
end
jac_iter(::PositionalCaputoEMCache) = tuple()
rand_cache(c::PositionalCaputoEMCache) = tuple()
ratenoise_cache(c::PositionalCaputoEMCache) = tuple(c.rtmp2)

# -- cache builder
function alg_cache(alg::PositionalCaputoEM, prob, u, ΔW, ΔZ, p,
                   rate_prototype,
                   noise_rate_prototype,
                   jump_rate_prototype,
                   ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits},
                   ::Type{tTypeNoUnits},
                   uprev, f, t, dt,
                   ::Type{Val{true}}) where {uEltypeNoUnits,
                                             uBottomEltypeNoUnits,
                                             tTypeNoUnits}
    tmp = zero(u)
    rtmp1 = zero(rate_prototype)
    rtmp2 = noise_rate_prototype === nothing ? nothing : zero(noise_rate_prototype)

    # store history only for x₁ increments (scalars)
    uhist1 = Window(zero(eltype(u)), nhist(alg))
    push!(uhist1, u[1] - uprev[1])

    # Caputo ingredients for x₁
    β1 = convert(eltype(u), alg.β1)
    correction1 = caputo_factor(β1, dt)
    weights1 = caputo_weights(β1, nhist(alg))

    PositionalCaputoEMCache(u, uhist1, weights1, correction1, tmp, rtmp1, rtmp2)
end

function wrap_integrator_cache!(C::PositionalCaputoEMCache, u, uprev)
    C.uhist1[end] = u[1] - uprev[1] # Just correct history for x
end

# -- single step
@muladd function perform_step!(integrator, cache::PositionalCaputoEMCache)
    @unpack uhist1, weights1, correction1, tmp, rtmp1, rtmp2 = cache
    @unpack t, dt, uprev, u, W, P, c, p = integrator

    # Drift f(xₙ)
    integrator.f(rtmp1, uprev, p, t)

    # Deterministic update:
    #   x₁ uses Caputo correction; others (i≥2) are plain EM.
    u[1] = uprev[1] + correction1 * dt * rtmp1[1]
    @inbounds @simd for i in 2:length(u)
        u[i] = uprev[i] + dt * rtmp1[i]
    end

    # Split-step choice for g
    u_choice = is_split_step(integrator.alg) ? u : uprev

    # Diffusion g(xₙ)
    integrator.g(rtmp2, u_choice, p, t)

    # Jump noise (additive term into tmp)
    if P !== nothing
        c(tmp, uprev, p, t, P.dW, nothing)
    end

    # Stochastic update:
    # scale noise; x₁ gets correction1, others are standard EM
    if is_diagonal_noise(integrator.sol.prob)
        @.. rtmp2 *= W.dW
        if P !== nothing
            u[1] += (rtmp2[1] + tmp[1]) * correction1
            @inbounds @simd for i in 2:length(u)
                u[i] += rtmp2[i] + tmp[i]
            end
        else
            u[1] += rtmp2[1] * correction1
            @inbounds @simd for i in 2:length(u)
                u[i] += rtmp2[i]
            end
        end
    else
        # non-diagonal noise: rtmp1 = rtmp2 * ΔW
        mul!(rtmp1, rtmp2, W.dW)
        if P !== nothing
            u[1] += (rtmp1[1] + tmp[1]) * correction1
            @inbounds @simd for i in 2:length(u)
                u[i] += rtmp1[i] + tmp[i]
            end
        else
            u[1] += rtmp1[1] * correction1
            @inbounds @simd for i in 2:length(u)
                u[i] += rtmp1[i]
            end
        end
    end

    # Caputo memory only for x₁
    u1 = u[1]
    @inbounds @simd for j in eachindex(weights1, uhist1)
        u1 = muladd(-weights1[j], uhist1[j], u1)
    end
    u[1] = u1

    # Roll x₁ history
    push!(uhist1, u[1] - uprev[1])

    return nothing
end

# Traits
alg_compatible(::DiffEqBase.AbstractSDEProblem, ::PositionalCaputoEM) = true
