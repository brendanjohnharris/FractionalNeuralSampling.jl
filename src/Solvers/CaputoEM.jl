"""
Solves the Caputo differential equation for fractional order 0 < β ≤ 1 with the L1 EM-based
approximation.

Dᵝₜx(t) = f(x) + g(x) ξ(t)

where ξ(t) is some noise process.

For β = 1 this reduces to the standard Euler-Maruyama method.
"""
struct CaputoEM{T} <: FractionalAlgorithm
    β::T
    nhist::Int
    function CaputoEM{T}(β::T, nhist::Int) where {T}
        if !(0 < β <= 1)
            throw(ArgumentError("Fractional order β must be in (0, 1]"))
        end
        new{T}(β, nhist)
    end
end # No split steps for now since we think noise is just additive.
function CaputoEM(β::T, nhist::Int) where {T <: AbstractFloat}
    CaputoEM{T}(β, nhist)
end

struct CaputoEMCache{T, uType <: AbstractArray{<:T}, rateType, rateNoiseType} <:
       StochasticDiffEqMutableCache
    u::uType # Current state
    uhist::Window{uType} # Circular buffer for history
    weights::Vector{T} # Weights for history terms
    correction::T # Correction factor
    tmp::uType
    rtmp1::rateType
    rtmp2::rateNoiseType
end
nhist(C::CaputoEM) = C.nhist
nhist(C::CaputoEMCache) = length(C.uhist)

full_cache(c::CaputoEM) = tuple(c.u, c.uhist, c.weights, c.correction, c.tmp, c.rtmp1)
jac_iter(c::CaputoEM) = tuple()
rand_cache(c::CaputoEM) = tuple()
ratenoise_cache(c::CaputoEM) = tuple(c.rtmp2)

caputo_factor(β::AbstractFloat, Δt::AbstractFloat) = gamma(2 - β) * Δt^(β - 1)

"""
The caputo weights for j = 1...n. THe weight for j = 0 is always 1, and extracted as a
corrected EM update by the solver
"""
function caputo_weights(β::AbstractFloat, n::Int)
    # w_j = (j + 1)^(1-β) - j^(1-β) # for j = 1, 2, ..., n
    # j = 1 means 1 time step in the past
    w = range(n, 1, step = -1)
    @. (w + 1)^(1 - β) - w^(1 - β)
end

function alg_cache(alg::CaputoEM, prob, u, ΔW, ΔZ, p,
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
    if noise_rate_prototype !== nothing
        rtmp2 = zero(noise_rate_prototype)
    else
        rtmp2 = nothing
    end
    uhist = Window(u, nhist(alg))
    push!(uhist, u - uprev) # History holds differenced values
    β = convert(eltype(u), alg.β)
    correction = caputo_factor(β, dt)
    weights = caputo_weights(β, nhist(alg))
    CaputoEMCache(u, uhist, weights, correction, tmp, rtmp1, rtmp2)
end

function wrap_integrator_cache!(C::CaputoEMCache, u, uprev)
    C.uhist[end] .= u .- uprev
end

@muladd function perform_step!(integrator, cache::CaputoEMCache)
    @unpack uhist, weights, correction, tmp, rtmp1, rtmp2 = cache # cache.u gets updated somewhere
    @unpack t, dt, uprev, u, W, P, c, p = integrator

    # * We are solving Dx = f(x) + g(x) dW
    # c = Γ(2-β)(Δt)ᵝ⁻¹ # Correction factor
    # w_j = (j + 1)^(1-β) - j^(1-β) # for j = 1, 2, ..., n
    # xₙ₊₁ = xₙ
    #      + c*f(xₙ)*Δt # EM drift
    #      + c*g(xₙ)*ηₙ*√[Δt] # EM diffusion
    #      - ∑ⱼ₌₁ wⱼ⁽ⁿ⁾ Δxₙ # History update.
    # ! Check noise timestep scaling

    # * Deterministic update
    # * We can split this into an EM update (scaled by c) and a history contribution
    integrator.f(rtmp1, uprev, p, t) # * Mutate the drift term f(xₙ)

    @.. u = uprev + correction * dt * rtmp1 # c * f(xₙ) * Δt

    # * Split step ignore
    if is_split_step(integrator.alg)
        u_choice = u
    else
        u_choice = uprev
    end

    # * Do noise coefficient
    integrator.g(rtmp2, u_choice, p, t) # * Update g(xₙ)

    # ! Jump noise
    if P !== nothing
        c(tmp, uprev, p, t, P.dW, nothing)
    end

    # * Apply random noise update
    if is_diagonal_noise(integrator.sol.prob)
        @.. rtmp2 *= W.dW # * Timestep scaling in the noise func. sqrt(Δt) for brownian motion
        if P !== nothing
            @.. u += (rtmp2 + tmp) * correction # c*g(xₙ)*ηₙ*√[Δt]
        else
            @.. u += rtmp2 * correction
        end
    else
        mul!(rtmp1, rtmp2, W.dW)
        if P !== nothing
            @.. u += (rtmp1 + tmp) * correction
        else
            @.. u += rtmp1 * correction
        end
    end

    # * Apply history update
    # * Each element of the history can be a vector or similar
    # * We are here assuming all derivatives are of the same fractional order
    for (dx, w) in zip(uhist, weights)
        @. u -= w .* dx
    end
    # * Roll history
    push!(uhist, u - uprev)

    return nothing
end

# * Traits
alg_compatible(prob::DiffEqBase.AbstractSDEProblem, alg::CaputoEM) = true
