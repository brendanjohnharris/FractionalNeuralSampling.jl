"""
Solves the Caputo differential equation with variable-wise fractional orders 0 < βᵢ ≤ 1
using the L1 EM-based approximation.

Dᵝⁱₜxᵢ(t) = fᵢ(x) + gᵢ(x) ξ(t)

where each component xᵢ can have its own fractional order βᵢ.
"""
struct MultiCaputoEM{T} <: FractionalAlgorithm
    β::Vector{T}  # Fractional order for each variable
    nhist::Int
    function MultiCaputoEM{T}(β::Vector{T}, nhist::Int) where {T}
        if !all(0 < b <= 1 for b in β)
            throw(ArgumentError("All fractional orders β must be in (0, 1]"))
        end
        new{T}(β, nhist)
    end
end

function MultiCaputoEM(β::Vector{T}, nhist::Int) where {T <: AbstractFloat}
    MultiCaputoEM{T}(β, nhist)
end

# Convenience constructor for uniform β across all variables
function MultiCaputoEM(β::T, nvars::Int, nhist::Int) where {T <: AbstractFloat}
    MultiCaputoEM(fill(β, nvars), nhist)
end

struct MultiCaputoEMCache{T, uType <: AbstractArray{<:T}, rateType, rateNoiseType} <:
       StochasticDiffEqMutableCache
    u::uType # Current state
    uhist::Window{uType} # Circular buffer for history
    weights::Matrix{T} # Weights for history terms (nvars × nhist)
    correction::Vector{T} # Correction factor for each variable
    tmp::uType
    rtmp1::rateType
    rtmp2::rateNoiseType
end

nhist(C::MultiCaputoEM) = C.nhist
nhist(C::MultiCaputoEMCache) = length(C.uhist)

function full_cache(c::MultiCaputoEMCache)
    tuple(c.u, c.uhist, c.weights, c.correction, c.tmp, c.rtmp1)
end
jac_iter(c::MultiCaputoEMCache) = tuple()
rand_cache(c::MultiCaputoEMCache) = tuple()
ratenoise_cache(c::MultiCaputoEMCache) = tuple(c.rtmp2)

"""
Compute weights matrix where each row corresponds to a different variable's β.
"""
function caputo_weights_multiorder(β::Vector{T}, n::Int) where {T}
    nvars = length(β)
    weights = Matrix{T}(undef, nvars, n)
    for (i, βᵢ) in enumerate(β)
        weights[i, :] = caputo_weights(βᵢ, n)
    end
    return weights
end

function alg_cache(alg::MultiCaputoEM, prob, u, ΔW, ΔZ, p,
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

    # Convert β vector to element type of u
    β = convert.(eltype(u), alg.β)

    # Check that β length matches state dimension
    nvars = length(u)
    if length(β) != nvars
        throw(DimensionMismatch("Length of β ($(length(β))) must match state dimension ($nvars)"))
    end

    # Compute correction factors for each variable
    correction = [caputo_factor(βᵢ, dt) for βᵢ in β]

    # Compute weights matrix
    weights = caputo_weights_multiorder(β, nhist(alg))

    MultiCaputoEMCache(u, uhist, weights, correction, tmp, rtmp1, rtmp2)
end

function wrap_integrator_cache!(C::MultiCaputoEMCache, u, uprev)
    C.uhist[end] .= u .- uprev
end

@muladd function perform_step!(integrator, cache::MultiCaputoEMCache)
    @unpack uhist, weights, correction, tmp, rtmp1, rtmp2 = cache
    @unpack t, dt, uprev, u, W, P, c, p = integrator

    # * Deterministic update
    integrator.f(rtmp1, uprev, p, t) # Mutate the drift term f(xₙ)

    # Apply variable-specific correction factors
    for i in eachindex(u)
        u[i] = uprev[i] + correction[i] * dt * rtmp1[i]
    end

    # * Split step handling
    if is_split_step(integrator.alg)
        u_choice = u
    else
        u_choice = uprev
    end

    # * Compute noise coefficient
    integrator.g(rtmp2, u_choice, p, t) # Update g(xₙ)

    # * Jump noise
    if P !== nothing
        c(tmp, uprev, p, t, P.dW, nothing)
    end

    # * Apply random noise update with variable-specific corrections
    if is_diagonal_noise(integrator.sol.prob)
        @.. rtmp2 *= W.dW # Timestep scaling in the noise func
        if P !== nothing
            for i in eachindex(u)
                u[i] += (rtmp2[i] + tmp[i]) * correction[i]
            end
        else
            for i in eachindex(u)
                u[i] += rtmp2[i] * correction[i]
            end
        end
    else
        mul!(rtmp1, rtmp2, W.dW)
        if P !== nothing
            for i in eachindex(u)
                u[i] += (rtmp1[i] + tmp[i]) * correction[i]
            end
        else
            for i in eachindex(u)
                u[i] += rtmp1[i] * correction[i]
            end
        end
    end

    # * Apply history update with variable-specific weights
    # Each variable uses its own row of the weights matrix
    for (j, dx) in enumerate(uhist)
        for i in eachindex(u)
            u[i] -= weights[i, j] * dx[i]
        end
    end

    # * Roll history
    push!(uhist, u - uprev)

    return nothing
end

# * Traits
alg_compatible(prob::DiffEqBase.AbstractSDEProblem, alg::MultiCaputoEM) = true
