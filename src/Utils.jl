module Utils

import StochasticDiffEq: StochasticDiffEqAlgorithm,
                         StochasticDiffEqMutableCache,
                         alg_cache, full_cache, jac_iter, rand_cache, ratenoise_cache,
                         perform_step!, is_split_step, is_diagonal_noise,
                         alg_compatible, DiffEqBase,
                         @cache, @muladd, @unpack, @..

export FractionalEM

struct FractionalEM <: StochasticDiffEqAlgorithm end # No split steps for now since we think noise is just additive.

struct FractionalEMCache{uType, rateType, rateNoiseType} <:
       StochasticDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    rtmp1::rateType
    rtmp2::rateNoiseType
end

full_cache(c::FractionalEM) = tuple(c.u, c.uprev, c.tmp, c.rtmp1)
jac_iter(c::FractionalEM) = tuple()
rand_cache(c::FractionalEM) = tuple()
ratenoise_cache(c::FractionalEM) = tuple(c.rtmp2)

function alg_cache(alg::FractionalEM, prob, u, ΔW, ΔZ, p, rate_prototype,
                   noise_rate_prototype,
                   jump_rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, f, t, dt,
                   ::Type{Val{true}}) where {uEltypeNoUnits, uBottomEltypeNoUnits,
                                             tTypeNoUnits}
    tmp = zero(u)
    rtmp1 = zero(rate_prototype)
    if noise_rate_prototype !== nothing
        rtmp2 = zero(noise_rate_prototype)
    else
        rtmp2 = nothing
    end
    FractionalEMCache(u, uprev, tmp, rtmp1, rtmp2)
end

@muladd function perform_step!(integrator, cache::FractionalEMCache)
    @unpack tmp, rtmp1, rtmp2 = cache
    @unpack t, dt, uprev, u, W, P, c, p = integrator

    # * Deterministic update
    integrator.f(rtmp1, uprev, p, t)

    @.. u = uprev + dt * rtmp1

    # * Split step ignore
    if is_split_step(integrator.alg)
        u_choice = u
    else
        u_choice = uprev
    end

    # * Do deterministic noise coefficient
    integrator.g(rtmp2, u_choice, p, t)

    # ! What is this??
    if P !== nothing
        c(tmp, uprev, p, t, P.dW, nothing)
    end

    # * Apply random noise update
    if is_diagonal_noise(integrator.sol.prob)
        @.. rtmp2 *= W.dW
        if P !== nothing
            @.. u += rtmp2 + tmp
        else
            @.. u += rtmp2
        end
    else
        mul!(rtmp1, rtmp2, W.dW)
        if P !== nothing
            @.. u += rtmp1 + tmp
        else
            @.. u += rtmp1
        end
    end
end

# * Traits
alg_compatible(prob::DiffEqBase.AbstractSDEProblem, alg::FractionalEM) = true

end
