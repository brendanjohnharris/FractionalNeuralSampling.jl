using RecursiveArrayTools
import ..NoiseProcesses: LevyProcess!
import ..Boundaries
using UnPack
using ApproxFun
using CompositionsBase
using IntervalSets
import ApproxFun.DomainSets
using SpecialFunctions

export AdaptiveWalkSampler, AdaptiveLevySampler

function domain_length(d::DomainSets.Domain)
    [d.b - d.a]
end
function domain_length(d::DomainSets.ProductDomain)
    map(domain_length, d.domains)
end

function adaptive_walk_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack γ, τ_r, τ_d, sp, D, kernel, ps, plan = ps

    x, a_K = u.x
    dx, da_K = du.x

    ∇V = (-) ∘ gradlogdensity(𝜋)

    K = Fun(sp, a_K)
    if !(sp isa TensorSpace) && !(x isa Number) # Handle 1D space
        ∇K = (D .* [K])
        ∇K = (∇K) .∘ only
        a_k̂ = plan * map(kernel ∘ Base.Fix2(-, only(x)), ps)
    else
        ∇K = (D .* [K]) # This is also kind of slow...
        a_k̂ = plan * map(kernel ∘ Base.Fix2(-, x), ps)
    end

    da_K .= -a_K / τ_d .+ a_k̂ / τ_r
    dx .= -(∇V(x) + [δK(x) for δK in ∇K])
end
function adaptive_walk_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack γ = ps
    dx, da_K = du.x
    dx .= sqrt(2γ)
    da_K .= 0.0
end

function AdaptiveWalkSampler(kernel, approx_n_modes; tspan,
                             γ, τ_r, τ_d,
                             u0 = [0.0],
                             boundaries = nothing, # Only makes sense as a periodic bound.
                             𝜋 = Density(default_density(first(u0.x))),
                             kwargs...)
    dimension = length(u0)

    sp = Boundaries.domain(boundaries) # So boundaries must be passed explicitly
    sp = prod(Fourier.(sp)) # Fourier ensures periodicity of the solution

    k = Fun(kernel, sp, approx_n_modes)

    n_modes = length(k.coefficients) # The number of modes in the adaptive potential
    a_K = zeros(n_modes) # The adaptive basis coefficients
    u0 = ArrayPartition(u0, a_K)
    noise_rate_prototype = map(real, similar(u0)) # nothing # similar(u0)

    noise = nothing #NoiseProcesses.LevyProcess!(1.3; ND = 1,
    #W0 = zero(u0))

    Ds = SVector{dimension, Int}.(eachrow(I(dimension)))
    if sp isa TensorSpace
        D = Derivative.([sp], Ds)
    else
        Ds = only.(Ds)
        D = Derivative.([sp], Ds)
    end

    ps = points(sp, approx_n_modes)
    plan = ApproxFunBase.plan_transform(sp, length(ps))

    p = ((; γ, τ_r, τ_d, sp, D, kernel, ps, plan), 𝜋)

    Sampler(adaptive_walk_f!, adaptive_walk_g!;
            callback = boundaries(),
            u0, noise_rate_prototype, noise, tspan,
            p,
            kwargs...)
end

# * Adaptive levy walk

function adaptive_levy_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack γ, τ_r, τ_d, sp, D, kernel, ps, plan = ps

    x, a_K = u.x
    dx, da_K = du.x

    ∇V = (-) ∘ gradlogdensity(𝜋)

    K = Fun(sp, a_K)
    if !(sp isa TensorSpace) && !(x isa Number) # Handle 1D space
        ∇K = (D .* [K])
        ∇K = (∇K) .∘ only
        a_k̂ = plan * map(kernel ∘ Base.Fix2(-, only(x)), ps)
    else
        ∇K = (D .* [K]) # This is also kind of slow...
        a_k̂ = plan * map(kernel ∘ Base.Fix2(-, x), ps)
    end

    da_K .= -a_K / τ_d .+ a_k̂ / τ_r
    dx .= -(∇V(x) + [δK(x) for δK in ∇K]) * gamma(α - 1) / (gamma(α / 2) .^ 2)
end
function adaptive_levy_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, γ = ps
    dx, da_K = du.x
    dx .= sqrt(2) * γ^(1 / α) # ? × dL in the integrator. This is matrix multiplication
    da_K .= 0.0
end
function AdaptiveLevySampler(kernel, approx_n_modes; tspan,
                             α, γ, τ_r, τ_d,
                             u0 = [0.0],
                             boundaries = nothing, # Only makes sense as a periodic bound.
                             𝜋 = Density(default_density(first(u0.x))),
                             kwargs...)
    dimension = length(u0)

    sp = Boundaries.domain(boundaries) # So boundaries must be passed explicitly
    sp = prod(Fourier.(sp)) # Fourier ensures periodicity of the solution

    k = Fun(kernel, sp, approx_n_modes)

    n_modes = length(k.coefficients) # The number of modes in the adaptive potential
    a_K = zeros(n_modes) # The adaptive basis coefficients
    u0 = ArrayPartition(u0, a_K)
    noise_rate_prototype = map(real, similar(u0)) # nothing # similar(u0)

    noise = NoiseProcesses.LevyProcess!(α; ND = length(u0.x[1]),
                                        W0 = zero(noise_rate_prototype))

    Ds = SVector{dimension, Int}.(eachrow(I(dimension)))
    if sp isa TensorSpace
        D = Derivative.([sp], Ds)
    else
        Ds = only.(Ds)
        D = Derivative.([sp], Ds)
    end

    ps = points(sp, approx_n_modes)
    plan = ApproxFunBase.plan_transform(sp, length(ps))

    p = ((; α, γ, τ_r, τ_d, sp, D, kernel, ps, plan), 𝜋)

    Sampler(adaptive_walk_f!, adaptive_walk_g!;
            callback = boundaries(),
            u0, noise_rate_prototype, noise, tspan,
            p,
            kwargs...)
end
