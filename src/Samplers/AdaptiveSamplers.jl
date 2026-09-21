using UnPack
using ApproxFun
using IntervalSets

import ApproxFun.DomainSets
import StaticArraysCore.SVector
import ..NoiseProcesses: LevyProcess!
import ..Boundaries
import SpecialFunctions: gamma

export AdaptiveWalkSampler, AdaptiveLevySampler

function domain_length(d::DomainSets.Domain)
    return [d.b - d.a]
end
function domain_length(d::DomainSets.ProductDomain)
    return map(domain_length, d.domains)
end

"""
Spectral machinery shared by the adaptive samplers: a Fourier space over the boundary
domain, the coefficients `a_K` of the adaptation kernel in that space, the derivative
operators used for ∇K, and the transform that projects each new kernel deposit.
"""
function kernel_parameters(kernel, approx_n_modes, u0, boundaries)
    isnothing(boundaries) &&
        throw(ArgumentError("Adaptive samplers need `boundaries`, which set the domain of the adaptation kernel"))
    dim = length(u0)
    sp = prod(Fourier.(Boundaries.domain(boundaries)))

    a_K = zeros(length(Fun(kernel, sp, approx_n_modes).coefficients))

    Ds = SVector{dim, Int}.(eachrow(I(dim)))
    sp isa TensorSpace || (Ds = only.(Ds))
    D = Derivative.([sp], Ds)

    grid_points = points(sp, length(a_K))
    plan = ApproxFunBase.plan_transform(sp, length(grid_points))

    return (; D, a_K, sp, plan, kernel, grid_points, dim)
end

# * Kernel callback (shared by both samplers)
kernel_condition(u, t, integrator) = true

function kernel_effect!(integrator)
    u = integrator.u
    p = integrator.p[1]
    dt = integrator.dt
    @unpack dim, sp, plan, kernel, grid_points, τ_d, τ_r = p

    x = first_dims(u, dim)

    if !(sp isa TensorSpace) && !(x isa Number)
        a_k̂ = plan * map(kernel ∘ Base.Fix2(-, only(x)), grid_points)
    else
        a_k̂ = plan * map(kernel ∘ Base.Fix2(-, x), grid_points)
    end

    @. p.a_K += (-p.a_K / τ_d + a_k̂ / τ_r) * dt

    return derivative_discontinuity!(integrator, false) # SciMLBase v3 name for u_modified!
end

"""
∇K(x), the gradient of the current adaptation kernel at `x`
"""
function gradkernel(x, ps)
    @unpack D, a_K, sp = ps
    K = Fun(sp, a_K)
    if !(sp isa TensorSpace) && !(x isa Number)
        ∇K_funcs = (D .* [K]) .∘ only
    else
        ∇K_funcs = (D .* [K])
    end
    return [f(x) for f in ∇K_funcs]
end

"""
The sampler shared by [`AdaptiveWalkSampler`](@ref) and [`AdaptiveLevySampler`](@ref):
only the drift prefactor and the noise process differ
"""
function adaptive_sampler(
        f!, g!, kernel, approx_n_modes, ps; u0, tspan, boundaries,
        𝜋, noise, noise_rate_prototype, alg, callback, kwargs...
    )
    p = (; ps..., kernel_parameters(kernel, approx_n_modes, u0, boundaries)...)
    kernelcallback = DiscreteCallback(kernel_condition, kernel_effect!)
    return Sampler(
        f!, g!;
        callback = CallbackSet(boundary_init(boundaries), kernelcallback, callback...),
        kwargs...,
        u0, noise_rate_prototype, noise, tspan,
        p, 𝜋, alg
    ) |> assert_dimension(; order = 1)
end

# * Adaptive walk (Gaussian noise)

function adaptive_walk_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack γ, dim = ps

    x = first_dims(u, dim)
    dx = first_dims(du, dim)

    ∇V = (-) ∘ gradlogdensity(𝜋)

    return dx .= -γ * (∇V(x) .+ gradkernel(x, ps))
end

function adaptive_walk_g!(du, u, p, t)
    ps, _ = p
    @unpack γ, dim = ps
    dx = first_dims(du, dim)
    return dx .= sqrt(2γ)
end

function AdaptiveWalkSampler(
        kernel, approx_n_modes;
        tspan, γ, τ_r, τ_d,
        u0 = [0.0],
        boundaries = nothing,
        noise_rate_prototype = similar(u0),
        𝜋 = default_density(u0; dims = length(u0)),
        noise = WienerProcess!(0.0, zero(u0)),
        alg = EM(),
        callback = (),
        kwargs...
    )
    return adaptive_sampler(
        adaptive_walk_f!, adaptive_walk_g!, kernel, approx_n_modes, (; γ, τ_r, τ_d);
        u0, tspan, boundaries, 𝜋, noise, noise_rate_prototype, alg, callback, kwargs...
    )
end

# * Adaptive Lévy walk

function adaptive_levy_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, γ, dim = ps

    x = first_dims(u, dim)
    dx = first_dims(du, dim)

    ∇V = (-) ∘ gradlogdensity(𝜋)

    return dx .= -γ * (∇V(x) .+ gradkernel(x, ps)) * gamma(α - 1) / (gamma(α / 2) .^ 2)
end

function adaptive_levy_g!(du, u, p, t)
    ps, _ = p
    @unpack α, γ, dim = ps
    dx = first_dims(du, dim)
    return dx .= γ^(1 / α)
end

function AdaptiveLevySampler(
        kernel, approx_n_modes;
        tspan, α, γ, τ_r, τ_d,
        u0 = [0.0],
        boundaries = nothing,
        noise_rate_prototype = similar(u0),
        𝜋 = default_density(u0; dims = length(u0)),
        noise = NoiseProcesses.LevyProcess!(
            α; ND = length(u0),
            W0 = zero(u0)
        ),
        alg = EM(),
        callback = (),
        kwargs...
    )
    return adaptive_sampler(
        adaptive_levy_f!, adaptive_levy_g!, kernel, approx_n_modes, (; α, γ, τ_r, τ_d);
        u0, tspan, boundaries, 𝜋, noise, noise_rate_prototype, alg, callback, kwargs...
    )
end
