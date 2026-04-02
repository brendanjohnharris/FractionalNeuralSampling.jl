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
    [d.b - d.a]
end
function domain_length(d::DomainSets.ProductDomain)
    map(domain_length, d.domains)
end

# * Kernel callback (shared by both samplers)
kernel_condition(u, t, integrator) = true

function kernel_effect!(integrator)
    u = integrator.u
    p = integrator.p[1]
    dt = integrator.dt
    @unpack dim, sp, plan, kernel, grid_points, τ_d, τ_r = p

    x = divide_dims(u, dim)[1]

    if !(sp isa TensorSpace) && !(x isa Number)
        a_k̂ = plan * map(kernel ∘ Base.Fix2(-, only(x)), grid_points)
    else
        a_k̂ = plan * map(kernel ∘ Base.Fix2(-, x), grid_points)
    end

    da_K = -p.a_K ./ τ_d .+ a_k̂ ./ τ_r
    p.a_K .+= da_K .* dt

    u_modified!(integrator, false)
end

# * Adaptive walk (Gaussian noise)

function adaptive_walk_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack γ, D, a_K, sp, dim = ps

    x = divide_dims(u, dim)[1]
    dx = divide_dims(du, dim)[1]

    ∇V = (-) ∘ gradlogdensity(𝜋)

    K = Fun(sp, a_K)
    if !(sp isa TensorSpace) && !(x isa Number)
        ∇K_funcs = (D .* [K]) .∘ only
    else
        ∇K_funcs = (D .* [K])
    end
    ∇K_val = [f(x) for f in ∇K_funcs]

    dx .= -γ * (∇V(x) .+ ∇K_val)
end

function adaptive_walk_g!(du, u, p, t)
    ps, _ = p
    @unpack γ, dim = ps
    dx = divide_dims(du, dim)[1]
    dx .= sqrt(2γ)
end

function AdaptiveWalkSampler(kernel, approx_n_modes;
                             tspan, γ, τ_r, τ_d,
                             u0 = [0.0],
                             boundaries = nothing,
                             noise_rate_prototype = similar(u0),
                             𝜋 = Density(default_density(first(u0))),
                             noise = WienerProcess!(0.0, zero(u0)),
                             alg = EM(),
                             callback = (),
                             kwargs...)
    dim = length(u0)

    sp_domain = Boundaries.domain(boundaries)
    sp = prod(Fourier.(sp_domain))

    k_init = Fun(kernel, sp, approx_n_modes)
    n_modes = length(k_init.coefficients)
    a_K = zeros(n_modes)

    Ds = SVector{dim, Int}.(eachrow(I(dim)))
    if sp isa TensorSpace
        D = Derivative.([sp], Ds)
    else
        Ds = only.(Ds)
        D = Derivative.([sp], Ds)
    end

    grid_points = points(sp, n_modes)
    plan = ApproxFunBase.plan_transform(sp, length(grid_points))

    p = (; γ, τ_r, τ_d, D, a_K, sp, plan, kernel, grid_points, dim)

    kernelcallback = DiscreteCallback(kernel_condition, kernel_effect!)
    Sampler(adaptive_walk_f!, adaptive_walk_g!;
            callback = CallbackSet(boundary_init(boundaries), kernelcallback, callback...),
            kwargs...,
            u0, noise_rate_prototype, noise, tspan,
            p, 𝜋, alg) |> assert_dimension(; order = 1)
end

# * Adaptive Lévy walk

function adaptive_levy_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, γ, D, a_K, sp, dim = ps

    x = divide_dims(u, dim)[1]
    dx = divide_dims(du, dim)[1]

    ∇V = (-) ∘ gradlogdensity(𝜋)

    K = Fun(sp, a_K)
    if !(sp isa TensorSpace) && !(x isa Number)
        ∇K_funcs = (D .* [K]) .∘ only
    else
        ∇K_funcs = (D .* [K])
    end
    ∇K_val = [f(x) for f in ∇K_funcs]

    dx .= -γ * (∇V(x) .+ ∇K_val) * gamma(α - 1) / (gamma(α / 2) .^ 2)
end

function adaptive_levy_g!(du, u, p, t)
    ps, _ = p
    @unpack α, γ, dim = ps
    dx = divide_dims(du, dim)[1]
    dx .= γ^(1 / α)
end

function AdaptiveLevySampler(kernel, approx_n_modes;
                             tspan, α, γ, τ_r, τ_d,
                             u0 = [0.0],
                             boundaries = nothing,
                             noise_rate_prototype = similar(u0),
                             𝜋 = Density(default_density(first(u0))),
                             noise = NoiseProcesses.LevyProcess!(α; ND = length(u0),
                                                                 W0 = zero(u0)),
                             alg = EM(),
                             callback = (),
                             kwargs...)
    dim = length(u0)

    sp_domain = Boundaries.domain(boundaries)
    sp = prod(Fourier.(sp_domain))

    k_init = Fun(kernel, sp, approx_n_modes)
    n_modes = length(k_init.coefficients)
    a_K = zeros(n_modes)

    Ds = SVector{dim, Int}.(eachrow(I(dim)))
    if sp isa TensorSpace
        D = Derivative.([sp], Ds)
    else
        Ds = only.(Ds)
        D = Derivative.([sp], Ds)
    end

    grid_points = points(sp, n_modes)
    plan = ApproxFunBase.plan_transform(sp, length(grid_points))

    p = (; α, γ, τ_r, τ_d, D, a_K, sp, plan, kernel, grid_points, dim)

    kernelcallback = DiscreteCallback(kernel_condition, kernel_effect!)
    Sampler(adaptive_levy_f!, adaptive_levy_g!;
            callback = CallbackSet(boundary_init(boundaries), kernelcallback, callback...),
            kwargs...,
            u0, noise_rate_prototype, noise, tspan,
            p, 𝜋, alg) |> assert_dimension(; order = 1)
end
