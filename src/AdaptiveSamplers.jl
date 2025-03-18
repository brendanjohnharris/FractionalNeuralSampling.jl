using RecursiveArrayTools
import ..NoiseProcesses: LevyProcess!
import ..Boundaries
using UnPack
using ApproxFun
using CompositionsBase
using IntervalSets
import ApproxFun.DomainSets

export AdaptiveWalkSampler

function domain_length(d::DomainSets.Domain)
    [d.b - d.a]
end
function domain_length(d::DomainSets.ProductDomain)
    map(domain_length, d.domains)
end

function adaptive_walk_f!(du, u, p, t)
    ps, 𝜋 = p
    γ, τ_r, τ_d, space, D, kernel, ps, plan = ps

    x, a_K = u.x
    dx, da_K = du.x

    ∇V = (-) ∘ gradlogdensity(𝜋)
    # Main.@infiltrate
    K = Fun(space, a_K)
    if !(space isa TensorSpace) && !(x isa Number) # Handle 1D space
        ∇K = (D .* [K])
        ∇K = (∇K) .∘ only
        a_k̂ = plan * map(kernel ∘ Base.Fix2(-, only(x)), ps)
    else
        ∇K = (D .* [K]) # This is also kind of slow...
        a_k̂ = plan * map(kernel ∘ Base.Fix2(-, x), ps)
    end

    # * So we do actually have to do the phase shift ourselves.
    # ks = points(space, length(a_k̂))
    # ϕ = -im .* dot.(ks, [x]) # Phase shift due to recentering for current position

    da_K .= -a_K / τ_d .+ a_k̂ / τ_r # Update the adaptive basis coefficients
    dx .= -(∇V(x) + [δK(x) for δK in ∇K]) # Update the position
end
function adaptive_walk_g!(du, u, p, t)
    ps, 𝜋 = p
    γ, _ = ps
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
    # eltype(a_K) <: Complex || throw(ArgumentError("The basis coefficients must be complex"))
    # kernel = x -> exp(-(norm(x)) / (2 * σ^2))
    dimension = length(u0)

    sp = Boundaries.domain(boundaries) # So boundaries must be passed explicitly
    sp = prod(Fourier.(sp)) # Fourier ensures periodicity of the solution

    k = Fun(kernel, sp, approx_n_modes)

    n_modes = length(k.coefficients) # The number of modes in the adaptive potential
    a_K = zeros(n_modes) # The adaptive basis coefficients
    u0 = ArrayPartition(u0, a_K)
    noise_rate_prototype = nothing # similar(u0)

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

    p = ((γ, τ_r, τ_d, sp, D, kernel, ps, plan), 𝜋)

    Sampler(adaptive_walk_f!, adaptive_walk_g!;
            callback = boundaries(),
            u0, noise_rate_prototype, noise, tspan,
            p,
            kwargs...)
end
