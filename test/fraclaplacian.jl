using ApproxFun
using CairoMakie
using DifferentiationInterface
using ForwardDiff
using Foresight
using TimeseriesTools
set_theme!(foresight(:physics))

begin # * Choose a test function
    # # V(x; σ = 2) = (1 - exp(-σ * (x)^2))
    # function V(x; σ = 50)
    #     (1 - exp(-σ * (x + 1)^2)) + (1 - exp(-σ * (x - 1)^2)) + (1 - exp(-σ * (x)^2))
    # end
    V(x) = (x^2 - 1)^2

    𝜋(x; kwargs...) = exp(-V(x; kwargs...))

    dV(x) = derivative(V, AutoForwardDiff(), x)
    ddV(x) = derivative(dV, AutoForwardDiff(), x)
end

begin # * Represent spectrally
    approx_n_modes = 1000
    domain = Interval(-5.0, 5.0)
    S = Laurent(domain)
    Vs = Fun(V, S, approx_n_modes)
    𝜋s = Fun(𝜋, S, approx_n_modes)
end
begin # * Compare
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "V(x)")
    xs = LinRange(-5, 5, 1000)
    lines!(ax, xs, V.(xs), color = :blue, label = "true")
    lines!(ax, xs, Vs.(xs), color = :red, linestyle = :dash, label = "approx")
    display(f)
end
begin # * Compare
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "𝜋(x)")
    xs = LinRange(-5, 5, 1000)
    lines!(ax, xs, 𝜋.(xs), color = :blue, label = "true")
    lines!(ax, xs, 𝜋s.(xs), color = :red, linestyle = :dash, label = "approx")
    display(f)
end

begin # * Regular derivative
    D = Derivative(S)
    DV = D * Vs
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "V'(x)")
    xs = LinRange(-5, 5, 1000)
    lines!(ax, xs, dV.(xs), label = "true")
    lines!(ax, xs, DV.(xs), color = :red, linestyle = :dash, label = "approx")
    display(f)
end

function maybeLaplacian(S::ApproxFunBase.DirectSumSpace) # * If the space is 1D, use regular second derivative
    Derivative(S, 2)
end
function maybeLaplacian(S::ApproxFunBase.AbstractProductSpace) # * If the space is multidimensional, use Laplacian
    Laplacian(S)
end
begin # * Laplacian? Constant for quadratic. Seems like Gibbs phenomenon gets stronger for 2nd derivatives...
    Δ = maybeLaplacian(S)
    ΔV = Δ * Vs
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "V''(x)")
    xs = LinRange(-4.5, 4.5, 1000)
    lines!(ax, xs, ddV.(xs), color = :blue, label = "true")
    lines!(ax, xs, ΔV.(xs), color = :red, linestyle = :dash, label = "approx")
    display(f)
end

export PowerOperator

abstract type Power{T, BT <: Operator, P <: Number} <: Operator{T} end

struct ConcretePower{T, BT <: Operator, P <: Number} <: Power{T, BT, P}
    op::BT
    p::P
end

function Power{T}(op::BT, p::P) where {T, BT <: Operator, P <: Number}
    return ConcretePower{T, BT, P}(op, p)
end
function Power(op::BT, p::P) where {BT <: Operator, P <: Number}
    Power{promote_type(eltype(op), P)}(op, p)
end

# The domain and range spaces are the same as the base operator's
ApproxFun.domainspace(P::ConcretePower) = ApproxFun.domainspace(P.op)
ApproxFun.rangespace(P::ConcretePower) = ApproxFun.rangespace(P.op)
ApproxFun.domain(P::ConcretePower) = ApproxFun.domain(P.op)

# The operator is also diagonal, so bandwidths are (0, 0)
ApproxFun.bandwidths(P::ConcretePower) = 0, 0

function Base.getindex(p::ConcretePower{T, BT, P}, k::Integer, j::Integer) where {T, BT, P}
    if k == j
        # The core logic: the new diagonal entry is the original
        # diagonal entry raised to the power p.
        a = p.op[k, k]
        if a == 0 && p.p < 0
            # ! We can't raise zero to a negative power
            # ! For potentials, the 0-frequency offset doesn't matter
            return zero(T)
        else
            return convert(T, a)^p.p
        end
    else
        # If the base operator is diagonal, so is the powered operator.
        return zero(T)
    end
end

begin # * Effective potential defined through fractional laplacian
    αs = Dim{:α}(0.0:0.25:2.0)
    Δ = maybeLaplacian(S)
    Veffs = map(αs) do α
                FΔ = Power(-Δ, (α / 2) - 1)  # Negative for power consistency. For alpha = 2, this is -Δ
                # *
                Veff = FΔ * Vs
                f = Figure()
                ax = Axis(f[1, 1], xlabel = "x", ylabel = "V(x)")
                xs = X(LinRange(-2.0, 2.0, 1000))
                V = Veff.(xs)
                return V .- maximum(V)
            end |> stack |> ToolsArray

    f = Figure()
    ax = Axis(f[1, 1], xlabel = L"x", ylabel = L"\tilde{V}(x)",
              title = "Effective potential for different α")
    p = traces!(ax, reverse(Veffs, dims = 2); colormap = sunrise, linewidth = 2)
    Colorbar(f[1, 2], p; label = L"\alpha")
    display(f)
end

begin # * Drift defined through fractional laplacian
    αs = Dim{:α}(1.0:0.25:2.0)
    Δ = maybeLaplacian(S)
    drifts = map(αs) do α
                 FΔ = Power(-Δ, (α / 2) - 1)  # Negative for power consistency. For alpha = 2, this is -Δ
                 # * We keep the frac laplacian well behaved by multiplying and dividing the
                 #   target distribution, like in fhmc paper
                 drift = (D * (FΔ * 𝜋s)) / 𝜋s # FΔ * 𝜋s
                 f = Figure()
                 ax = Axis(f[1, 1], xlabel = "x", ylabel = "V(x)")
                 xs = X(LinRange(-2.0, 2.0, 100))
                 V = drift.(xs)
                 # * Integrate to get 'potential'
                 V = -cumsum(V) * (xs[2] - xs[1])
                 return V
             end |> stack |> ToolsArray

    drifts = drifts .- minimum(drifts, dims = 1)
    f = Figure()
    ax = Axis(f[1, 1], xlabel = L"x", ylabel = L"\tilde{V}(x)",
              title = "Drift for different α", limits = (nothing, (0, 1)))
    p = traces!(ax, reverse(drifts, dims = 2); colormap = sunrise, linewidth = 2)
    Colorbar(f[1, 2], p; label = L"\alpha")
    display(f)
end
