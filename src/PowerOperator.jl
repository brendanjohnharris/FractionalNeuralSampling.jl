import ApproxFun: Operator, Derivative
import ApproxFun

export Power

"""
    Power(op::Operator, p::Number)

A diagonal `ApproxFun` operator raised to the power `p`, entry by entry.

Used to build the fractional Laplacian ``(-\\Delta)^{(\\alpha-2)/2}`` that the
space-fractional samplers need. The Laplacian is diagonal in the Fourier basis, so raising
it to a fractional power is a power of each diagonal entry.

A zero diagonal entry raised to a negative power is left at zero rather than diverging.
That entry is the zero-frequency mode, which carries the offset of the potential and so
does not affect a gradient; apply the fractional operator before the gradient operator for
this to hold.
"""
abstract type Power{T, BT <: Operator, P <: Number} <: Operator{T} end

struct ConcretePower{T, BT <: Operator, P <: Number} <: Power{T, BT, P}
    op::BT
    p::P
end

function Power{T}(op::BT, p::P) where {T, BT <: Operator, P <: Number}
    return ConcretePower{T, BT, P}(op, p)
end
function Power(op::BT, p::P) where {BT <: Operator, P <: Number}
    return Power{promote_type(eltype(op), P)}(op, p)
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
            # ! For potentials, the 0-frequency offset doesn't matter, but make sure the
            # fractional operator is applied PRIOR to the regular gradient operator.
            return a
        else
            return convert(T, a)^p.p
        end
    else
        # If the base operator is diagonal, so is the powered operator.
        return zero(T)
    end
end
