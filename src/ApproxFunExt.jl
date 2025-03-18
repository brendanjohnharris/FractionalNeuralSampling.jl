import ApproxFunBase: Fourier

export Translation

abstract type Translation{S, T} <: Operator{T} end

struct ConcreteTranslation{S <: Space, T} <: Translation{S, T}
    G::Fun{S, T}
end

function Translation(G::Fun{S, T}) where {S, T}
    # @assert isfinite(arclength(domain(G)))
    ConcreteTranslation(G)
end
ApproxFunBase.domain(C::ConcreteTranslation) = ApproxFunBase.domain(C.G)
ApproxFunBase.domainspace(C::ConcreteTranslation) = ApproxFunBase.space(C.G)
ApproxFunBase.rangespace(C::ConcreteTranslation) = ApproxFunBase.space(C.G)
function bandwidths(C::ConcreteTranslation)
    error("Please implement translation bandwidths on " * string(space(C.G)))
end
function getindex(C::ConcreteTranslation, k::Integer, j::Integer)
    error("Please implement translation getindex on " * string(space(C.G)))
end

const _Laurent = Laurent{PeriodicSegment{R}, T} where {R <: Real, T}
bandwidths(C::ConcreteTranslation{_Laurent}) where {R <: Real, T} = (0, 0)
const _Fourier = Fourier{PeriodicSegment{R}, T} where {R <: Real, T}
bandwidths(C::ConcreteTranslation{_Fourier}) where {R <: Real, T} = (1, 1)

function getindex(C::ConcreteTranslation{_Laurent, T2}, k::Integer,
                  j::Integer) where {R <: Real, T1, T2}
    fourier_index::Integer = if isodd(k)
        div(k - 1, 2)
    else
        -div(k, 2)
    end
    if k == j && k ≤ ncoefficients(C.G)
        return (exp(-2pi * 1im / arclength(domain(C.G)) * fourier_index *
                    first(domain(C.G))) * arclength(domain(C.G)) * C.G.coefficients[k])::T2
    else
        return zero(T2)
    end
end

function getindex(C::ConcreteTranslation{_Fourier, T2}, k::Integer,
                  j::Integer) where {R <: Real, T1, T2}
    fourier_index::Integer = if isodd(k)
        div(k - 1, 2)
    else
        div(k, 2)
    end
    if k < 1 || j < 1 || ncoefficients(C.G) == 0
        return zero(T2)
    elseif k == 1
        if j == k
            return (arclength(domain(C.G)) * C.G.coefficients[1])::T2
        else
            return zero(T2)
        end
    elseif 2 * fourier_index ≤ ncoefficients(C.G)
        Gs = if 2 * fourier_index ≤ ncoefficients(C.G)
            C.G.coefficients[2 * fourier_index]
        else
            zero(T2)
        end # sine coefficient
        Gc = if 2 * fourier_index + 1 ≤ ncoefficients(C.G)
            C.G.coefficients[2 * fourier_index + 1]
        else
            zero(T2)
        end # cosine coefficient
        phase = 2pi / arclength(domain(C.G)) * fourier_index * first(domain(C.G))
        if iseven(k) && j == k
            return (arclength(domain(C.G)) * (Gc * cos(phase) - Gs * sin(phase)) / 2)::T2
        elseif iseven(k) && j == k + 1
            return (arclength(domain(C.G)) * (Gc * sin(phase) + Gs * cos(phase)) / 2)::T2
        elseif isodd(k) && j == k
            return (arclength(domain(C.G)) * (Gc * cos(phase) - Gs * sin(phase)) / 2)::T2
        elseif isodd(k) && j == k - 1
            return (arclength(domain(C.G)) * (-Gc * sin(phase) - Gs * cos(phase)) / 2)::T2
        else
            return zero(T2)
        end
    else
        return zero(T2)
    end
end
