module Boundaries
using SciMLBase
import SciMLBase: DECallback
using IntervalSets
using Statistics
using LinearAlgebra

export AbstractBoundary, AbstractContinuousBoundary, AbstractBoxBoundary, ReflectingBox

# ? Boundary conditions are just callbacks
abstract type AbstractBoundary{D} end
"""
Continuous boundaries can be arbitrarily shaped, but require evaluating all distances at
once, and local gradients around the boundary. Hard to do...
"""
abstract type AbstractContinuousBoundary{D} <: AbstractBoundary{D} end
function (B::AbstractContinuousBoundary)(; kwargs...)
    # ? boundary is a function that, given a point, evaluates whether the point is "in" the
    # ? boundary (in >0, out<= 0)
    ContinuousCallback(getconditon(B), getaffect(B); save_positions = (false, true),
                       kwargs...)
end

"""
Box boundaries have edges that are parallel to the axes. Reflections are easier to calculate
"""
abstract type AbstractBoxBoundary{D} <: AbstractBoundary{D} end
function (B::Type{<:AbstractBoxBoundary})(intervals)
    D = length(intervals)
    B(zip(map(extrema, intervals)...)...)
end
function (B::Type{<:AbstractBoxBoundary})(intervals::Interval...)
    B(intervals)
end

function _boxdist(point, min_corner, max_corner)
    map(point, min_corner, max_corner) do p, min_c, max_c
        if p < min_c
            return p - min_c  # Negative if outside the min edge
        elseif p > max_c
            return p - max_c  # Negative if outside the max edge
        else
            return min(p - min_c, max_c - p)  # Positive if inside, closer to which edge
        end
    end
end
function boxdist(point, min_corner, max_corner)
    return minimum(_boxdist(point, min_corner, max_corner))  # Overall distance to the closest edge
end
function reflectvelocity!(velocity, point, min_corner, max_corner)
    dists = _boxdist(point, min_corner, max_corner)
    _, edge = findmin(dists)
    velocity[edge] *= -1
end
struct ReflectingBox{D} <: AbstractBoxBoundary{D}
    min_corner::NTuple{D}
    max_corner::NTuple{D}
end
_corners(R::AbstractBoxBoundary) = (R.min_corner, R.max_corner)
function corners(R::AbstractBoxBoundary)
    c1, c2 = _corners(R)
    D = length(c1)
    O = middle.(c1, c2)
    cs = Vector{typeof(c1)}()
    for i in 0:(2^D - 1)
        c = []
        for j in 1:D
            if (i >> (j - 1)) & 1 == 1
                push!(c, c2[j])
            else
                push!(c, c1[j])
            end
        end
        push!(cs, typeof(c1)(c))
    end
    sort!(cs, by = c -> atan(c[2] - O[2], c[1] - O[1])) # Get them in manifold order
    return cs
end
function getaffect(R::ReflectingBox)
    integrator -> (view(integrator.u, 2), view(integrator.u, 1), corners(R)...)
end
function getcondition(R::ReflectingBox)
    (u, t, integrator) -> boxdist(u, corners(R)...)
end

end # module
