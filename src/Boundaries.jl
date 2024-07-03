module Boundaries
using SciMLBase
import SciMLBase: DECallback
using IntervalSets
using Statistics
using LinearAlgebra

export AbstractBoundary, AbstractContinuousBoundary, AbstractBoxBoundary, ReflectingBox,
       NoBoundary, PeriodicBox

# ? Boundary conditions are just callbacks
abstract type AbstractBoundary{D} end
"""
Continuous boundaries can be arbitrarily shaped, but require evaluating all distances at
once, and local gradients around the boundary. Hard to do...
"""
abstract type AbstractContinuousBoundary{D} <: AbstractBoundary{D} end

struct NoBoundary <: AbstractBoundary{nothing} end
(::NoBoundary)(; kwargs...) = nothing
# function (B::AbstractContinuousBoundary)(; kwargs...)
#     ContinuousCallback(getconditon(B), getaffect(B); save_positions = (false, true),
#                        kwargs...)
# end

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
            return max_c - p  # Negative if outside the max edge
        else
            return min((p - min_c), (max_c - p))  # Positive if inside, closer to which edge
        end
    end
end
function boxdist(point, min_corner, max_corner)
    d = minimum(_boxdist(point, min_corner, max_corner))  # Overall distance to the closest edge
    return d
end
function reflectvelocity!(velocity, point, min_corner, max_corner)
    dists = _boxdist(point, min_corner, max_corner)
    _, edge = findmin(dists)
    velocity[edge] = -velocity[edge]
    edge_faces = [min_corner[edge], max_corner[edge]]
    _, closest_edge = findmin(abs.(edge_faces .- point[edge]))
    point[edge] = edge_faces[closest_edge] .- (point[edge] .- edge_faces[closest_edge]) # ! Is this best???
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
    sort!(cs, by = c -> -acos((c .- O)[1] ./ norm(c .- O)))
    return cs
end
function getaffect(R::ReflectingBox)
    function affect!(integrator)
        reflectvelocity!(view(integrator.u, 2), view(integrator.u, 1), _corners(R)...)
    end
end
function getcondition(R::AbstractBoxBoundary)
    function condition(u, t, integrator)
        d = boxdist(u, _corners(R)...)
        d < 0
    end
end

struct PeriodicBox{D} <: AbstractBoxBoundary{D}
    min_corner::NTuple{D}
    max_corner::NTuple{D}
end
function reenterbox!(velocity, point, min_corner, max_corner)
    dists = _boxdist(point, min_corner, max_corner)
    _, edge = findmin(dists)
    edge_faces = [min_corner[edge], max_corner[edge]]
    edgedists = abs.(edge_faces .- point[edge])
    _, old_edge = findmin(edgedists)
    _, new_edge = findmax(edgedists)
    point[edge] = edge_faces[new_edge] .+ (point[edge] .- edge_faces[old_edge])
end
function getaffect(R::PeriodicBox)
    function affect!(integrator)
        reenterbox!(view(integrator.u, 2), view(integrator.u, 1), _corners(R)...)
    end
end

function (B::AbstractBoxBoundary)(; kwargs...)
    DiscreteCallback(getcondition(B), getaffect(B); save_positions = (false, false), # ! Will need to think about this more
                     kwargs...)
end

end # module
