module Boundaries
using SciMLBase
import SciMLBase: DECallback
using IntervalSets
using Statistics
using LinearAlgebra
using LogDensityProblems
import ..FractionalNeuralSampling: divide_dims, Densities.AbstractDensity

export AbstractBoundary, AbstractContinuousBoundary, AbstractBoxBoundary, ReflectingBox,
       NoBoundary, PeriodicBox, ReentrantBox, domain, gridaxes, grid

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
    B(zip(map(extrema, intervals)...)...)
end
function (B::Type{<:AbstractBoxBoundary})(intervals...)
    B(intervals)
end
function (B::Type{<:AbstractBoxBoundary})(interval::AbstractInterval)
    B(zip(map(extrema, [interval])...)...)
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
function getaffect(R::ReflectingBox{D}) where {D}
    function affect!(integrator)
        x, v = divide_dims(integrator.u, D)
        reflectvelocity!(v, x, _corners(R)...)
    end
end
function getcondition(R::AbstractBoxBoundary{D}) where {D}
    function condition(u, t, integrator)
        x = divide_dims(u, D) |> first
        d = boxdist(x, _corners(R)...)
        d < 0
    end
end

struct PeriodicBox{D, Re} <: AbstractBoxBoundary{D}
    min_corner::NTuple{D}
    max_corner::NTuple{D}
    function PeriodicBox(min_corner::NTuple{D}, max_corner::NTuple{D};
                         reset = false) where {D}
        new{D, reset}(min_corner, max_corner)
    end
end
function reenterbox!(velocity, point, min_corner, max_corner; reset = false)
    dists = _boxdist(point, min_corner, max_corner)
    _, edge = findmin(dists)
    edge_faces = [min_corner[edge], max_corner[edge]]
    edgedists = abs.(edge_faces .- point[edge])
    _, old_edge = findmin(edgedists)
    _, new_edge = findmax(edgedists)
    point[edge] = edge_faces[new_edge] .+ (point[edge] .- edge_faces[old_edge])
    if reset
        velocity .= 0
    end
end
function getaffect(R::PeriodicBox{D, Re}) where {D, Re}
    function affect!(integrator)
        x, v = divide_dims(integrator.u, D)
        reenterbox!(v, x, _corners(R)...; reset = Re)
    end
end

"""
A 'half-periodic' box where one edge is permeable and one edge is re-entrant.
"""
struct ReentrantBox{D, Re} <: AbstractBoxBoundary{D}
    reentrance::NTuple{D}
    exit::NTuple{D}
    relpos::NTuple{D, Bool}
    function ReentrantBox(reentrance::NTuple{D}, exit::NTuple{D};
                          reset = true) where {D}
        new{D, reset}(reentrance, exit, Tuple(exit .> reentrance))
    end
end
function ReentrantBox(exitreenter::Pair{<:Real, <:Real};
                      kwargs...)
    ReentrantBox((exitreenter.second,), (exitreenter.first,); kwargs...)
end
function ReentrantBox(exitreenter::Vararg{<:Pair{<:Real, <:Real}};
                      kwargs...)
    ReentrantBox(getproperty.(exitreenter, :second),
                 getproperty.(exitreenter, :first); kwargs...)
end
function ReentrantBox(exitreenter; kwargs...) # Prevent careless inputs by forcing pairs
    throw(ArgumentError("Must provide pairs of enter=>reentrance values for each dimension, where all pairs have the same type."))
end
function ReentrantBox(exitreenter::Pair{<:NTuple{D}, <:NTuple{D}};
                      kwargs...) where {D}
    ReentrantBox(exitreenter.second, exitreenter.first; kwargs...)
end
_corners(R::ReentrantBox) = (R.reentrance, R.exit)
exits(R::ReentrantBox) = R.exit
reentrances(R::ReentrantBox) = R.reentrance
function getaffect(R::ReentrantBox{D, Re}) where {D, Re}
    reentrance = reentrances(R)
    function affect!(integrator)
        vars = divide_dims(integrator.u, D)
        x = first(vars)
        for i in eachindex(x)
            x[i] = reentrance[i]
        end
        if Re && length(vars) > 1
            vars[2] .= 0.0
        end
    end
end
function getcondition(R::ReentrantBox{D}) where {D}
    # Pre-compute values outside the inner function
    reentrance, exit = _corners(R)
    relpos = R.relpos
    function condition(u, t, integrator)
        x = divide_dims(u, D) |> first

        for i in 1:length(x)
            y = x[i]
            ex = reentrance[i]
            en = exit[i]
            rel = relpos[i]

            # Early return once we find a condition match
            if rel ? (y > en) : (y < en)
                return true
            end
        end
        return false
    end
end

function (B::AbstractBoxBoundary)(; kwargs...)
    DiscreteCallback(getcondition(B), getaffect(B); save_positions = (false, false), # ! Will need to think about this more
                     kwargs...)
end

function domain(R::AbstractBoxBoundary)
    [Interval(is...) for is in zip(R.min_corner, R.max_corner)]
end

function gridaxes(R::AbstractBoxBoundary{D}, n::Int) where {D}
    map(domain(R)) do r
        range(r, length = n)
    end
end

function grid(R::AbstractBoxBoundary, n)
    r = gridaxes(R, n)
    Iterators.product(r...)
end

end # module
