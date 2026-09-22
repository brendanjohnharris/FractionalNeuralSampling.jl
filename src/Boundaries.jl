module Boundaries
using SciMLBase
import SciMLBase: DECallback
using IntervalSets
using Statistics
using Distributions
using LinearAlgebra
using LogDensityProblems
import ..FractionalNeuralSampling: divide_dims, first_dims, Densities.AbstractDensity

export AbstractBoundary, AbstractContinuousBoundary, AbstractBoxBoundary, ReflectingBox,
    NoBoundary, PeriodicBox, ReentrantBox, gridaxes, grid

# ? Boundary conditions are just callbacks
"""
    AbstractBoundary{D}

Supertype of the boundary conditions, over a `D`-dimensional position. A boundary is a
callback, so it is passed to any sampler through the `boundaries` keyword and composes with
whatever other callbacks the solve already carries.

The concrete types are [`ReflectingBox`](@ref), [`PeriodicBox`](@ref) and
[`ReentrantBox`](@ref), with [`NoBoundary`](@ref) as the explicit no-op.
"""
abstract type AbstractBoundary{D} end
"""
Continuous boundaries can be arbitrarily shaped, but require evaluating all distances at
once, and local gradients around the boundary. Hard to do...
"""
abstract type AbstractContinuousBoundary{D} <: AbstractBoundary{D} end

"""
    NoBoundary()

An explicit no-op boundary, equivalent to passing `boundaries = nothing`.
"""
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
function (B::Type{<:AbstractBoxBoundary})(intervals; kwargs...)
    return B(zip(map(extrema, intervals)...)...; kwargs...)
end
function (B::Type{<:AbstractBoxBoundary})(intervals...; kwargs...)
    return B(intervals; kwargs...)
end
function (B::Type{<:AbstractBoxBoundary})(interval::AbstractInterval; kwargs...)
    return B(zip(map(extrema, [interval])...)...; kwargs...)
end

"""
Promote a pair of corners to a common element type, so that box fields are concrete
"""
function _promote_corners(c1::NTuple{D}, c2::NTuple{D}) where {D}
    T = promote_type(map(typeof, c1)..., map(typeof, c2)...)
    return map(Base.Fix1(convert, T), c1), map(Base.Fix1(convert, T), c2)
end

function _boxdist(point, min_corner, max_corner)
    return map(point, min_corner, max_corner) do p, min_c, max_c
        if p < min_c
            return p - min_c  # Negative if outside the min edge
        elseif p > max_c
            return max_c - p  # Negative if outside the max edge
        else
            return min((p - min_c), (max_c - p))  # Positive if inside, closer to which edge
        end
    end
end
"""
Whether any coordinate of `point` lies outside the box; equivalent to a negative
[`boxdist`](@ref), but without materialising the distances
"""
function isoutside(point, min_corner, max_corner)
    return any(eachindex(point)) do i
        point[i] < min_corner[i] || point[i] > max_corner[i]
    end
end
function boxdist(point, min_corner, max_corner)
    d = minimum(_boxdist(point, min_corner, max_corner))  # Overall distance to the closest edge
    return d
end
"""
Mirror `point` about the nearest box face, reversing the corresponding component of
`velocity`. `velocity === nothing` for first-order systems, which carry no momentum
"""
function reflectvelocity!(velocity, point, min_corner, max_corner)
    dists = _boxdist(point, min_corner, max_corner)
    _, edge = findmin(dists)
    isnothing(velocity) || (velocity[edge] = -velocity[edge])
    edge_faces = [min_corner[edge], max_corner[edge]]
    _, closest_edge = findmin(abs.(edge_faces .- point[edge]))
    return point[edge] = edge_faces[closest_edge] .- (point[edge] .- edge_faces[closest_edge]) # ! Is this best???
end
"""
    ReflectingBox(interval)
    ReflectingBox(interval, interval, ...)
    ReflectingBox(min_corner::NTuple, max_corner::NTuple)

A box whose faces reflect. When the position leaves the box it is mirrored about the
nearest face and the corresponding velocity component is reversed; a first-order sampler
carries no momentum, so only the position is mirrored.

Takes one `IntervalSets` interval per dimension.

```julia
ReflectingBox(-4 .. 4)             # one dimension
ReflectingBox(-4 .. 4, -2 .. 2)    # two dimensions
```

A jump larger than the box is mirrored to a point still outside it, and the callback fires
again on the following step.
"""
struct ReflectingBox{D, T} <: AbstractBoxBoundary{D}
    min_corner::NTuple{D, T}
    max_corner::NTuple{D, T}
    function ReflectingBox(min_corner::NTuple{D}, max_corner::NTuple{D}) where {D}
        mn, mx = _promote_corners(min_corner, max_corner)
        return new{D, eltype(mn)}(mn, mx)
    end
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
    return function affect!(integrator)
        vars = divide_dims(integrator.u, D)
        return if length(vars) == 1
            x = vars[1]
            reflectvelocity!(nothing, x, _corners(R)...)
        else
            x = vars[1]
            v = vars[2]
            reflectvelocity!(v, x, _corners(R)...)
        end
    end
end
function getcondition(R::AbstractBoxBoundary{D}) where {D}
    return function condition(u, t, integrator)
        return isoutside(first_dims(u, D), _corners(R)...)
    end
end

"""
    PeriodicBox(interval; reset = false)
    PeriodicBox(interval, interval, ...; reset = false)
    PeriodicBox(min_corner::NTuple, max_corner::NTuple; reset = false)

A box whose faces wrap. When the position leaves the box it is wrapped modulo the box
width, so it re-enters from the opposite face however far it overshot.

Takes one `IntervalSets` interval per dimension. With `reset = true` the velocity is zeroed
at each wrap, which suits a state that should not carry momentum across the boundary.

```julia
PeriodicBox(-4 .. 4)
PeriodicBox(-4 .. 4, -2 .. 2)
PeriodicBox(-4 .. 4; reset = true)
```
"""
struct PeriodicBox{D, Re, T} <: AbstractBoxBoundary{D}
    min_corner::NTuple{D, T}
    max_corner::NTuple{D, T}
    function PeriodicBox(
            min_corner::NTuple{D}, max_corner::NTuple{D};
            reset = false
        ) where {D}
        mn, mx = _promote_corners(min_corner, max_corner)
        return new{D, reset, eltype(mn)}(mn, mx)
    end
end
"""
Wrap `point` into the box. `velocity` is zeroed if `reset`, and is `nothing` for
first-order systems, which carry no momentum
"""
function reenterbox!(velocity, point, min_corner, max_corner; reset = false)
    for i in eachindex(point)
        box_width = max_corner[i] - min_corner[i]

        # Shift point to origin-based coordinates
        shifted = point[i] - min_corner[i]

        # Wrap using modulo
        wrapped = mod(shifted, box_width)

        # Shift back to box coordinates
        point[i] = wrapped + min_corner[i]
    end

    if reset && !isnothing(velocity)
        velocity .= 0
    end

    return point
end

"""
Corrects the integrator cache after a boundary reset, by setting the stored history
difference to the difference of the post-update values. A no-op for caches without history
"""
function wrap_integrator_cache!(C, u, uprev)
    return
end
function wrap_integrator_cache!(integrator)
    return wrap_integrator_cache!(integrator.cache, integrator.u, integrator.uprev)
end
function getaffect(R::PeriodicBox{D, Re}) where {D, Re}
    return function affect!(integrator)
        vars = divide_dims(integrator.u, D)
        if length(vars) == 1
            x = vars[1] # ! Updated position
            reenterbox!(nothing, x, _corners(R)...; reset = Re)
        else
            x = vars[1]
            v = vars[2]
            reenterbox!(v, x, _corners(R)...; reset = Re)
        end
        return wrap_integrator_cache!(integrator)
    end
end

"""
    ReentrantBox(exit => reentrance; reset = true)
    ReentrantBox(exit => reentrance, exit => reentrance, ...; reset = true)

A half-periodic box: the trajectory is reset to `reentrance` whenever it passes `exit`,
and is unconstrained in the other direction.

Takes a pair per dimension rather than an interval, since an interval would not say which
end is the exit; passing an interval throws. With `reset = true`, the default, the velocity
is zeroed on re-entry.

```julia
ReentrantBox(4 => 0)           # exits at x = 4, re-enters at x = 0, free below
ReentrantBox(4 => 0, 2 => -2)  # two dimensions
```
"""
struct ReentrantBox{D, Re, T} <: AbstractBoxBoundary{D}
    reentrance::NTuple{D, T}
    exit::NTuple{D, T}
    relpos::NTuple{D, Bool}
    function ReentrantBox(
            reentrance::NTuple{D}, exit::NTuple{D};
            reset = true
        ) where {D}
        re, ex = _promote_corners(reentrance, exit)
        return new{D, reset, eltype(re)}(re, ex, Tuple(ex .> re))
    end
end
function ReentrantBox(
        exitreenter::Pair{<:Real, <:Real};
        kwargs...
    )
    return ReentrantBox((exitreenter.second,), (exitreenter.first,); kwargs...)
end
function ReentrantBox(
        exitreenter::Vararg{Pair{<:Real, <:Real}};
        kwargs...
    )
    return ReentrantBox(
        getproperty.(exitreenter, :second),
        getproperty.(exitreenter, :first); kwargs...
    )
end
function ReentrantBox(exitreenter; kwargs...) # Prevent careless inputs by forcing pairs
    throw(ArgumentError("Must provide pairs of enter=>reentrance values for each dimension, where all pairs have the same type."))
end
function ReentrantBox(exitreenter::AbstractInterval; kwargs...) # Prevent careless inputs by forcing pairs
    throw(ArgumentError("Must provide pairs of enter=>reentrance values for each dimension, where all pairs have the same type."))
end
function ReentrantBox(
        exitreenter::Pair{<:NTuple{D}, <:NTuple{D}};
        kwargs...
    ) where {D}
    return ReentrantBox(exitreenter.second, exitreenter.first; kwargs...)
end
_corners(R::ReentrantBox) = (R.reentrance, R.exit)
exits(R::ReentrantBox) = R.exit
reentrances(R::ReentrantBox) = R.reentrance
function getaffect(R::ReentrantBox{D, Re}) where {D, Re}
    reentrance = reentrances(R)
    return function affect!(integrator)
        vars = divide_dims(integrator.u, D)
        x = first(vars)
        for i in eachindex(x)
            x[i] = reentrance[i]
        end
        return if Re && length(vars) > 1
            vars[2] .= 0.0
        end
    end
end
function getcondition(R::ReentrantBox{D}) where {D}
    # Pre-compute values outside the inner function
    reentrance, exit = _corners(R)
    relpos = R.relpos
    return function condition(u, t, integrator)
        x = first_dims(u, D)

        for i in eachindex(x)
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
    return DiscreteCallback(
        getcondition(B), getaffect(B); save_positions = (false, true),
        kwargs...
    )
end

boundary_init(B::AbstractBoxBoundary; kwargs...) = B(; kwargs...)
boundary_init(::Nothing; kwargs...) = nothing
boundary_init(; kwargs...) = nothing
boundary_init(D::DECallback; kwargs...) = D

function domain(R::AbstractBoxBoundary)
    return [Interval(minmax(is...)...) for is in zip(_corners(R)...)]
end

"""
    gridaxes(B::AbstractBoxBoundary, n::Int)

One `range` of `n` points per dimension, spanning the box. [`grid`](@ref) takes their
product.
"""
function gridaxes(R::AbstractBoxBoundary{D}, n::Int) where {D}
    return map(domain(R)) do r
        range(r, length = n)
    end
end

"""
    grid(B::AbstractBoxBoundary, n)

An iterator over the `n`-per-dimension grid of points spanning the box, as the product of
[`gridaxes`](@ref). Useful for evaluating a density or an adaptation kernel over the region
a sampler is confined to.
"""
function grid(R::AbstractBoxBoundary, n)
    r = gridaxes(R, n)
    return Iterators.product(r...)
end

function (Base.in)(point, R::AbstractBoxBoundary)
    return !isoutside(point, _corners(R)...)
end

end # module
