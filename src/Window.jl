"""
    Window{T}

Fixed-length circular buffer with O(1) push and access.
Index 1 is the oldest element and index `end` the most recent.
"""
mutable struct Window{T} <: AbstractVector{T}
    data::Vector{T}
    len::Int
    head::Int

    function Window{T}(len::Int) where {T}
        return new{T}(Vector{T}(undef, len), len, 0)
    end

    function Window(data::Vector{T}) where {T}
        return new{T}(data, length(data), 0)
    end
end

Window(data::NTuple) = Window(collect(data))
Window(u::T, len::Int) where {T} = Window([zero(u) for _ in 1:len]) # Not `fill`: mutable `u` would alias

# Core functionality with optimizations
Base.size(w::Window) = (w.len,)
Base.length(w::Window) = w.len

# Optimized getindex with branchless arithmetic
@inline function Base.getindex(w::Window, i::Int)
    @boundscheck 1 <= i <= w.len || throw(BoundsError(w, i))
    idx = w.head - w.len + i
    # Branchless: add len if idx <= 0
    idx += w.len * (idx <= 0)
    @inbounds return w.data[idx]
end

# Optimized setindex
@inline function Base.setindex!(w::Window, val, i::Int)
    @boundscheck 1 <= i <= w.len || throw(BoundsError(w, i))
    idx = w.head - w.len + i
    idx += w.len * (idx <= 0)
    @inbounds w.data[idx] = val
    return val
end

# Optimized push with branchless wrap
@inline function Base.push!(w::Window, val)
    # Branchless increment: add 1, subtract len if we hit the boundary
    w.head = w.head + 1 - w.len * (w.head == w.len)
    @inbounds w.data[w.head] = val
    return w
end

"""
    roll!(w::Window)

Advance the head and return the (stale) slot now at index `end`, for callers that
overwrite the slot in place rather than allocating a new element.
"""
@inline function roll!(w::Window)
    w.head = w.head + 1 - w.len * (w.head == w.len)
    return @inbounds w.data[w.head]
end

# Iteration support
function Base.iterate(w::Window, state = 1)
    return state > w.len ? nothing : (@inbounds(w[state]), state + 1)
end

# Vector operations
Base.similar(w::Window, ::Type{T}, dims::Dims) where {T} = similar(w.data, T, dims)
Base.similar(w::Window) = Window(similar(w.data))
Base.copy(w::Window) = Window(copy(w.data))

# Broadcasting
Base.BroadcastStyle(::Type{<:Window}) = Broadcast.ArrayStyle{Window}()
function Base.similar(
        bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Window}},
        ::Type{ElType}
    ) where {ElType}
    return Window{ElType}(length(bc))
end

# Convenience methods
Base.firstindex(w::Window) = 1
Base.lastindex(w::Window) = w.len
@inline current(w::Window) = @inbounds w[w.len]
@inline oldest(w::Window) = @inbounds w[1]

# Convert to contiguous array
function as_vector(w::Window)
    out = similar(w.data)
    @inbounds for i in 1:(w.len)
        out[i] = w[i]
    end
    return out
end
