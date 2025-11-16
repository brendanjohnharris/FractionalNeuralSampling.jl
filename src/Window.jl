# """
# Efficient circular buffer for in-practice fixed-size history storage.
# Index 1 = oldest, Index end = most recent
# """
# struct Window{T} <: AbstractVector{T}
#     data::Vector{T}
#     len::Int
#     head::Base.RefValue{Int}

#     function Window(data::Vector{T}) where {T}
#         new{T}(data, length(data), Ref(0))
#     end
# end
# Window(data::NTuple) = Window(collect(data))
# Window(u::T, len::Int) where {T} = Window(ntuple(_ -> zero(u), len))

# # Core functionality
# Base.size(w::Window) = (w.len,)
# Base.length(w::Window) = w.len

# # Circular indexing - index 1 is oldest, index end is most recent
# @inline function Base.getindex(w::Window, i::Int)
#     @boundscheck 1 <= i <= w.len || throw(BoundsError(w, i))
#     idx = mod1(w.head[] - (w.len - i), w.len)
#     @inbounds return w.data[idx]
# end

# @inline function Base.setindex!(w::Window, val, i::Int)
#     @boundscheck 1 <= i <= w.len || throw(BoundsError(w, i))
#     idx = mod1(w.head[] - (w.len - i), w.len)
#     @inbounds w.data[idx] = val
#     return val
# end

# # Push new element (drops oldest)
# @inline function Base.push!(w::Window, val)
#     w.head[] = mod1(w.head[] + 1, w.len)
#     @inbounds w.data[w.head[]] = val
#     return w
# end

# # Make it iterable (iterates from oldest to newest)
# function Base.iterate(w::Window, state=1)
#     state > w.len ? nothing : (w[state], state + 1)
# end

# # Support for vector operations
# Base.similar(w::Window, ::Type{T}, dims::Dims) where {T} = similar(w.data, T, dims)
# Base.similar(w::Window) = Window(similar(w.data))
# Base.copy(w::Window) = Window(copy(w.data))

# # For broadcasting
# Base.BroadcastStyle(::Type{<:Window}) = Broadcast.ArrayStyle{Window}()
# function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Window}},
#     ::Type{ElType}) where {ElType}
#     Window{ElType}(length(bc))
# end

# # Get underlying data in current order (for operations that need contiguous array)
# function as_vector(w::Window)
#     out = similar(w.data)
#     @inbounds for i in 1:(w.len)
#         out[i] = w[i]
#     end
#     return out
# end

# # Convenience methods
# Base.firstindex(w::Window) = 1
# Base.lastindex(w::Window) = w.len

# # Get most recent element (same as w[end])
# current(w::Window) = w[end]

# # Get oldest element (same as w[1])
# oldest(w::Window) = w[1]


# # ...................................
# """
# Efficient circular buffer for in-practice fixed-size history storage.
# Index 1 = oldest, Index end = most recent
# """
# struct Window{T} <: AbstractVector{T}
#     data::Vector{T}
#     len::Int
#     head::Base.RefValue{Int}

#     function Window(data::Vector{T}) where {T}
#         new{T}(data, length(data), Ref(0))
#     end
# end

# Window(data::NTuple) = Window(collect(data))
# Window(u::T, len::Int) where {T} = Window(ntuple(_ -> zero(u), len))

# # Core functionality
# Base.size(w::Window) = (w.len,)
# Base.length(w::Window) = w.len

# # Fast circular indexing without mod1
# @inline function Base.getindex(w::Window, i::Int)
#     @boundscheck 1 <= i <= w.len || throw(BoundsError(w, i))
#     idx = w.head[] - w.len + i
#     idx = ifelse(idx > 0, idx, idx + w.len)
#     @inbounds return w.data[idx]
# end

# @inline function Base.setindex!(w::Window, val, i::Int)
#     @boundscheck 1 <= i <= w.len || throw(BoundsError(w, i))
#     idx = w.head[] - w.len + i
#     idx = ifelse(idx > 0, idx, idx + w.len)
#     @inbounds w.data[idx] = val
#     return val
# end

# # Fast push without mod1
# @inline function Base.push!(w::Window, val)
#     w.head[] = ifelse(w.head[] == w.len, 1, w.head[] + 1)
#     @inbounds w.data[w.head[]] = val
#     return w
# end

# # Make it iterable (iterates from oldest to newest)
# function Base.iterate(w::Window, state=1)
#     state > w.len ? nothing : (w[state], state + 1)
# end

# # Support for vector operations
# Base.similar(w::Window, ::Type{T}, dims::Dims) where {T} = similar(w.data, T, dims)
# Base.similar(w::Window) = Window(similar(w.data))
# Base.copy(w::Window) = Window(copy(w.data))

# # For broadcasting
# Base.BroadcastStyle(::Type{<:Window}) = Broadcast.ArrayStyle{Window}()
# function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Window}},
#     ::Type{ElType}) where {ElType}
#     Window{ElType}(length(bc))
# end

# # Get underlying data in current order (for operations that need contiguous array)
# function as_vector(w::Window)
#     out = similar(w.data)
#     @inbounds for i in 1:(w.len)
#         out[i] = w[i]
#     end
#     return out
# end

# # Convenience methods
# Base.firstindex(w::Window) = 1
# Base.lastindex(w::Window) = w.len

# # Get most recent element (same as w[end])
# current(w::Window) = w[end]

# # Get oldest element (same as w[1])
# oldest(w::Window) = w[1]



"""
Optimized Window implementations for Julia
Choose based on your use case
"""

# ==============================================================================
# RECOMMENDED: Best general-purpose implementation
# ==============================================================================
"""
    Window{T}

High-performance circular buffer with O(1) push and access.
- Index 1 = oldest element
- Index end = most recent element
- Optimized with branchless arithmetic and direct field access
"""
mutable struct Window{T} <: AbstractVector{T}
    data::Vector{T}
    len::Int
    head::Int

    function Window{T}(len::Int) where {T}
        new{T}(Vector{T}(undef, len), len, 0)
    end

    function Window(data::Vector{T}) where {T}
        new{T}(data, length(data), 0)
    end
end

Window(data::NTuple{N,T}) where {N,T} = Window(collect(data))
Window(u::T, len::Int) where {T} = Window(fill(zero(u), len))

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

# Iteration support
function Base.iterate(w::Window, state=1)
    state > w.len ? nothing : (@inbounds(w[state]), state + 1)
end

# Vector operations
Base.similar(w::Window, ::Type{T}, dims::Dims) where {T} = similar(w.data, T, dims)
Base.similar(w::Window) = Window(similar(w.data))
Base.copy(w::Window) = Window(copy(w.data))

# Broadcasting
Base.BroadcastStyle(::Type{<:Window}) = Broadcast.ArrayStyle{Window}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Window}},
    ::Type{ElType}) where {ElType}
    Window{ElType}(length(bc))
end

# Convenience methods
Base.firstindex(w::Window) = 1
Base.lastindex(w::Window) = w.len
@inline current(w::Window) = @inbounds w[w.len]
@inline oldest(w::Window) = @inbounds w[1]

# Convert to contiguous array
function as_vector(w::Window)
    out = similar(w.data)
    @inbounds for i in 1:w.len
        out[i] = w[i]
    end
    return out
end

# # ==============================================================================
# # SPECIALIZED: Power-of-2 sizes (even faster)
# # ==============================================================================
# """
#     WindowP2{T,N}

# Ultra-fast circular buffer for power-of-2 sizes.
# Uses bitwise operations for maximum performance.
# """
# mutable struct WindowP2{T,N} <: AbstractVector{T}
#     data::Vector{T}
#     mask::Int
#     head::Int

#     function WindowP2{T,N}() where {T,N}
#         ispow2(N) || error("N must be a power of 2")
#         new{T,N}(Vector{T}(undef, N), N - 1, 0)
#     end

#     function WindowP2(data::Vector{T}) where {T}
#         N = length(data)
#         ispow2(N) || error("Length must be a power of 2")
#         new{T,N}(data, N - 1, 0)
#     end
# end

# Base.size(::WindowP2{T,N}) where {T,N} = (N,)
# Base.length(::WindowP2{T,N}) where {T,N} = N

# @inline function Base.getindex(w::WindowP2{T,N}, i::Int) where {T,N}
#     @boundscheck 1 <= i <= N || throw(BoundsError(w, i))
#     # Bitwise AND for ultra-fast modulo
#     idx = ((w.head - N + i - 1) & w.mask) + 1
#     @inbounds return w.data[idx]
# end

# @inline function Base.setindex!(w::WindowP2{T,N}, val, i::Int) where {T,N}
#     @boundscheck 1 <= i <= N || throw(BoundsError(w, i))
#     idx = ((w.head - N + i - 1) & w.mask) + 1
#     @inbounds w.data[idx] = val
#     return val
# end

# @inline function Base.push!(w::WindowP2, val)
#     w.head = (w.head & w.mask) + 1
#     @inbounds w.data[w.head] = val
#     return w
# end
