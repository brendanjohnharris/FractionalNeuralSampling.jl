"""
Efficient circular buffer for in-practice fixed-size history storage.
Index 1 = oldest, Index end = most recent
"""
struct Window{T} <: AbstractVector{T}
    data::Vector{T}
    len::Int
    head::Base.RefValue{Int}

    function Window(data::Vector{T}) where {T}
        new{T}(data, length(data), Ref(0))
    end
end
Window(data::NTuple{N, T}) where {N, T} = Window(collect(data))
Window(u::T, len::Int) where {T} = Window(ntuple(_ -> zero(u), len))

# Core functionality
Base.size(w::Window) = (w.len,)
Base.length(w::Window) = w.len

# Circular indexing - index 1 is oldest, index end is most recent
@inline function Base.getindex(w::Window, i::Int)
    @boundscheck 1 <= i <= w.len || throw(BoundsError(w, i))
    idx = mod1(w.head[] - (w.len - i), w.len)
    @inbounds return w.data[idx]
end

@inline function Base.setindex!(w::Window, val, i::Int)
    @boundscheck 1 <= i <= w.len || throw(BoundsError(w, i))
    idx = mod1(w.head[] - (w.len - i), w.len)
    @inbounds w.data[idx] = val
    return val
end

# Push new element (drops oldest)
@inline function Base.push!(w::Window, val)
    w.head[] = mod1(w.head[] + 1, w.len)
    @inbounds w.data[w.head[]] = val
    return w
end

# Make it iterable (iterates from oldest to newest)
function Base.iterate(w::Window, state = 1)
    state > w.len ? nothing : (w[state], state + 1)
end

# Support for vector operations
Base.similar(w::Window, ::Type{T}, dims::Dims) where {T} = similar(w.data, T, dims)
Base.similar(w::Window) = Window(similar(w.data))
Base.copy(w::Window) = Window(copy(w.data))

# For broadcasting
Base.BroadcastStyle(::Type{<:Window}) = Broadcast.ArrayStyle{Window}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Window}},
                      ::Type{ElType}) where {ElType}
    Window{ElType}(length(bc))
end

# Get underlying data in current order (for operations that need contiguous array)
function as_vector(w::Window)
    out = similar(w.data)
    @inbounds for i in 1:(w.len)
        out[i] = w[i]
    end
    return out
end

# Convenience methods
Base.firstindex(w::Window) = 1
Base.lastindex(w::Window) = w.len

# Get most recent element (same as w[end])
current(w::Window) = w[end]

# Get oldest element (same as w[1])
oldest(w::Window) = w[1]
