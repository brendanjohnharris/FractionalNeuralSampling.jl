using FFTW
using Random
using SpecialFunctions
using Statistics
using StableDistributions

export lfsn, lfsm

"""
    a_tilda(N, m, M, α, H)

Compute the kernel coefficients for LFSM generation.
Based on the discretization of (t-s)₊^(H-1/α).
"""
function a_tilda(N::Int, m::Int, M::Int, α::Float64, H::Float64)
    # Create vector with zeros padding
    a = zeros(Float64, m * (M + N))

    X2 = m^(-1 / α)
    Ha = H - 1 / α

    # First m coefficients
    m_inv_Ha = (1 / m)^Ha
    scale = m_inv_Ha * X2
    @inbounds for j in 1:m
        a[j] = j^Ha * scale
    end

    # Remaining coefficients up to m*M
    @inbounds for j in (m + 1):(m * M)
        a[j] = (j^Ha - (j - m)^Ha) * scale
    end

    return a
end

const FLSN_SCALE = 4 * erfinv(0.5) # Gives a variance of sqrt(2) for the Gaussian case, since IQR of a standard Normal is 2*sqrt(2)*erfinv(0.5)

"""
    lfsn(N, m, M, α, H; sigma=1.0, rng=Random.default_rng())

Generate Linear Fractional Stable Noise (LFSN) .

# Arguments
- `N::Int`: Number of points of the LFSM
- `m::Int`: Discretization parameter (points between motion points)
- `M::Int`: Truncation parameter (lookback window)
- `α::Float64`: Stability parameter ∈ (0, 2]
- `H::Float64`: Hurst parameter ∈ (0, 1)
- `sigma::Float64=1.0`: Scale parameter
- `rng`: Random number generator (default: `Random.default_rng()`)

"""
# function lfsn(N::Int, α::Float64, H::Float64; m::Int = 128, M::Int = 1000,
#               sigma::Float64 = 1.0, rng = Random.default_rng(),
#               dt = 1)

#     # * Choose faster parameters based on inputs
#     total_length = m * (N + M)
#     next_pow_2 = 2^ceil(Int, log2(total_length))

#     # * Choose even m
#     m = iseven(m) ? m : m + 1

#     # * Choose N to make pow2 samples
#     _N = N
#     N = next_pow_2 ÷ m - M
#     total_length = m * (N + M)
#     @assert ispow2(total_length)
#     @assert next_pow_2 == total_length

#     # Validate parameters
#     @assert 0<α<=2 "α must be in (0, 2]"
#     @assert 0<H<1 "H must be in (0, 1)"
#     @assert sigma>0 "sigma must be positive"
#     @assert N > 0&&m > 0 && M > 0 "N, m, M must be positive"

#     # Generate stable random variables (Lévy increments)
#     # Stable(α, β, scale, location) with β=0 for symmetric
#     d = Stable(α, 0.0, 1.0, 0.0)
#     Ẑ = rand(rng, d, total_length) |> complex
#     ℱ = plan_fft!(Ẑ)
#     ℱ * Ẑ  # In-place FFT of Ẑ

#     # Compute kernel coefficients
#     â = a_tilda(N, m, M, α, H) |> complex
#     â .*= sigma
#     ℱ * â

#     # Multiply in frequency domain
#     @. â *= Ẑ

#     # Apply inverse FFT (ifft already includes normalization)
#     ifft!(â)

#     # Extract the relevant portion (skip first m*M points)
#     W = view(â, (m * M + 1):(m * M + m * N))

#     # Sample at every m-th point
#     index_m = m:m:length(W)
#     Z = map(real, view(W, index_m))

#     m = median(Z)
#     iqr = quantile(Z, 0.75) - quantile(Z, 0.25)
#     @. Z = FLSN_SCALE * (Z - m) / iqr  # Robust std

#     # Get middle _N samples
#     idxs = ((N - _N) ÷ 2):(length(Z) - (N - _N) ÷ 2 - 1)
#     Z = view(Z, idxs)

#     @assert length(Z) == _N

#     Z .*= dt^(H)
#     return Z
# end

lfsm(args...; kwargs...) = cumsum(lfsn(args...; kwargs...))

function lfsn(N::Int, α::A, H::B; m::Int = 128, M::Int = 1000,
              sigma = 1.0, rng = Random.default_rng(),
              dt = 1) where {A <: Real, B <: Real}
    T = promote_type(A, B)
    total_length = m * (N + M)
    next_pow_2 = 2^ceil(Int, log2(total_length))
    m = iseven(m) ? m : m + 1
    _N = N
    N = next_pow_2 ÷ m - M
    total_length = m * (N + M)

    # Validate parameters
    @assert 0<α<=2 "α must be in (0, 2]"
    @assert 0<H<1 "H must be in (0, 1)"
    @assert sigma>0 "sigma must be positive"
    @assert N > 0&&m > 0 && M > 0 "N, m, M must be positive"

    # Pre-allocate all arrays as complex from the start
    Ẑ = Vector{Complex{T}}(undef, total_length)
    â = Vector{Complex{T}}(undef, total_length)
    result = Vector{T}(undef, _N)

    # Fill kernel coefficients (directly as complex)
    X2 = m^(-1 / α)
    Ha = H - 1 / α
    m_inv_Ha = (1 / m)^Ha
    scale = m_inv_Ha * X2 * sigma
    fill!(â, Complex(0.0, 0.0))

    # First m coefficients
    @inbounds for j in 1:m
        â[j] = Complex(j^Ha * scale, 0.0)
    end

    # Pre-compute powers for j values
    j_powers = Vector{T}(undef, m * M)
    for j in 1:(m * M)
        j_powers[j] = j^Ha
    end

    # Remaining coefficients up to m*M
    @inbounds for j in (m + 1):(m * M)
        â[j] = Complex((j_powers[j] - j_powers[j - m]) * scale, 0.0)
    end

    # Fill with Lévy increments
    d = Stable(α, 0.0, 1.0, 0.0)
    for i in eachindex(Ẑ)
        Ẑ[i] = Complex(rand(rng, d), 0.0)
    end

    ℱ = plan_fft!(Ẑ)
    ℱ * Ẑ
    ℱ * â
    â .*= Ẑ
    ifft!(â)

    # Extract real parts at every m-th point directly into result
    offset = m * M
    idx = 1
    for i in 1:N
        if idx <= _N
            pos = offset + i * m
            result[idx] = real(â[pos])
            idx += 1
        end
    end

    # Normalize in-place
    med = median(result)
    iqr = quantile(result, 0.75) - quantile(result, 0.25)
    scale_factor = dt^(H) * FLSN_SCALE / iqr
    @. result = (result - med) * scale_factor

    return result
end
