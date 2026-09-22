using FFTW
using Random
using SpecialFunctions
using Statistics
using StableDistributions

export lfsn, lfsm

const FLSN_SCALE = 4 * erfinv(0.5) # Gives a variance of sqrt(2) for the Gaussian case, since IQR of a standard Normal is 2*sqrt(2)*erfinv(0.5)

"""
    lfsm(N, α, H; m = 128, M = 1000, sigma = 1.0, dt = 1, rng = Random.default_rng())

Linear fractional stable motion: the cumulative sum of [`lfsn`](@ref).

The stable-law analogue of fractional Brownian motion, self-similar with Hurst exponent `H`
and built from α-stable increments. `H` controls the correlation between successive
increments, so `H > 1/2` gives persistent excursions and `H < 1/2` antipersistent ones,
while `α` controls their tails. The two exponents are set independently.

This is the noise the fractional-in-time samplers need: [`tFOLE`](@ref) takes α = 2 with
H = 1 - β/2, and [`bFOLE`](@ref) and [`bFNS`](@ref) take H = 1/2 - β/2 + 1/α.

See [`lfsn`](@ref) for the arguments.
"""
lfsm(args...; kwargs...) = cumsum(lfsn(args...; kwargs...))

"""
    lfsn(N, α, H; m = 128, M = 1000, sigma = 1.0, dt = 1, rng = Random.default_rng())

Generate linear fractional stable noise (LFSN). The result has a `scale` of 1 at `dt = 1`,
so for β = 1 it corresponds to draws from a Lévy distribution with σ = 1.

# Arguments
- `N::Int`: Number of points to return
- `α::Real`: Stability parameter ∈ (0, 2]
- `H::Real`: Hurst parameter ∈ (0, 1)
- `m::Int = 128`: Discretization parameter (points between motion points); rounded up to even
- `M::Int = 1000`: Truncation parameter (lookback window)
- `sigma = 1.0`: Scale parameter
- `dt = 1`: Time step, which scales the result by `dt^H`
- `rng`: Random number generator (default: `Random.default_rng()`)

!!! note
    The in-place plan used here segfaults under multithreaded FFTW
    (JuliaMath/FFTW.jl#236), so the transform runs through
    [`FractionalNeuralSampling.serial_fftw`](@ref); the caller's thread count is restored
    afterwards.
"""
function lfsn(
        N::Int, α::A, H::B; m::Int = 128, M::Int = 1000,
        sigma = 1.0, rng = Random.default_rng(),
        dt = 1
    ) where {A <: Real, B <: Real}
    # Validate parameters
    @assert 0 < α <= 2 "α must be in (0, 2]"
    @assert 0 < H < 1 "H must be in (0, 1); got α=$α, H=$H)"
    @assert sigma > 0 "sigma must be positive"
    @assert N > 0&&m > 0 && M > 0 "N, m, M must be positive"

    T = promote_type(A, B)
    m = iseven(m) ? m : m + 1
    # Pad to (a multiple of m just under) a power of two, for the FFT. The padding is
    # applied after m is made even, so the padded series always covers the requested N
    total_length = m * (2^ceil(Int, log2(m * (N + M))) ÷ m)

    # Pre-allocate all arrays as complex from the start
    Ẑ = Vector{Complex{T}}(undef, total_length)
    â = Vector{Complex{T}}(undef, total_length)
    result = Vector{T}(undef, N)
    # Fill kernel coefficients (directly as complex)
    X2 = m^(-1 / α)
    Ha = H - 1 / α
    m_inv_Ha = (1 / m)^Ha
    scale = m_inv_Ha * X2 * sigma
    fill!(â, Complex(zero(T), zero(T)))

    # First m coefficients
    @inbounds for j in 1:m
        â[j] = Complex(j^Ha * scale, zero(T))
    end

    # Pre-compute powers for j values
    j_powers = Vector{T}(undef, m * M)
    for j in 1:(m * M)
        j_powers[j] = j^Ha
    end

    # Remaining coefficients up to m*M
    @inbounds for j in (m + 1):(m * M)
        â[j] = Complex((j_powers[j] - j_powers[j - m]) * scale, zero(T))
    end

    # Fill with Lévy increments
    d = Stable(α, zero(T), one(T), zero(T))
    for i in eachindex(Ẑ)
        Ẑ[i] = Complex(rand(rng, d), zero(T))
    end

    serial_fftw() do
        ℱ = plan_fft!(Ẑ)
        ℱ * Ẑ
        ℱ * â
        â .*= Ẑ
        ifft!(â)
    end

    # Extract real parts at every m-th point directly into result
    offset = m * M
    @inbounds for i in 1:N
        result[i] = real(â[offset + i * m])
    end

    # Normalize in-place
    med = median(result)
    iqr = quantile(result, 0.75) - quantile(result, 0.25)
    scale_factor = dt^(H) * FLSN_SCALE / iqr
    @. result = (result - med) * scale_factor

    return result
end
