using FractionalNeuralSampling
using Test
using Hurst

begin  # Test Hurst exponent
    α = 1.5
    β = 0.8
    H = 1 - β / 2
    x = lfsn(100000, α, H; dt = 0.01) |> cumsum
    H̃ = hurst_exponent(x, 1:10)
    @test abs(first(H̃) - H) < 0.05
end

using FFTW
using Random
using Statistics
function fractional_gaussian_noise(N::Int, H::Float64; rng = Random.GLOBAL_RNG, dt = 1)
    # Validate inputs
    @assert 0<H<1 "Hurst exponent H must be in (0, 1)"
    @assert N>0 "N must be positive"

    # Autocovariance function for fGn
    function gamma_fgn(k, H)
        return 0.5 * (abs(k - 1)^(2H) - 2 * abs(k)^(2H) + abs(k + 1)^(2H))
    end

    # Build the circulant covariance matrix first row
    # We need 2N values for the circulant embedding
    g = zeros(2N)
    g[1] = 1.0  # variance at lag 0

    for k in 1:N
        g[k + 1] = gamma_fgn(k, H)
    end

    # Fill the second half (circulant symmetry)
    for k in 1:(N - 1)
        g[2N - k + 1] = g[k + 1]
    end

    # Compute eigenvalues via FFT
    lambda = real(fft(g))

    # Check for non-negative eigenvalues (should be satisfied for valid H)
    if any(lambda .< -1e-10)
        @warn "Negative eigenvalues detected. Results may be inaccurate."
    end
    lambda = max.(lambda, 0)  # Ensure non-negative

    # Generate white noise in frequency domain
    V1 = randn(rng, N + 1)
    V2 = randn(rng, N - 1)

    # Construct the frequency domain representation
    W = zeros(ComplexF64, 2N)
    W[1] = sqrt(lambda[1] / (2N)) * V1[1]
    W[N + 1] = sqrt(lambda[N + 1] / (2N)) * V1[N + 1]

    for k in 2:N
        W[k] = sqrt(lambda[k] / (4N)) * (V1[k] + im * V2[k - 1])
        W[2N - k + 2] = conj(W[k])
    end

    # Inverse FFT to get the fGn
    Z = real(ifft(W))

    # Return middle N samples
    idxs = (length(Z) ÷ 4 + 1):(length(Z) * 3 ÷ 4)
    Z = Z[idxs]
    Z = (Z .- mean(Z)) ./ std(Z)
    Z = Z .* dt^(H)  # Scale by dt^H to account for time step
    return Z
end

begin # * Test against fgn
    α = 2.0
    β = 0.8
    H = 1 - β / 2
    N = 100000
    dt = 0.01
    x = lfsn(N, α, H; dt) |> cumsum
    y = fractional_gaussian_noise(N, H; dt) |> cumsum

    H̃x = hurst_exponent(x, 1:10)
    H̃y = hurst_exponent(y, 1:10)
    @test first(H̃x)≈first(H̃y) atol=0.05
end

for α in (1.3, 1.5, 1.8, 2.0)
    for β in (0.3, 0.5, 0.8)
        H = 1 - β / 2
        x = lfsn(100000, α, H; dt = 0.01) |> cumsum
        H̃ = hurst_exponent(x, 1:10)
        @test first(H̃)≈H rtol=0.15
    end
end

begin # * Compare with levy process
    using StableDistributions
    β = 1.0
    for α in [1.2, 1.5, 1.8, 2.0]
        H = β - 1 + 1 / α #  1 - β / 2
        dt = 0.01
        N = 20000

        x = lfsn(N, α, H; dt)
        D = Stable(α, 0.0, 1.0, 0.0)
        y = rand(D, N) .* dt^(1 / α)

        dx = fit(Stable, x)
        dy = fit(Stable, y)
        @test dx.α≈dy.α rtol=0.1
        @test dx.α≈α rtol=0.1
        @test dx.σ≈dy.σ rtol=0.075
    end
end
