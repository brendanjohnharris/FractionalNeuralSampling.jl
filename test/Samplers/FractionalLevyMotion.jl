using StableDistributions
using LinearAlgebra
using CairoMakie
using DiffEqNoiseProcess

begin
    α = 2.0
    β = 0.0
    γ = 0.01
    dt = 0.025
    tspan = 1000.0
    xmax = 6
    σ = 0.5
    A = 1.0
    ND = 2
    H = 0.2

    function param_check(N, dt, tspan)
        if dt !== nothing && tspan !== nothing
            # (# of time points) - (# of time intervals) = 1
            N = Int(floor(tspan / dt)) + 1
        elseif N !== nothing && tspan !== nothing
            dt = tspan / N
        elseif N !== nothing && dt !== nothing
            tspan = N * dt
        else
            error("At least two of N, tspan, and dt must be provided.")
        end
        return N, dt, tspan
    end

    # LEVY NOISE
    # Generates a vector noise sampled from a Levy Alpha Stable Distribution with parameters:
    # α ∈ [1,2]: Tail index
    # β ∈ [-1,1]: Skewness
    # γ > 0: Scale parameter
    # δ: Location parameter
    function levyNoise(α = 1.5, dt = 1, tspan = 1000; β = 0, γ = 1, δ = 0, ND = 1,
                       max = nothing, seed = nothing)
        N = Int(floor(tspan / dt)) + 1
        dist = Stable(α, β, γ, δ)
        if (seed !== nothing)
            Random.seed!(seed)
        end
        ξ = rand(dist, N)

        if (ND == 1)
            x = ξ
        elseif (ND == 2)
            θ = (2π) .* rand(N)
            x = hcat(ξ .* cos.(θ), ξ .* sin.(θ))
        else
            error("ND must be 1 or 2")
        end

        if (max !== nothing)
            x = clampComponents(x, max)
        end

        # For standard Levy Motion, B(0) = 0
        x[1, :] .= 0

        return x
    end

    function FractionalLM(H::Float64, α::Float64; dt = nothing, tspan = nothing,
                          N = nothing,
                          μ = 1.0, maxStep = nothing, ND = 1, method = :slice,
                          uncorNoise = false, seed = nothing, k = 1.0)
        if (μ < 0 || μ > 1)
            error("μ must be in the range [0,1]")
        end

        # Must have at least 2 of 3 (N, dt, tspan)
        N, dt, tspan = param_check(N, dt, tspan)

        x = zeros(N, ND)

        # Construct a list of N random steps from Levy-α distribution
        dL = levyNoise(α, dt, tspan, ND = ND, max = maxStep, seed = seed)

        # Determine the lower time bound for integration at each step
        # i_mins = [max(1, round(Int, i - 1 - μ * N)) for i in 1:N]

        # Compute kernel exponent
        p = (2 * H - 1) / α

        # Precompute scaling factor
        # G = (dt^(p+1))/(gamma(p+1.0) * (p+1.0))
        G = (dt^(p + 1 - 1 / α)) / (p + 1.0)
        # G = (dt^(p+1))

        # VECTOR SLICE METHOD: O(N) SPACE
        if (method == :slice)
            # This constructs the last column of the weight matrix
            #  All other columns are slices of this column
            v = [(N .- i + 1)^(p + 1.0) .- (N .- i) .^ (p + 1.0) for i in 1:N]
            for i in 1:N
                if (ND == 1)
                    x[i, 1] = G .* dot(dL[1:i, 1], v[(end - i + 1):end])
                elseif (ND == 2)
                    x[i, 1] = G .* dot(dL[1:i, 1], v[(end - i + 1):end])
                    x[i, 2] = G .* dot(dL[1:i, 2], v[(end - i + 1):end])
                end
            end
        end

        # MATRIX METHOD: O(N^2) SPACE
        if (method == :matrix)
            v_mat = zeros(N, N)

            for j in 1:N
                for i in 1:j
                    v_mat[i, j] = (j - i + 1)^(p + 1.0) - ((j - i + 1) - 1)^(p + 1.0)
                end
            end

            if (ND == 1)
                x = G .* (dL[2:end, 1]' * v_mat)'
            elseif (ND == 2)
                x[:, 1] = G .* (dL[2:end, 1]' * v_mat)'
                x[:, 2] = G .* (dL[2:end, 2]' * v_mat)'
            end
        end

        # TODO: NORMALISATION FACTOR BASED ON FBM
        # x = x / dt

        if (uncorNoise)
            return x, dL
        end
        # return x ./ 10
        return x * k
    end
end

fLM = FractionalLM(H, α, dt = dt, tspan = tspan, μ = 1.0, maxStep = nothing, ND = ND)


begin # * Plot
    x = fLM[:, 1]

    lines((x[1:10000]))
end

function noiseMatToNoiseGrid(X; dt = nothing, tspan = nothing, t = nothing,)
    if (t === nothing)
        if (tspan === nothing || dt === nothing)
            error("You must provide either a vector of times t, or a simulation time tspan and a time step dt")
        end
        t = collect(0:dt:tspan)
    end
    # Add columns for the momentum coords (no noise)
    X_padded = hcat(X, zeros(size(X)))
    # Convert Matrix to Vector of 4x4 Matrices (Noise on momenta)
    v = vec([Diagonal(X_padded[i, :]) for i in 1:size(X_padded, 1)])

    # v = NoiseGrid(t, v)

    return v
end
xx = noiseMatToNoiseGrid(fLM, dt = dt, tspan = tspan)
