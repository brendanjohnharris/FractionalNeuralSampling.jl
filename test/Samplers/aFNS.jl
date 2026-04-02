using FractionalNeuralSampling
using StochasticDiffEq
using DiffEqCallbacks
using Distributions
using TimeseriesTools
using LinearAlgebra
using ApproxFun
using CairoMakie
using Foresight
Foresight.set_theme!(Foresight.foresight(:physics))

import FractionalNeuralSampling: Density

# * Parameters
begin
    dt = 0.001
    tspan = 50.0
    γ = 50
    α = 1.8
    τ_r = 0.001
    τ_d = 0.2
    σ = 0.5
    approx_n_modes = 100

    domain = -5 .. 5
    boundaries = ReflectingBox(domain)
    𝜋 = MixtureModel([Normal(-2, 0.2), Normal(2, 0.2)]) |> Density
    u0 = [0.0]

    # Repulsive Gaussian kernel: pushes the particle away from recently visited locations
    kernel(x) = exp(-x^2 / σ)
    # end

    # # * Run adaptive Lévy sampler (with kernel saving)
    # begin

    # Save snapshots of the adaptive kernel coefficients
    saved_aK = SavedValues(Float64, Vector{Float64})
    save_callback = SavingCallback((u, t, integrator) -> copy(integrator.p[1].a_K),
                                   saved_aK;
                                   saveat = 0.0:dt:tspan)

    S = AdaptiveLevySampler(kernel, approx_n_modes;
                            α, γ, τ_r, τ_d, u0, 𝜋, tspan, boundaries, dt,
                            callback = (save_callback,))

    sol = solve(S, EM(); dt)
    x = Timeseries(sol) |> eachcol |> first
    x = x[𝑡 = Near(0:(dt):last(times(x)))]
    x = rectify(x, dims = 𝑡; tol = 0)

    # Retrieve the Fourier space from the sampler parameters
    sp = S.p[1].sp

    # Evaluate the effective potential V_eff(x, t) = -log π (x) + K(x, t) on a grid
    xgrid_hm = range(-5, 5, length = 100)
    t_saved = saved_aK.t
    logπ_vals = [logdensity(𝜋, xi) for xi in xgrid_hm]

    # Subsample time for the heatmap (every n-th saved step)
    thin = max(1, length(t_saved) ÷ 5000)
    t_hm = t_saved[1:thin:end]
    aK_hm = saved_aK.saveval[1:thin:end]

    # Build the effective potential matrix: V_eff[i_x, i_t]
    V_eff = Matrix{Float64}(undef, length(xgrid_hm), length(t_hm))
    for (j, aK_j) in enumerate(aK_hm)
        K_j = Fun(sp, aK_j)
        for (i, xi) in enumerate(xgrid_hm)
            V_eff[i, j] = K_j(xi) # - logπ_vals[i]
        end
    end
    # end

    # # * Plot 1: Time series with effective potential heatmap
    # begin
    f = Figure(size = (900, 400))
    ax1 = Axis(f[1, 1]; xlabel = "Time", ylabel = "x",
               title = "Adaptive Lévy Sampler (α = $α)")

    imax = findfirst(t_hm .>= 2)
    hm = heatmap!(ax1, collect(t_hm)[1:imax], collect(xgrid_hm),
                  V_eff'[1:imax, :];
                  colormap = :bone)

    # Trajectory on top
    lines!(ax1, x[𝑡 = 0 .. 2]; linewidth = 1, color = :crimson)
    hlines!(ax1, [-2, 2]; color = :cornflowerblue, linestyle = :dash, linewidth = 2)
    xlims!(ax1, 0, min(tspan, 2))
    ylims!(ax1, -5, 5)

    Colorbar(f[1, 2], hm)

    ax = Axis(f[1, 3]; xlabel = "Frequency (Hz)", ylabel = "Power")
    S = spectrum(x, 0.5; padding = 500)
    plotspectrum!(ax, S)
    colsize!(f.layout, 1, Relative(0.6))
    display(f)
end

# * Plot 2: Histogram vs target density
begin
    f = Figure(size = (600, 400))
    ax = Axis(f[1, 1]; xlabel = "x", ylabel = "Density",
              title = "Sampled vs Target Distribution")
    hist!(ax, collect(x); normalization = :pdf, bins = -5:0.1:5, color = (:steelblue, 0.6))
    xgrid = -5:0.01:5
    lines!(ax, xgrid, 𝜋.(xgrid); color = :red, linewidth = 2, label = "Target")
    axislegend(ax)
    display(f)
end

# * Dwell time analysis
#   A "dwell" is a contiguous period spent in one basin (x < 0 or x > 0)
begin
    signs = sign.(collect(x))
    transitions = diff(signs) .!= 0
    transition_idx = findall(transitions)
    dwell_times = diff(transition_idx) .* dt

    f = Figure(size = (600, 400))
    ax = Axis(f[1, 1]; xlabel = "Dwell time", ylabel = "Density",
              title = "Dwell Time Distribution")
    ziggurat!(ax, dwell_times;
              color = (:cornflowerblue, 0.7), bins = 0.05:0.05:1, normalization = :pdf)
    display(f)
end
