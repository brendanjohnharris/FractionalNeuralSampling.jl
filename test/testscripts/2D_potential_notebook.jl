### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 543309cc-7fdc-4222-9397-bda7393b3d95
begin # Imports
    using CairoMakie
    using Foresight
    using DifferentialEquations
    using FractionalNeuralSampling
    using Distributions
    using LinearAlgebra
    import FractionalNeuralSampling: Density
    set_theme!(foresight(:physics))
end

# ╔═╡ 393e4f89-5c2b-4cd7-a5e6-e6826b61cd93
md"""
# 2D FNS

This notebook simulates of the fractional neural sampling diffusion model for a target distribution, chosen here to be `Ng=3` Gaussians arranged on a ring.
The tail index `α` controls the strength of Levy jumps, the momentum parameter `β` controls the strength of local oscillations, and `γ` controls the noise strength.

To run this notebook locally, install Julia (https://github.com/JuliaLang/juliaup) and follow the instructions at the top right ("Edit or run this notebook").
"""

# ╔═╡ afc5129d-944a-4da1-94f6-1476af95d150
projectdir = mktempdir() # Set this to your preferred environment folder

# ╔═╡ 9a2ed080-e424-11ef-382f-ad3f0a2f55ad
begin
    using Pkg
    Pkg.activate(projectdir)
    Pkg.add(["Distributions"
             "LinearAlgebra"])
end

# ╔═╡ b9dd9769-1eb0-4805-89c6-0b2679b64846
Pkg.add.(["CairoMakie"
          "Foresight"])

# ╔═╡ c5c9c0be-0b2d-4110-a47d-6db5a8adc0c3
Pkg.add("DifferentialEquations") # Differential equations is a huge package, so very slow to install. Say 5 minutes.

# ╔═╡ 10b7ee5d-cc36-4457-b90b-e90cb14d0bce
Pkg.add(url = "https://github.com/brendanjohnharris/FractionalNeuralSampling.jl",
        rev = "main") # This is an unregistered package

# ╔═╡ 2766e8fc-a834-4778-82a9-93baf5f35d7a
begin # Generate a distribution to sample
    Δx = 3 # Controls spacing between wells
    Ng = 3
    phis = range(0, stop = 2π, length = Ng + 1)[1:Ng]
    centers = Δx .* exp.(im * phis)
    d = MixtureModel([MvNormal([real(c), imag(c)], I(2)) for c in centers])
    D = Density(d)
end

# ╔═╡ 39d087d3-05c2-41b5-be65-ca657df6f8f8
begin # Run simulations
    αs = [2.0, 1.6, 1.2]
    res = map(αs) do α
        L = FractionalNeuralSampler(;
                                    u0 = [Δx 0 0 0],
                                    tspan = 500.0,
                                    α = α, # Tail index
                                    β = 0.1, # Momentum strength
                                    γ = 0.05, # Noise strength
                                    𝜋 = D, # The target distribution
                                    seed = 41)

        sol = solve(L, EM(); dt = 0.001) # Must use EM() (Euler-Maruyama) algorithm
        x, y = eachrow(sol[1:2, :]) # Extract the two position variables
    end
end

# ╔═╡ edeffa19-4eda-4fa8-863d-e59e9e3349ac
begin # Plot
    f = Figure(size = (900, 300))
    gs = subdivide(f, 1, 3)
    map(αs, res, gs) do α, (x, y), g
        xmax = maximum(abs.(extrema(vcat(x, y)))) * 1.1
        xs = range(-xmax, xmax, length = 100)
        ax = Axis(g[1, 1], title = "α = $α", aspect = DataAspect())
        heatmap!(ax, xs, xs, D.(collect.(Iterators.product(xs, xs))),
                 colormap = seethrough(:turbo)) # Plot the target distribution
        lines!(ax, x[1:10:end], y[1:10:end],
               color = (:black, 0.7),
               linewidth = 1) # Plot the trajectories
        hidedecorations!(ax)
    end
    f
end

# ╔═╡ Cell order:
# ╟─393e4f89-5c2b-4cd7-a5e6-e6826b61cd93
# ╠═afc5129d-944a-4da1-94f6-1476af95d150
# ╠═9a2ed080-e424-11ef-382f-ad3f0a2f55ad
# ╠═b9dd9769-1eb0-4805-89c6-0b2679b64846
# ╠═c5c9c0be-0b2d-4110-a47d-6db5a8adc0c3
# ╠═10b7ee5d-cc36-4457-b90b-e90cb14d0bce
# ╠═543309cc-7fdc-4222-9397-bda7393b3d95
# ╠═2766e8fc-a834-4778-82a9-93baf5f35d7a
# ╠═39d087d3-05c2-41b5-be65-ca657df6f8f8
# ╠═edeffa19-4eda-4fa8-863d-e59e9e3349ac
