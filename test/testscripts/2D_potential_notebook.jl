### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 543309cc-7fdc-4222-9397-bda7393b3d95
begin # Imports. May have to wait for the cells above, then rerun.
    using CairoMakie
    using Foresight
    using DifferentialEquations
    using FractionalNeuralSampling
    using Distributions
    using LinearAlgebra
	using PlutoUI
	
    import FractionalNeuralSampling: Density, divide_dims
	import SpecialFunctions: gamma
	import RecursiveArrayTools: ArrayPartition
	
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
    Pkg.add(["Distributions",
             "LinearAlgebra",
			 "SpecialFunctions",
			 "RecursiveArrayTools",
			 "PlutoUI"])
end

# ╔═╡ b9dd9769-1eb0-4805-89c6-0b2679b64846
Pkg.add.(["CairoMakie",
          "Foresight"])

# ╔═╡ c5c9c0be-0b2d-4110-a47d-6db5a8adc0c3
Pkg.add("DifferentialEquations") # Differential equations is a huge package, so very slow to install. Say 5 minutes.

# ╔═╡ 10b7ee5d-cc36-4457-b90b-e90cb14d0bce
Pkg.add(url = "https://github.com/brendanjohnharris/FractionalNeuralSampling.jl",
        rev = "fully_fractional") # This is an unregistered package

# ╔═╡ 2aec5c65-5e24-4327-8038-ceb70ff29d8d
md"""
## Fractional Neural Sampler

$$dx_t = -\gamma c_\alpha \nabla V(x_t)  dt  + \beta  p_t dt + \gamma^{1/\alpha}dL^\alpha_t$$

$$dp_t = -\beta c_\alpha \nabla V(x_t) dt$$


where:
- ``V(x) = -\ln[\pi(x)]`` is the potential associated with a target distribution ``\pi(x)``;
- ``dL^\alpha_t`` is the increment of a Levy process with tail index ``\alpha``;
- ``\gamma`` is the noise strength;
- ``\beta`` is the momentum parameter; and
- ``c_\alpha = \Gamma(\alpha - 1)/\Gamma(\alpha / 2)^2`` is the correction factor for the local approximation to the fractional spatial derivative.
"""


# ╔═╡ 61a0a490-a7e2-44c5-8a8d-7ae54d5e2412
md"""
### Static potential
"""

# ╔═╡ 2766e8fc-a834-4778-82a9-93baf5f35d7a
begin # Generate a distribution to sample
    Δx = 3 # Controls spacing between wells
    Ng = 3
    ϕs = range(0, stop = 2π, length = Ng + 1)[1:Ng]
    centers = Δx .* exp.(im * ϕs)
    d = MixtureModel([MvNormal([real(c), imag(c)], I(2)) for c in centers])
    D = Density(d)
end

# ╔═╡ 39d087d3-05c2-41b5-be65-ca657df6f8f8
begin # Run simulations
    αs = [2.0, 1.6, 1.2]
	x0 = [Δx, 0.0] 
	p0 = [0.0, 0.0] # Be careful with types; use 0.0 not 0
    res = map(αs) do α
        L = FractionalNeuralSampler(;
                                    u0 = ArrayPartition(x0, p0),
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
		Xs = collect.(Iterators.product(xs, xs))
		
        ax = Axis(g[1, 1], title = "α = $α", aspect = DataAspect())
        heatmap!(ax, xs, xs, D.(Xs),
                 colormap = seethrough(:turbo)) # Plot the target distribution
        lines!(ax, x[1:10:end], y[1:10:end],
               color = (:black, 0.7),
               linewidth = 1) # Plot the trajectories
        hidedecorations!(ax)
    end
    f
end

# ╔═╡ 8b1a7d48-426a-4c76-b63a-61693a457281
md"""
## Time-varying potential

To introduce a time-varying potential, we need to turn ``\pi`` into a function of time.
"""

# ╔═╡ 3a24c88c-fca5-4643-85c0-2190c4a13b5d
function afns_f!(du, u, p, t)
	    (α, β, γ), 𝜋 = p
	    x, v = divide_dims(u, length(u) ÷ 2)

		# Here we have replaced 𝜋 -> 𝜋(t)
	    b = gradlogdensity(𝜋(t))(x) * gamma(α - 1) / (gamma(α / 2) .^ 2)
	
	    dx, dv = divide_dims(du, length(du) ÷ 2)
	    dx .= γ .* b .+ β .* v
	    dv .= β .* b
	end

# ╔═╡ 131da237-ca04-4793-b954-12e3c56c47d9
function afns_g!(du, u, p, t) # Same as original equations
	    (α, β, γ), 𝜋 = p
	    dx, dv = divide_dims(du, length(du) ÷ 2)
	    dx .= γ^(1 / α) # ? × dL in the integrator.
	    dv .= 0.0
	end

# ╔═╡ b159f245-f421-4323-8958-c0df43f5b994
function aFractionalNeuralSampler(;
	                                 tspan, α, β, γ, u0, 𝜋,
	                                 boundaries = nothing,
	                                 noise_rate_prototype = similar(u0),
	                                 noise = nothing,
	                                 kwargs...)
		if isnothing(noise)
			noise = NoiseProcesses.LevyProcess!(α; ND = 2, W0 = zero(u0))
		end
	    Sampler(afns_f!, afns_g!; callback = boundaries, kwargs..., u0,
	            noise_rate_prototype, noise,
	            tspan, p = ((α, β, γ), 𝜋))
	end

# ╔═╡ 38762d7b-6c2c-4f32-8187-39299902bd73
begin
	τ_rise = 500
	τ_rest = 500
	function f1(t::Real)
		total_period = 2*(τ_rise + τ_rest)
	    t_eff = mod(t, total_period)
	    t_phase1_end = τ_rise
	    t_phase2_end = τ_rise + τ_rest
	    t_phase3_end = 2 * τ_rise + τ_rest
	
	    if t_eff < t_phase1_end
	        return τ_rise == 0 ? 1.0 : t_eff / τ_rise
	    elseif t_eff < t_phase2_end
	        return 1.0
	    elseif t_eff < t_phase3_end
	        return τ_rise == 0 ? 0.0 : 1.0 - ((t_eff - t_phase2_end) / τ_rise)
	    else
	        return 0.0
	    end
	end
	f2(t::Real) =  1 .- f1(t)
end

# ╔═╡ 26fb7322-af5e-4e9c-8edd-4371f2c61611
begin # Plot adaptation parameter
	t = 0:5000
	h = Figure()
	axx = Axis(h[1, 1], xlabel="time", ylabel="f")
	lines!(axx, t, f1; label="Well 1 weight")
	lines!(axx, t, f2; label="Well 2 weight")
	axislegend(axx)
	h
end

# ╔═╡ ad673030-1b26-492d-aab2-0ca1bf983b0d
begin
	wells = [MvNormal([-3.0, 3.0], I(2)), MvNormal([3.0, -3.0], I(2))]
	G(t) = Density(MixtureModel(wells, [f1(t), f2(t)]))
end

# ╔═╡ a0ea8f09-296b-4782-8d4c-dd5ca738e2af
begin # * Run time-varying simulation
L = aFractionalNeuralSampler(;
							u0 = ArrayPartition(x0, p0),
							tspan = 5000.0,
							α = 1.4, # Tail index
							β = 0.1, # Momentum strength
							γ = 0.05, # Noise strength
							𝜋 = G, # The target distribution
							seed = 41)
end

# ╔═╡ cd9f5ac8-cdbf-45b7-af67-1ba33c7df82d
begin
	sol = solve(L, EM(); dt = 0.001) # Takes about 5 seconds
	x, y = eachrow(sol[1:2, :])
end

# ╔═╡ 6db08281-8842-4eba-bf94-808454fa05c6
begin
 	fig = Figure()

	xmax = 7
	xs = range(-xmax, xmax, length = 200)
	Xs = collect.(Iterators.product(xs, xs))
	X = Observable(G(0).(Xs)) # For recording
	xy = Observable([Point2f([NaN, NaN])]) 
	xy_last = Observable(Point2f([NaN, NaN]))
	color = (:black, 0.5)
	
	ax = Axis(fig[1, 1], aspect = DataAspect())
	heatmap!(ax, xs, xs, X, colormap = seethrough(:turbo), colorrange=(0, 0.2))
	lines!(ax, xy; linewidth = 1, color)
	scatter!(ax, xy_last; markersize=10, color=:red)
	hidedecorations!(ax)
	hidespines!(ax)
end

# ╔═╡ b60add8e-22c4-42c3-a666-7753b0dac569
begin
	xy[] = [Point2f([NaN, NaN])]
	file = record(fig, "tfns.mp4", range(1, length(sol), step=1000);
	        framerate = 48) do i
		t = sol.t[i]
	    X[] = G(t).(Xs)
		push!(xy[], Point2f(sol[1:2, i]))
		length(xy[]) > 100 && popfirst!(xy[])
		xy_last[] = last(xy[])
	end
end

# ╔═╡ 0d7ca9f8-ac50-40cc-bce7-a258aab6a7f8
PlutoUI.LocalResource(file)

# ╔═╡ Cell order:
# ╟─393e4f89-5c2b-4cd7-a5e6-e6826b61cd93
# ╠═afc5129d-944a-4da1-94f6-1476af95d150
# ╠═9a2ed080-e424-11ef-382f-ad3f0a2f55ad
# ╠═b9dd9769-1eb0-4805-89c6-0b2679b64846
# ╠═c5c9c0be-0b2d-4110-a47d-6db5a8adc0c3
# ╠═10b7ee5d-cc36-4457-b90b-e90cb14d0bce
# ╠═543309cc-7fdc-4222-9397-bda7393b3d95
# ╟─2aec5c65-5e24-4327-8038-ceb70ff29d8d
# ╟─61a0a490-a7e2-44c5-8a8d-7ae54d5e2412
# ╠═2766e8fc-a834-4778-82a9-93baf5f35d7a
# ╠═39d087d3-05c2-41b5-be65-ca657df6f8f8
# ╠═edeffa19-4eda-4fa8-863d-e59e9e3349ac
# ╟─8b1a7d48-426a-4c76-b63a-61693a457281
# ╠═3a24c88c-fca5-4643-85c0-2190c4a13b5d
# ╠═131da237-ca04-4793-b954-12e3c56c47d9
# ╠═b159f245-f421-4323-8958-c0df43f5b994
# ╠═38762d7b-6c2c-4f32-8187-39299902bd73
# ╠═26fb7322-af5e-4e9c-8edd-4371f2c61611
# ╠═ad673030-1b26-492d-aab2-0ca1bf983b0d
# ╠═a0ea8f09-296b-4782-8d4c-dd5ca738e2af
# ╠═cd9f5ac8-cdbf-45b7-af67-1ba33c7df82d
# ╠═6db08281-8842-4eba-bf94-808454fa05c6
# ╠═b60add8e-22c4-42c3-a666-7753b0dac569
# ╠═0d7ca9f8-ac50-40cc-bce7-a258aab6a7f8
