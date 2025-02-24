using CairoMakie
using Foresight
using DifferentialEquations
using FractionalNeuralSampling
using Distributions
using LinearAlgebra
using Interpolations
using FileIO
using Normalization
using IntervalSets
# using CairoMakie
# using CairoMakie.Colors
using GLMakie
using GLMakie.Colors
import FractionalNeuralSampling: Density
set_theme!(foresight(:physics))

begin
    if false # * MVNormal wells
        Δx = 3 # Controls spacing between wells
        Ng = 3
        phis = range(0, stop = 2π, length = Ng + 1)[1:Ng]
        centers = Δx .* exp.(im * phis)
        d = MixtureModel([MvNormal([real(c), imag(c)], I(2)) for c in centers])
        xmax = 5
        xs = range(-xmax, xmax, length = 100)
        D = Density(d)
    elseif true # Load an image surrogate
        img = load("test/test_image_blur.jpg")
        img = img[1:50:end, 1:50:end]
        img = Colors.Gray.(img)
        img = getproperty.(img, :val)
        img = convert(Matrix{Float64}, img)
        img = permutedims(img, [2, 1]) |> reverse
        img = .-(MinMax(img)(img) .- 1) # SO that blacks become peaks of the normalized density

        mask = FractionalNeuralSampling.Densities.vignette(img, 0.5)
        img = img .* mask # So the edges are 0

        xs = range(-1, 1, length = size(img, 1))
        xmax = maximum(xs)
        @assert -xmax == minimum(xs)
        img = img ./ sum(img) ./ step(xs)^2

        itp = Interpolations.scale(interpolate(img, BSpline(Cubic(Line(OnGrid())))), xs, xs)
        itp = extrapolate(itp, eps())
        D = Density(itp)
        D([0.1, 0.1])
        logdensity(D, [0.1, 0.1])
        logdensity(D, [-0.1, 0.1])
        gradlogdensity(D, [0.1, 0.1])
    end

    f = Figure()#size = (900, 300))
    αs = [2.0, 1.6, 1.2]
    gs = subdivide(f, 1, 3)
    # map(αs, gs) do α, g
    α = 1.6
    g = first(gs)
    begin
        xmax = 1.1
        cxmax = 0.8
        trans = length(sol) ÷ 2

        surf = potential(D)
        subd = 10
        box = ReflectingBox(-xmax .. xmax, -xmax .. xmax)
        L = LevyFlightSampler(;
                              u0 = [0 0 0 0],
                              tspan = 500.0,
                              α = α,
                              β = 0.5,
                              γ = 0.01,
                              𝜋 = D,
                              seed = 42,
                              boundaries = box())
        sol = solve(L, EM(); dt = 0.001)
        xs = range(-xmax, xmax, length = 1000)
        x, y = eachrow(sol[1:2, :])

        zs = surf.(collect.(Iterators.product(xs, xs)))
        z = map(surf ∘ collect, zip(x[trans:subd:end], y[trans:subd:end]))

        ax = Axis3(g[1, 1], title = "α = $α",
                   limits = (nothing, nothing, (minimum(z), maximum(z))),
                   aspect = (5.0, 5.0, 2 / 3))
        # heatmap!(ax, xs, xs, potential(D).(collect.(Iterators.product(xs, xs))),
        # colormap=seethrough(:turbo))
        idxs = xs .∈ [-cxmax .. cxmax] # Idxs of zs within the original domain
        contour3d!(ax, xs, xs, zs, colormap = :turbo, colorrange = extrema(zs[idxs, idxs]),
                   linewidth = 1, levels = range(Interval(extrema(zs[idxs, idxs])...), 20))
        # reverse(seethrough(:turbo, -1))
        subd = 10

        begin
            # Get the subsequences for plotting.
            xseg = x[trans:subd:end]
            yseg = y[trans:subd:end]
            # Use the already-computed z corresponding to these indices.
            # (Assuming the length of z matches xseg/yseg.)
            zseg = z

            # Set a threshold for breaking the line segments.
            thr = 0.75  # adjust as needed
            markersize = 0.03

            # Create 3D points for all indices.
            pts = [Point3f(xi, yi, zi) for (xi, yi, zi) in zip(xseg, yseg, zseg)]

            # Initialize a segment starting at the first point.
            seg = [pts[1]]

            for i in 2:length(pts)
                # Compute displacement from previous point.
                d = norm(pts[i] - pts[i - 1])
                if d > thr
                    # Plot the current segment if it has at least two points.
                    if length(seg) >= 2
                        lines!(ax, seg,
                               color = (:black, 0.5), linewidth = 2)
                        meshscatter!(ax, [seg[1]]; color = cucumber, markersize)  # starting point
                        meshscatter!(ax, [seg[end]]; color = :crimson, markersize)  # ending point
                    else
                        # For an isolated point, mark it as both start and end.
                        meshscatter!(ax, [seg[1]]; color = cucumber, markersize)
                    end
                    # Start a new segment from the current point.
                    seg = [pts[i]]
                else
                    # Continue the current segment.
                    push!(seg, pts[i])
                end
            end

            # Plot the final segment if it exists.
            if length(seg) >= 2
                lines!(ax, seg, color = (:black, 0.5), linewidth = 2)
                meshscatter!(ax, [seg[1]]; color = cucumber, markersize)
                meshscatter!(ax, [seg[end]]; color = :crimson, markersize)
            else
                meshscatter!(ax, [seg[1]]; color = cucumber, markersize)
            end
        end
    end
    # begin
    #     lines!(ax, x[trans:subd:end], y[trans:subd:end], z, color = (:black, 0.5),
    #            linewidth = 2)
    # end
    f
end
