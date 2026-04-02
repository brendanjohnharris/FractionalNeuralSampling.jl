using TestEnv;
TestEnv.activate();
using CairoMakie
using Foresight
using FractionalNeuralSampling
using Distributions
using LinearAlgebra
using Interpolations
using FileIO
using Normalization
using IntervalSets
using CairoMakie
using CairoMakie.Colors
using RecursiveArrayTools
import FractionalNeuralSampling: Density
set_theme!(foresight(:physics))

begin
    Δx = 3 # Controls spacing between wells
    Ng = 3
    phis = range(0, stop = 2π, length = Ng + 1)[1:Ng]
    centers = Δx .* exp.(im * phis)
    prior = [0.05, 0.5, 5.0]
    width = [1, 0.5, 0.1]
    prior ./= sum(prior)
    d = MixtureModel([MvNormal([real(c), imag(c)], I(2) * s)
                      for (c, s) in zip(centers, width)],
                     prior)
    xmax = 5
    xs = range(-xmax, xmax, length = 100)
    D = Density(d)

    # heatmap(xs, xs, potential(D).(collect.(Iterators.product(xs, xs))),
    #         colormap = :turbo)
    # end
    # begin
    f = Figure()#size = (900, 300))
    αs = [2.0, 1.6, 1.2]
    gs = subdivide(f, 1, 3)
    # map(αs, gs) do α, g
    α = 1.5
    g = first(gs)
    begin
        xmax = 7
        cxmax = 5

        surf = potential(D)
        subd = 10
        box = ReflectingBox(-xmax .. xmax, -xmax .. xmax)
        @info "Starting simulation"
        L = FNS(;
                u0 = ArrayPartition([0.0, 0.0], [0.0, 0.0]),
                tspan = 100.0,
                α = α,
                β = 4,
                γ = 0.8,
                𝜋 = D,
                seed = 42,
                boundaries = box())
        sol = solve(L, EM(); dt = 0.001)
        trans = length(sol) ÷ 2
        xs = range(-xmax, xmax, length = 1000)
        x, y = eachrow(sol[1:2, :])

        zs = surf.(collect.(Iterators.product(xs, xs)))
        z = map(surf ∘ collect, zip(x[trans:subd:end], y[trans:subd:end]))

        ax = Axis3(g[1, 1], title = "α = $α",
                   #    limits = (nothing, nothing, (minimum(z) - 1, maximum(z))),
                   aspect = (5.0, 5.0, 2 / 3), elevation = π / 6)
        hidedecorations!(ax)
        hidespines!(ax)
        # heatmap!(ax, xs, xs, potential(D).(collect.(Iterators.product(xs, xs))),
        #          colormap = :turbo)

        idxs = xs .∈ [-cxmax .. cxmax] # Idxs of zs within the original domain
        p = surface!(ax, xs, xs, zs, colormap = :turbo,
                     colorrange = extrema(zs[idxs, idxs]), rasterize=true)
        contour3d!(ax, xs, xs, zs, colormap = :turbo, colorrange = extrema(zs[idxs, idxs]),
                   linewidth = 1, levels = range(Interval(extrema(zs[idxs, idxs])...),
                                                 20))

        # reverse(seethrough(:turbo, -1))
        subd = 10

        if false
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

        # * Add colorbar
        Colorbar(g[1, 2], p, label = "Potential", height = Relative(0.3))
    end
    begin
        lines!(ax, x[trans:subd:end], y[trans:subd:end], z, color = :crimson,
               linewidth = 1)
    end
    save("test/review_schematic.pdf", f)
    f
end
