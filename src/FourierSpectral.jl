using FFTW
using FastTransforms # For evaluation via nufft
using LinearAlgebra
using SciMLOperators

kernel = x -> exp(-(norm(x)^2) / (2 * 1.0^2))
boundaries = PeriodicBox(-3 .. 3, -3 .. 3)

begin
    n_modes = 40

    x0 = boundaries.min_corner
    domains = Boundaries.domain(boundaries)
    Ls = domains .|> IntervalSets.width
    M = 2 * n_modes + 1  # grid points per dimension

    # # Create 1D grids (dropping the duplicate endpoint)
    # xs = map(domains) do x
    #     range(x; length = M + 1)[1:(end - 1)]
    # end

    grid = map(collect, Boundaries.grid(boundaries, M + 1))[1:(end - 1), 1:(end - 1)]

    z = map(kernel, grid)

    ℱ = Matrix{Complex{eltype(z)}}(undef, size(z))
    ℱ = FFTW.plan_fft!(ℱ, 1:ndims(ℱ))

    a = (ℱ * z) / (M^2) # Fourier representation of the kernel

    dx = Ls / M # 'Sampling frequency'
    freqs = fftfreq.(M, 1 / dx)
    freqs = map(collect, Iterators.product(freqs...))

    Ds = map(eachindex(freqs |> first)) do n
        fs = map(Base.Fix2(getindex, n), freqs)
        im .* view(fs, :) |> SciMLOperators.DiagonalOperator
    end

    # Evaluate at a given point
    realspace(a, x) = sum(a .* exp.(im * 2π * map(Base.Fix2(dot, x), freqs)))
    ẑ = realspace.([a], grid)

    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1]; xlabel = "Position x", ylabel = "Position y")
    heatmap!(ax, map(real, ẑ)) # * Is wrong
    ax = Axis(fig[1, 2]; xlabel = "Position x", ylabel = "Position y")
    heatmap!(ax, map(real, z))
    fig
end
