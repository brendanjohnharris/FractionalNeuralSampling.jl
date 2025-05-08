using CairoMakie
using Foresight
using LinearAlgebra
using ApproxFun
set_theme!(foresight(:physics))

begin # * Construct a potential
    V(xy) = xy[1]^2 + xy[2]^2
end
begin # * Construct a space
    boundaries = [-2 .. 2, -2 .. 2]
    sp = prod(Fourier.(boundaries)) # Fourier ensures periodicity of the solution
    v = Fun(V, sp) # Automatic basis expansion
    Δ = ApproxFun.Laplacian(sp)
end
begin # * construct the derivative
    α = 2.0
    ℒ = (-(Δ))^2 # Fractional Laplacian
end

begin # * Plot the potential and the derivative
    f = Figure(size = (800, 300))
    ax = Axis(f[1, 1], title = "Potential")
    xs = range(-2, 2, length = 100)
    lines!(ax, xs, v.(xs), label = "V(x)")

    ax = Axis(f[1, 2], title = "Derivative")
    lines!(ax, xs, dV.(xs), label = "dV(x)/dx")
    f
end
