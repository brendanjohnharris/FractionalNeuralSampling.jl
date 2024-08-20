using FractionalCalculus
using CairoMakie

x = -2.25:0.01:2.25
# * Construct bimodal potential
f(x) = -4x^2 + x^4
lines(x, f.(x))
d = fracdiff(f, 1.5, 1.5, 0.01, RieszSymmetric())
plot(d)
