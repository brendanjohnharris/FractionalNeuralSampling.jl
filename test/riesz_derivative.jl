using Symbolics
using SymbolicNumericIntegration
using CairoMakie
@variables x y k

G(; μ, σ) = exp(-(x - μ)^2 / (2 * σ^2)) / sqrt(2 * pi * σ^2)

N = G(; μ = 1, σ = 1)

ℱ(f) = integrate(f, (x, -Inf, Inf); symbolic = true)
ℱ(N)

I = integrate(N, (x, -Inf, Inf); symbolic = true)

I = integrate(exp(-(x)^2))

using FFTW
using ForwardDiff: derivative

begin
    g(x) = exp((-((x - 8))^2) / 2) / sqrt(2 * pi)
    g1(x) = derivative(g, x)
    g2(x) = derivative(g1, x)
    xs = 0:0.001:16
    ys = g.(xs)
    y1s = g1.(xs)
    y2s = g2.(xs)
    α = 0.5

    D(ys; α) = irfft(-(abs.(rfftfreq(ys |> length)) .^ α) .* rfft(ys), length(ys))
    ŷs = D(ys; α)
    # plot(xs, y1s ./ maximum(y1s))
    # plot(xs, y2s ./ real.(y2s)[xs .== 8])
    plot(xs, real.(ŷs) ./ real.(ŷs)[xs .== 8] .+ 0.2)

    yys = fracdiff(g, α, 16, step(xs), RieszSymmetric())
    plot!(xs[2:(end - 1)], yys[2:(end - 1)] ./ real.(yys)[xs .== 8] .+ 0.3)
    current_figure() |> display
end
