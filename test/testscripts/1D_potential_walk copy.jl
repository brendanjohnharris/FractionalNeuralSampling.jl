using CairoMakie
using DifferentialEquations

# Define the full complex system with feedback on μ.
# The state vector is u = [x, y, μ] with z = x + i*y.
function hopf_complex!(du, u, p, t)
    # Unpack state variables.
    x, y, mu = u
    # Compute r² = x² + y² and r = √(r²)
    r2 = x^2 + y^2
    r = sqrt(r2)
    r4 = r2^2

    # Unpack parameters:
    # c_re, c_im, d_re, d_im: complex coefficients
    # ω: frequency
    # a: rate constant for the μ dynamics,
    # r_target: target amplitude for μ dynamics.
    c_re, c_im, d_re, d_im, ω, a, r_target = p

    # Compute the oscillator using the full complex form:
    # z˙ = (mu + iω)z + c|z|²z + d|z|⁴z.
    # Split into real and imaginary parts:
    #
    # Real part:
    #   du[1] = mu*x - ω*y + c_re*r²*x - c_im*r²*y + d_re*r⁴*x - d_im*r⁴*y.
    # Imaginary part:
    #   du[2] = ω*x + mu*y + c_re*r²*y + c_im*r²*x + d_re*r⁴*y + d_im*r⁴*x.
    du[1] = mu * x - ω * y + c_re * r2 * x - c_im * r2 * y + d_re * r4 * x - d_im * r4 * y
    du[2] = ω * x + mu * y + c_re * r2 * y + c_im * r2 * x + d_re * r4 * y + d_im * r4 * x

    # Control parameter evolution for μ using a linear competition:
    # If r is below r_target, then (r_target - r) > 0 and μ increases;
    # if r is above r_target, then (r_target - r) < 0 and μ decreases.
    du[3] = a * (r_target - r)
end

# Set parameters.
c_re = 1.0      # Real part of c (for subcritical Hopf, c_re > 0)
c_im = 0.0      # Imaginary part of c.
d_re = -1.0     # Real part of d (for subcritical Hopf, d_re < 0)
d_im = 0.0      # Imaginary part of d.
ω = 0.5       # Angular frequency.
a = 0.01     # Rate constant for the μ dynamics.
r_target = 0.3      # Target amplitude (zero crossing of μ dynamics).

# Bundle parameters into a tuple.
p = (c_re, c_im, d_re, d_im, ω, a, r_target)

# Initial conditions.
# We choose a small initial amplitude (set x = r0, y = 0) and an initial μ.
r0 = 0.1
x0 = r0
y0 = 0.0
mu0 = 0.0        # Starting control parameter.
u0 = [x0, y0, mu0]

# Time span for the simulation.
tspan = (0.0, 1000.0)

# Define and solve the ODE problem.
prob = ODEProblem(hopf_complex!, u0, tspan, p)
sol = solve(prob, Tsit5(), abstol = 1e-8, reltol = 1e-8)

# --- Plotting ---

# Function to extract amplitude r = √(x² + y²) from the solution.
function extract_r(sol)
    return [sqrt(u[1]^2 + u[2]^2) for u in sol.u]
end

r_vals = extract_r(sol)
x_vals = getindex.(sol.u, 1)  # x-coordinate vs. time.
y_vals = getindex.(sol.u, 2)  # y-coordinate vs. time.

# Create four panels:
# 1. Amplitude r(t)
# 2. Control parameter μ(t)
# 3. Phase portrait: r versus μ
# 4. x–coordinate vs. time.
fig = Figure(resolution = (800, 1200))

# Panel 1: Amplitude Dynamics r(t)
ax1 = Axis(fig[1, 1], xlabel = "Time", ylabel = "Amplitude r",
           title = "Amplitude Dynamics")
lines!(ax1, sol.t, r_vals, color = :blue, label = "r(t)")
axislegend(ax1, position = :rt)

# Panel 2: Control Parameter Dynamics μ(t)
ax2 = Axis(fig[2, 1], xlabel = "Time", ylabel = "Control Parameter μ",
           title = "Control Parameter Dynamics")
lines!(ax2, sol.t, getindex.(sol.u, 3), color = :red, label = "μ(t)")
axislegend(ax2, position = :rt)

# Panel 3: Phase Portrait (r vs. μ)
ax3 = Axis(fig[3, 1], xlabel = "Amplitude r", ylabel = "Control Parameter μ",
           title = "Phase Portrait (r vs. μ)")
lines!(ax3, r_vals, getindex.(sol.u, 3), color = :green, label = "Trajectory")
axislegend(ax3, position = :rt)

# Panel 4: x–coordinate Dynamics x(t)
ax4 = Axis(fig[4, 1], xlabel = "Time", ylabel = "x",
           title = "x-Coordinate Dynamics")
lines!(ax4, sol.t, x_vals, color = :magenta, label = "x(t)")
axislegend(ax4, position = :rt)

save("hopf_oscillator.png", fig)
fig |> display

# Create a 3D figure showing time vs x(t) vs y(t)
fig3D = Figure(resolution = (600, 600))
ax3D = Axis3(fig3D[1, 1],
              xlabel = "Time t",
              ylabel = "x(t)",
              zlabel = "y(t)",
              title = "Hopf Oscillator in 3D (t, x, y)")
lines!(ax3D, t_vals, x_vals, y_vals, color = :blue)

save("hopf_oscillator_3d.png", fig3D)
fig3D



begin
###########################################################
#  BIFURCATION DIAGRAM for the quintic Hopf normal form   #
#  (sub‑critical:  c_re > 0 ,  d_re < 0)                  #
###########################################################
using CairoMakie

# ---- model coefficients -----------------------------------------------------
c_re = 1.01      # cubic term  (use 1.0 if you kept the original values)
d_re = -1.0      # quintic term

# ---- parameter range for μ --------------------------------------------------
μ_min, μ_max = -0.3, 0.06          # wide enough to see all branches
μs = range(μ_min, μ_max; length = 800)

# ---- containers for the different branches ---------------------------------
μ_eq0_stable    = Float64[] ; μ_eq0_unstable = Float64[]
r_eq0_stable    = Float64[] ; r_eq0_unstable = Float64[]

μ_lc_stable     = Float64[] ; μ_lc_unstable  = Float64[]
r_lc_stable     = Float64[] ; r_lc_unstable  = Float64[]

# ---- scan μ and classify the roots -----------------------------------------
for μ in μs
    # --- the trivial fixed point r = 0 ---------------------------------------
    if μ < 0
        push!(μ_eq0_stable,   μ) ; push!(r_eq0_stable,   0.0)   # stable (Re λ < 0)
    else
        push!(μ_eq0_unstable, μ) ; push!(r_eq0_unstable, 0.0)   # unstable
    end

    # --- positive roots of d s² + c s + μ = 0 (with s = r²) ------------------
    Δ = c_re^2 - 4d_re*μ           # discriminant
    Δ < 0 && continue              # no real roots ⇒ no limit cycles

    # two candidate roots in s = r²
    for s in ((-c_re + sqrt(Δ))/(2d_re), (-c_re - sqrt(Δ))/(2d_re))
        s > 1e-12 || continue      # discard negative / tiny roots
        r = sqrt(s)

        # linear stability of the limit cycle:
        #    d(ṙ)/dr at r*>0  →  sign = 2s(c + 2ds)
        stab = 2s*(c_re + 2d_re*s) < 0

        if stab
            push!(μ_lc_stable, μ) ; push!(r_lc_stable, r)
        else
            push!(μ_lc_unstable, μ) ; push!(r_lc_unstable, r)
        end
    end
end

# ---- plotting ---------------------------------------------------------------
fig = Figure(resolution = (640, 420))
ax  = Axis(fig[1, 1],
           xlabel = "control parameter μ",
           ylabel = "radius r",
           title  = "Bifurcation diagram (sub‑critical Hopf)")

# r = 0 branch
lines!(ax, μ_eq0_stable,   r_eq0_stable,
       color = :black, linewidth = 2,  label = "r = 0 (stable)")
lines!(ax, μ_eq0_unstable, r_eq0_unstable,
       color = :black, linewidth = 2, linestyle = :dash,
       label = "r = 0 (unstable)")

# limit‑cycle branches
scatter!(ax, μ_lc_stable,   r_lc_stable,
         color = :dodgerblue, markersize = 4, label = "limit cycle (stable)")
scatter!(ax, μ_lc_unstable, r_lc_unstable,
         color = :crimson,   markersize = 4, label = "limit cycle (unstable)")

axislegend(ax, position = :rt)
fig |> display                     # show in a live session
save("bifurcation_diagram.png", fig)
end
