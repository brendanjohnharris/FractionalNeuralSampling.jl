using DifferentialEquations, CairoMakie

# =======================================
# 1. Define the double‐well potential
# =======================================
# We use a quartic double-well potential with minima at x = ±d.
# Here V0 sets the well depth (V(±d) = -V0) and the barrier at x=0 is 0.
function V(x, V0, d)
    return V0 * ((x^2 - d^2)^2 / d^4 - 1)
end

function dVdx(x, V0, d)
    return (4 * V0 * x * (x^2 - d^2)) / d^4
end

# =======================================
# 2. Define the augmented system of SDEs
# =======================================
# Let u[1] = x(t) and u[2] = m(t) where
#   m(t) = (1/τ)∫₀ᵗ e^{-(t-s)/τ} x(s) ds,
# so that
#   ẋ = -V'(x) + α (x - m) + √(2D) ξ(t)
#   ḿ = (x - m)/τ
#
# The noise acts only on x.
function f!(du, u, p, t)
    # Unpack state: x = u[1], m = u[2]
    x, m = u
    V0, d, α, τ, D = p
    du[1] = -dVdx(x, V0, d) + α * (x - m)
    du[2] = (x - m) / τ
end

function g!(du, u, p, t)
    # Only the x component has noise.
    V0, d, α, τ, D = p
    du[1, 1] = sqrt(2 * D)
    du[2, 1] = 0.0
end

# =======================================
# 3. Set parameters and initial conditions
# =======================================
# Potential and dynamics parameters
V0 = 4.0       # amplitude; wells at x = ±d have V = -V0, barrier at 0 is 0.
d = 5.0       # location of the wells at x = ±d
α = 0.4      # strength of the repelling (adaptation) force
τ = 50.0     # memory time constant
D = 0.6       # noise intensity

# Initial conditions:
# Start with the particle in the left well and the memory set equal to the initial x.
x0 = -d
m0 = x0
u0 = [x0, m0]

# Parameter tuple passed to the functions
p = (V0, d, α, τ, D)
tspan = (0.0, 2000.0)

# =======================================
# 4. Set up and solve the SDE problem
# =======================================
prob = SDEProblem(f!, g!, u0, tspan, p)
sol = solve(prob, EM(), dt = 0.05)

# =======================================
# 5. Animation Setup using CairoMakie
# =======================================
# We will animate the following:
#   - Upper panel: the static double-well potential V(x) (red curve) along with a blue marker at (x, V(x))
#     representing the current particle position.
#   - Lower panel: the full time-series x(t) (blue curve) with a vertical dashed line indicating current time.
#
# Pre-calculate the static potential on a grid.
x_grid = range(-15, 15, length = 500)
V_static = [V(x, V0, d) for x in x_grid]

# For animation, sample the solution at regular intervals.
sample_interval = 50
indices = 1:sample_interval:length(sol.t)
if indices[end] != length(sol.t)
    indices = vcat(indices, length(sol.t))
end
n_frames = length(indices)
t_anim = sol.t[indices]
x_anim = sol[1, indices]

# Create a figure with two vertically arranged axes.
fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "V",
           title = "Static Potential and Particle Position", limits = ((-10, 10), (-5, 5)))
ax2 = Axis(fig[2, 1], xlabel = "Time", ylabel = "x",
           title = "Time Series of x(t)")

# Upper panel: Plot the static potential (red curve)
lines!(ax1, x_grid, V_static, color = :red, linewidth = 2, label = "V(x)")

# Create an Observable for the particle's position marker.
# (We display the particle at (x, V(x)) for clarity.)
init_x = x_anim[1]
particle_pos_obs = Observable(Point2f(init_x, V(init_x, V0, d)))
scatter!(ax1, particle_pos_obs, color = :blue, markersize = 10)

axislegend(ax1)

# Lower panel: Plot the full time series x(t)
lines!(ax2, sol.t, sol[1, :], color = :blue, linewidth = 2, label = "x(t)")
# Set y-limits a bit beyond the range of x(t)
time_min = minimum(sol[1, :]) - 1
time_max = maximum(sol[1, :]) + 1
# Create an Observable for a vertical line indicating current time.
time_line_obs = Observable(([t_anim[1], t_anim[1]], [time_min, time_max]))
tx = @lift $time_line_obs[1]
ty = @lift $time_line_obs[2]
lines!(ax2, tx, ty,
       color = :black, linestyle = :dash, linewidth = 2)

# =======================================
# 6. Record the Animation
# =======================================
record(fig, "animation.mp4", 1:n_frames; framerate = 30) do i
    # Update the particle marker on the upper panel.
    current_x = x_anim[i]
    current_y = V(current_x, V0, d)
    particle_pos_obs[] = Point2f(current_x, current_y)

    # Update the vertical time indicator on the lower panel.
    current_t = t_anim[i]
    time_line_obs[] = ([current_t, current_t], [time_min, time_max])
end

println("Animation saved as animation.mp4")
