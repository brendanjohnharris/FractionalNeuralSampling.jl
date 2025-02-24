using DifferentialEquations, GLMakie, StatsBase

# =======================================
# 1. Define the triple–well potential and its derivative
# =======================================
# We choose:
#   V(x) = a*x^6 - 2a*d^2*x^4 + a*d^4*x^2 - V0,
# with a = 1/d^6.
# This potential has minima at x = -d, 0, and d with V = -V0.
function V(x, V0, d)
    a = 1.0 / d^6
    return a * x^6 - 2 * a * d^2 * x^4 + a * d^4 * x^2 - V0
end

function dVdx(x, V0, d)
    a = 1.0 / d^6
    return 6 * a * x^5 - 8 * a * d^2 * x^3 + 2 * a * d^4 * x
end

# =======================================
# 2. Define the augmented overdamped SDE system using an exponential kernel
# =======================================
# The state vector is now u = [x, m] with:
#
#   (i)   ẋ = -V'(x) - α m + √(2D)*ξ(t),
#   (ii)  ṁ = -λ m + x,   where λ = 1/τ.
#
# Here m(t) approximates the convolution with the exponential kernel.
function f!(du, u, p, t)
    # Unpack parameters.
    V0, d, α, D, τ = p
    x = u[1]
    m = u[2]

    # Overdamped Langevin eq. for x(t):
    du[1] = -dVdx(x, V0, d) - α * m
    # Memory variable: ṁ = - (1/τ)*m + x.
    du[2] = -(1 / τ) * m + x
end

# The noise is applied only in the x-equation.
function g!(du, u, p, t)
    V0, d, α, D, τ = p
    du[1, 1] = sqrt(2 * D)  # noise in x
    du[2, 1] = 0.0          # no noise in the memory variable m
end

# =======================================
# 3. Set parameters and initial conditions
# =======================================
# Potential parameters:
V0 = 4.0         # Depth scale; wells at x = -d, 0, and x = d have V = -V0.
d = 5.0         # Location of the wells.

# Dynamics parameters:
α = 5.0         # Coupling strength for the memory acting on the position.
D = 0.1        # Noise intensity.
τ = 50.0        # Characteristic time of the exponential kernel (adjust as desired).

# Since we use a single memory mode, we set:
#   N = 1, λ = 1/τ, and w = 1 are now absorbed into the equations.
# Initial conditions:
x0 = 0.0              # Starting at x = 0 (for example).
m0 = 0.0              # Memory variable initially zero.
u0 = [x0, m0]         # State vector: [x, m].

# Pack parameters into a tuple.
p = (V0, d, α, D, τ)
tspan = (0.0, 1000.0)   # Total simulation time.

# =======================================
# 4. Set up and solve the SDE problem
# =======================================
# Use a small time step (dt = 0.001) to resolve the dynamics.
prob = SDEProblem(f!, g!, u0, tspan, p)
sol = solve(prob, EM(), dt = 0.001)

# =======================================
# 5. Plotting and Animation Setup using GLMakie
# =======================================
# Pre-calculate the static triple–well potential on a grid.
x_grid = range(-15, 15, length = 500)
V_static = [V(x, V0, d) for x in x_grid]

# For animation, sample the solution at regular intervals.
sample_interval = 10  # Adjust as desired.
indices = 1:sample_interval:length(sol.t)
if indices[end] != length(sol.t)
    indices = vcat(indices, length(sol.t))
end
n_frames = length(indices)
t_anim = sol.t[indices]
x_anim = sol[1, indices]  # Extract x(t).

# Create a figure with two subplots:
#   - Upper panel: the static triple–well potential with the current particle position.
#   - Lower panel: the full time series of x(t).
fig = Figure(resolution = (800, 600))
ax1 = Axis(fig[1, 1],
           xlabel = "x", ylabel = "V",
           title = "Triple–well Potential & Particle Position",
           limits = ((-10, 10), (-4.1, -3.5)))
ax2 = Axis(fig[2, 1],
           xlabel = "Time", ylabel = "x",
           title = "Time Series of x(t)")

# Upper panel: plot the static potential.
lines!(ax1, x_grid, V_static, color = :red, linewidth = 2, label = "V(x)")
# Create an Observable for the particle’s current position.
particle_pos_obs = Observable(Point2f(x_anim[1], V(x_anim[1], V0, d)))
scatter!(ax1, particle_pos_obs, color = :blue, markersize = 10)
axislegend(ax1)

# Lower panel: plot the full time series.
lines!(ax2, sol.t, sol[1, :], color = :blue, linewidth = 2, label = "x(t)")
time_min = minimum(sol[1, :]) - 1
time_max = maximum(sol[1, :]) + 1
time_line_obs = Observable(([t_anim[1], t_anim[1]], [time_min, time_max]))
tx = @lift $time_line_obs[1]
ty = @lift $time_line_obs[2]
lines!(ax2, tx, ty, color = :black, linestyle = :dash, linewidth = 2)

# =======================================
# 6. (Optional) Record an Animation
# =======================================
# To record an animation to a video file, uncomment the block below.
record(fig, "overdamped_langevin_memory_triplewell_exponential.mp4", 1:1000:n_frames;
       framerate = 30) do i
    current_x = x_anim[i]
    particle_pos_obs[] = Point2f(current_x, V(current_x, V0, d))
    current_t = t_anim[i]
    tx[] = [current_t, current_t]
end

println("Overdamped Langevin simulation with an exponential memory kernel complete.")
fig
