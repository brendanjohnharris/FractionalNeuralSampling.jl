using DifferentialEquations, CairoMakie, Interpolations

# -------------------------
# Parameters and grid setup
# -------------------------
V0 = 1.0       # amplitude (sets the depth of the wells to -V0)
d_well = 5.0   # wells are located at x = ±d_well
sigma = 1.0    # (unused in the quartic potential, kept for compatibility)
alpha = 1.0    # strength of the adaptation contribution to the effective potential
γ = 4.0        # adaptation writing strength
τ = 100.0      # adaptation decay time constant
ε = 0.3        # width of the kernel for adaptation
D = 0.5        # diffusion constant

# Grid for the adaptation field A(x,t)
x_min = -15.0
x_max = 15.0
N = 301
grid = range(x_min, x_max, length = N) |> collect

# Initial conditions:
# Start with the particle at one well and zero adaptation everywhere.
x0 = -d_well
A0 = zeros(N)
u0 = [x0; A0]

# -------------------------
# Model Functions
# -------------------------

# Define the quartic double-well potential V(x)
# This potential has minima at x = ±d_well with V = -V0 and a maximum at x = 0 with V = 0.
function V(x, V0, d_well, sigma)
    return V0 * (((x^2 - d_well^2)^2) / (d_well^4) - 1)
end

# Its derivative dV/dx
function dVdx(x, V0, d_well, sigma)
    return V0 * (4 * x * (x^2 - d_well^2)) / (d_well^4)
end

# The adaptation kernel: a narrow Gaussian.
function kernel(z, ε)
    return exp(-z^2 / (2 * ε^2)) / (sqrt(2π) * ε)
end

# -------------------------
# Drift and Diffusion Functions
# -------------------------
# The state vector is u = [ x, A₁, A₂, …, A_N ]
# Only the particle x gets noise; the A fields are deterministic.

function f!(du, u, p, t)
    # Unpack state
    x = u[1]
    Avec = u[2:end]

    # Unpack parameters
    V0, d_well, sigma, α, γ, τ, ε, grid, D = p

    # Build an interpolation of the adaptation field.
    # We use a linear interpolation (with linear extrapolation outside the grid)
    A_itp = LinearInterpolation(grid, Avec, extrapolation_bc = Line())

    # Estimate dA/dx at the current particle position.
    δ = 1e-3
    dA_dx = (A_itp(x + δ) - A_itp(x - δ)) / (2δ)

    # SDE for x:
    #   dx/dt = -d/dx[V(x) + α A(x,t)] + sqrt(2D)*ξ(t)
    du[1] = -dVdx(x, V0, d_well, sigma) - α * dA_dx

    # ODE for each grid point of A(x,t):
    #   ∂_t A(x_i,t) = -A(x_i,t)/τ + (γ/τ) * kernel(x_i - x(t), ε)
    for (i, xi) in enumerate(grid)
        du[i + 1] = -Avec[i] / τ + (γ / τ) * kernel(xi - x, ε)
    end
end

# The diffusion only acts on x.
function g!(du, u, p, t)
    # u[1] is x, the rest are A fields.
    V0, d_well, sigma, α, γ, τ, ε, grid, D = p
    du[1, 1] = sqrt(2 * D)
    for i in 2:length(u)
        du[i, 1] = 0.0
    end
end

# -------------------------
# Set up and solve the SDE
# -------------------------
p = (V0, d_well, sigma, alpha, γ, τ, ε, grid, D)
tspan = (0.0, 200.0)
prob = SDEProblem(f!, g!, u0, tspan, p)
sol = solve(prob, EM(), dt = 0.01)

# -------------------------
# Plotting the Results
# -------------------------
# (1) Plot the trajectory x(t) over a heatmap of the static potential.
begin
    heatmap(sol.t[1:100:end], grid,
            repeat(V.(grid, V0, d_well, sigma), 1, length(sol.t[1:100:end]))',
            colormap = :viridis)
    lines!(sol.t, sol[1, :], label = "x(t)", color = :black)
    current_figure()
end

# Choose a sampling interval for animation frames:
sample_interval = 50
indices = 1:sample_interval:length(sol.t)
if indices[end] != length(sol.t)
    indices = vcat(indices, length(sol.t))
end
n_frames = length(indices)
t_anim = sol.t[indices]
x_anim = sol[1, indices]

# Precompute the static potential on the grid:
V_static = [V(x, V0, d_well, sigma) for x in grid]

# Create a figure with two vertically arranged axes.
fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "V",
           title = "Static (red) and Effective (dashed) Potentials; Particle Position")
ax2 = Axis(fig[2, 1], xlabel = "Time", ylabel = "x", title = "Time Series of x(t)")

# Upper panel: plot static potential V(x) in red.
lines!(ax1, grid, V_static, color = :red, linewidth = 2, label = "Static V(x)")

# Create an Observable for the effective potential.
# At time t = t_anim[1] the adaptation field is:
A_first = sol.u[indices[1]][2:end]
V_eff_first = [V(x, V0, d_well, sigma) + alpha * A_first[i] for (i, x) in enumerate(grid)]
V_eff_obs = Observable(V_eff_first)
# Plot effective potential (dashed black).
lines!(ax1, grid, V_eff_obs, color = :black, linestyle = :dash, linewidth = 2,
       label = "Effective V_eff(x,t)")

# Create an Observable for the particle's (display) position.
init_x = x_anim[1]
particle_pos_obs = Observable(Point2f(init_x, V(init_x, V0, d_well, sigma)))
# Plot the particle as a blue marker.
scatter!(ax1, particle_pos_obs, color = :blue, markersize = 8)

axislegend(ax1)

# Lower panel: plot the full time series x(t) in blue.
lines!(ax2, sol.t, sol[1, :], color = :blue, linewidth = 2, label = "x(t)")
# For the vertical time indicator, choose reasonable y-limits:
time_min = minimum(sol[1, :]) - 1
time_max = maximum(sol[1, :]) + 1
time_line_obs = Observable(([t_anim[1], t_anim[1]], [time_min, time_max]))
tlo_x = @lift $time_line_obs[1]
tlo_y = @lift $time_line_obs[2]
lines!(ax2, tlo_x, tlo_y, color = :black, linestyle = :dash, linewidth = 2)

record(fig, "animation.mp4", 1:n_frames; framerate = 30) do i
    # Update the effective potential observable.
    A_frame = sol.u[indices[i]][2:end]  # adaptation field at the current frame
    V_eff_frame = [V(x, V0, d_well, sigma) + alpha * A_frame[j]
                   for (j, x) in enumerate(grid)]
    V_eff_obs[] = V_eff_frame

    # Update the particle marker observable.
    current_x = x_anim[i]
    current_y = V(current_x, V0, d_well, sigma)
    particle_pos_obs[] = Point2f(current_x, current_y)

    # Update the vertical time indicator in the lower panel.
    current_t = t_anim[i]
    time_line_obs[] = ([current_t, current_t], [time_min, time_max])
end
