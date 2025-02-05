import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------
# Simulation parameters
# ---------------------------
dt = 0.01  # time step
T = 40  # total simulation time (shorter for demonstration)
nsteps = int(T / dt)
sample_interval = 5  # store every 5th step for animation

# Model parameters
V0 = 50.0  # depth of each potential well
d = 5.0  # separation between the two wells
sigma = 1.0  # width of each Gaussian well
D = 0.005  # base noise strength
lambda_adapt = 50.0  # coupling constant for adaptation (increases noise amplitude)
tau = 10.0  # memory time constant for adaptation
L = 1.0  # characteristic length scale: if |x(t)-x(s)| << L, the integrand is near 1


# ---------------------------
# Define static potential V(x) and its derivative dV/dx
# ---------------------------
def V(x):
    return -V0 * (
        np.exp(-((x - d / 2) ** 2) / (2 * sigma**2))
        + np.exp(-((x + d / 2) ** 2) / (2 * sigma**2))
    )


def dVdx(x):
    term1 = (x - d / 2) * np.exp(-((x - d / 2) ** 2) / (2 * sigma**2))
    term2 = (x + d / 2) * np.exp(-((x + d / 2) ** 2) / (2 * sigma**2))
    return (V0 / sigma**2) * (term1 + term2)


# ---------------------------
# Define a heuristic effective potential for visualization.
# Here we define:
#   V_eff(x,t) = V(x) * [D/(D + lambda_adapt*A(t))]
# so that as A(t) increases the effective well depth is reduced.
# ---------------------------
def V_eff(x, A):
    return V(x) * (D / (D + lambda_adapt * A))


# ---------------------------
# Allocate arrays for time, particle position, and adaptation variable
# ---------------------------
t_arr = np.linspace(0, T, nsteps)
x_arr = np.zeros(nsteps)
A_arr = np.zeros(nsteps)

# Initial condition: start near 0 (but not exactly at the unstable equilibrium)
x_arr[0] = 0.1
A_arr[0] = 0.0

# For animation, store snapshots every sample_interval steps
t_anim = []
x_anim = []
A_anim = []

# ---------------------------
# Simulation loop (Euler–Maruyama + integrodifferential adaptation)
# ---------------------------
# We'll compute A[n] using a summation over past history.
for n in range(nsteps - 1):
    # --- Compute adaptation variable A[n] ---
    # We approximate: A[n] = sum_{j=0}^{n} w_{n-j} * exp(-((x[n]-x[j])^2/(2L^2))
    indices = np.arange(n + 1)
    weights = (dt / tau) * np.exp(-((n - indices) * dt) / tau)
    integrand = np.exp(-((x_arr[n] - x_arr[: n + 1]) ** 2) / (2 * L**2))
    A_arr[n] = np.sum(weights * integrand)

    # --- Update particle position ---
    # Noise amplitude is sqrt(2*(D + lambda_adapt * A[n])*dt)
    noise = np.sqrt(2 * (D + lambda_adapt * A_arr[n]) * dt) * np.random.randn()
    x_arr[n + 1] = x_arr[n] - dVdx(x_arr[n]) * dt + noise

    # --- Store data for animation every sample_interval steps ---
    if n % sample_interval == 0:
        t_anim.append(t_arr[n])
        x_anim.append(x_arr[n])
        A_anim.append(A_arr[n])

# Also store the final step.
t_anim.append(t_arr[-1])
x_anim.append(x_arr[-1])
A_anim.append(A_arr[-1])
n_frames = len(t_anim)

# ---------------------------
# Set up the figure with two subplots (stacked vertically)
# Upper panel: potential curves and particle marker.
# Lower panel: time series of x(t) with vertical time indicator.
# ---------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

# Upper panel: Plot static potential and effective potential.
x_plot = np.linspace(-15, 15, 500)
V_static = V(x_plot)
ax1.plot(x_plot, V_static, "r-", lw=2, label="Static potential V(x)")
# Plot initial effective potential using A = A_anim[0]
V_eff_initial = V_eff(x_plot, A_anim[0])
(eff_line,) = ax1.plot(
    x_plot, V_eff_initial, "k--", lw=2, label="Effective potential V_eff(x,t)"
)
# Plot particle marker (for the initial frame, position (x, V(x)) is shown)
init_x = x_anim[0]
# For visualization we plot the particle at (x, V(x)), though the dynamics are unchanged.
(particle_marker,) = ax1.plot([init_x], [V(init_x)], "bo", markersize=8, zorder=5)
ax1.set_xlim(-15, 15)
ax1.set_ylim(np.min(V_static) - 5, np.max(V_static) + 5)
ax1.set_xlabel("x")
ax1.set_ylabel("V")
ax1.set_title("Static (red) and Effective (dashed) Potentials; Particle Position")
ax1.legend()

# Lower panel: Plot the time series x(t) and a vertical dashed line showing current time.
ax2.plot(t_arr, x_arr, "b-", lw=2, label="x(t)")
ax2.set_xlim(t_arr[0], t_arr[-1])
ax2.set_xlabel("Time")
ax2.set_ylabel("x")
ax2.set_title("Time Series of x(t)")
time_line = ax2.axvline(t_anim[0], color="k", linestyle="--", lw=2)


# ---------------------------
# Animation update function
# ---------------------------
def update(frame):
    # Retrieve current adaptation A and particle position.
    current_A = A_anim[frame]
    current_x = x_anim[frame]
    # Update the effective potential dashed line.
    V_eff_current = V_eff(x_plot, current_A)
    eff_line.set_ydata(V_eff_current)
    # Update the particle marker (plotted at (x, V(x)) for clarity)
    particle_marker.set_data([current_x], [V(current_x)])
    # Update the vertical time line in the time-series plot.
    current_time = t_anim[frame]
    time_line.set_xdata([current_time])
    return eff_line, particle_marker, time_line


# Create the animation.
anim = FuncAnimation(fig, update, frames=n_frames, interval=20, blit=True)

plt.tight_layout()
plt.show()
