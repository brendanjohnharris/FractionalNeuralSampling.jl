#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Simulation parameters
dt = 0.005  # time step
T = 100.0  # total simulation time
N = int(T / dt)  # number of time steps
t_array = np.linspace(0, T, N + 1)

# Model parameters
gamma = 0.02  # noise strength
tau_k = 4  # adaptation (kernel) time scale (driving amplitude)
tau_d = 1  # decay time constant for the adaptation field
sigma = 0.2  # spatial scale of the Gaussian kernel


# -----------------------------
# Triple-well potential and its derivative
def V(x):
    """
    Triple well potential:
       V(x)= x^6 - 2 x^4 + x^2.
    This potential has three minima at x = -1, 0, and 1.
    """
    return x**6 - 2 * x**4 + x**2


def potential_force(x):
    """
    Force from the triple well potential:
       V'(x)= 6x^5 - 8x^3 + 2x.
    """
    return 6 * x**5 - 8 * x**3 + 2 * x


# -----------------------------
# Simulation: Euler–Maruyama integration of the SDE
# SDE:
#   dx/dt = -V'(x) + sqrt(2*gamma)*noise + (1/(tau_k*sigma^2)) *
#           ∫_0^t dt' exp[-(t-t')/tau_d] (x(t)-x(t')) exp[-(x(t)-x(t'))^2/(2*sigma^2)]
x_traj = np.zeros(N + 1)
x_traj[0] = 0.0  # initial condition

for n in range(N):
    if n == 0:
        memory_term = 0.0
    else:
        differences = x_traj[n] - x_traj[: n + 1]
        spatial_kernel = np.exp(-(differences**2) / (2 * sigma**2))
        time_decay = np.exp(-(t_array[n] - t_array[: n + 1]) / tau_d)
        memory_integral = dt * np.sum(differences * spatial_kernel * time_decay)
        memory_term = memory_integral / (tau_k * sigma**2)
    drift = -potential_force(x_traj[n]) + memory_term
    noise = np.sqrt(2 * gamma * dt) * np.random.randn()
    x_traj[n + 1] = x_traj[n] + drift * dt + noise

# -----------------------------
# Prepare for animation with two subplots
# Top panel: effective potential and ball
# Bottom panel: time series of x(t)

# Define spatial grid for effective potential
x_grid = np.linspace(-2, 2, 400)
static_pot = V(x_grid)  # static triple-well potential

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [2, 1]}
)
fig.tight_layout(pad=3.0)

# Top panel settings
ax1.set_xlim(-2, 2)
ax1.set_ylim(-0.1, 0.4)
ax1.set_xlabel("x")
ax1.set_ylabel("Potential")
ax1.set_title("Effective Potential & Particle Position")
(static_line,) = ax1.plot(x_grid, static_pot, "k--", label="Static potential V(x)")
(eff_line,) = ax1.plot([], [], "orange", lw=2, label="Effective potential")
(ball,) = ax1.plot([], [], "ro", markersize=8)
ax1.legend()

# Bottom panel settings: Time series of x(t)
ax2.set_xlim(0, T)
ax2.set_ylim(np.min(x_traj) - 0.5, np.max(x_traj) + 0.5)
ax2.set_xlabel("Time")
ax2.set_ylabel("x(t)")
ax2.set_title("Trajectory of x(t)")
(time_line,) = ax2.plot([], [], "b-", lw=1)


def compute_effective_potential(x_vals, x_traj, n, dt, tau_k, tau_d, sigma):
    """
    Compute V_eff(x,t)= V(x) + K(x,t),
    with K(x,t) ≈ (dt/tau_k)*sum_{i=0}^{n} exp[-(t[n]-t[i])/tau_d]*exp[-(x - x_traj[i])^2/(2*sigma^2)]
    """
    decay = np.exp(-(t_array[n] - t_array[: n + 1]) / tau_d)
    K_eff = (dt / tau_k) * np.sum(
        np.exp(-((x_vals[:, None] - x_traj[: n + 1]) ** 2) / (2 * sigma**2)) * decay,
        axis=1,
    )
    return V(x_vals) + K_eff


def effective_potential_at_ball(x_val, x_traj, n, dt, tau_k, tau_d, sigma):
    """
    Compute the effective potential at x_val:
    V_eff(x,t)= V(x) + (dt/tau_k)*sum_{i=0}^{n} exp[-(t[n]-t[i])/tau_d]*exp[-(x - x_traj[i])^2/(2*sigma^2)]
    """
    decay = np.exp(-(t_array[n] - t_array[: n + 1]) / tau_d)
    K_eff = (dt / tau_k) * np.sum(
        np.exp(-((x_val - x_traj[: n + 1]) ** 2) / (2 * sigma**2)) * decay
    )
    return V(x_val) + K_eff


def update(frame):
    n = frame
    # Update effective potential (top panel)
    V_eff = compute_effective_potential(x_grid, x_traj, n, dt, tau_k, tau_d, sigma)
    eff_line.set_data(x_grid, V_eff)
    x_ball = x_traj[n]
    y_ball = effective_potential_at_ball(x_ball, x_traj, n, dt, tau_k, tau_d, sigma)
    ball.set_data([x_ball], [y_ball])
    ax1.set_title(f"Effective Potential & Position, Time = {t_array[n]:.2f}")

    # Update time series (bottom panel)
    time_line.set_data(t_array[: n + 1], x_traj[: n + 1])
    ax2.set_xlim(0, t_array[n] if t_array[n] > 10 else 10)  # extend x-axis gradually
    ax2.set_title("Trajectory of x(t)")

    return eff_line, ball, time_line


# Create animation: update every 10th time step for speed
anim = FuncAnimation(fig, update, frames=range(0, N + 1, 10), interval=50, blit=True)

plt.show()
