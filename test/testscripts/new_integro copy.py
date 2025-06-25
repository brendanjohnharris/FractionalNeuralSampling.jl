#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Simulation parameters
dt = 0.01  # time step
T = 100.0  # total simulation time
N = int(T / dt)  # number of time steps
t_array = np.linspace(0, T, N + 1)

# Model parameters
gamma = 0.00005  # noise strength
tau_k = 2.5  # adaptation (kernel) time scale (driving amplitude)
tau_d = 5  # decay time constant for the adaptation field
sigma = 0.5  # spatial scale of the Gaussian kernel

# -----------------------------
# Potential parameters for a ring of three wells
R = 1.0  # radius of the ring
k_r = 10.0  # stiffness for the radial part
A = 0.5  # amplitude of the angular modulation


def V(xy):
    """
    Potential function in 2d.

    In polar coordinates (r, θ), we define:
      V(r,θ) = (k_r/2)*(r - R)^2 - A*cos(3θ),
    so that the particle is attracted to r=R and there are three angular wells.
    """
    x, y = xy[0], xy[1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return 0.5 * k_r * (r - R) ** 2 - A * np.cos(3 * theta)


def gradV(xy):
    """
    Gradient of V(x,y) computed in Cartesian coordinates.

    For the radial part:
       V_r = 0.5 * k_r * (r - R)^2,
       ∇V_r = k_r*(r-R)*(x/r, y/r).

    For the angular part:
       V_a = -A*cos(3θ)   with   dθ/dx = -y/(r^2), dθ/dy = x/(r^2),
       ∇V_a = (3A*sin(3θ)*(y/r^2), -3A*sin(3θ)*(x/r^2)).
    """
    x, y = xy[0], xy[1]
    r = np.sqrt(x**2 + y**2)
    if r < 1e-8:
        r = 1e-8
    theta = np.arctan2(y, x)
    grad_rad = k_r * (r - R) * np.array([x / r, y / r])
    grad_ang = 3 * A * np.sin(3 * theta) * np.array([-y / r**2, x / r**2])
    return grad_rad + grad_ang


# -----------------------------
# Simulation: Euler–Maruyama integration in 2d

# x_traj will store the trajectory as an array of shape (N+1, 2)
x_traj = np.zeros((N + 1, 2))
# Set the initial condition near one of the wells (e.g., near (R, 0)).
x_traj[0] = np.array([R, 0.0])

for n in range(N):
    if n == 0:
        mem_term = np.array([0.0, 0.0])
    else:
        # Compute differences between current state and all previous states.
        diff = x_traj[n] - x_traj[: n + 1]  # shape: (n+1, 2)
        diff2 = np.sum(diff**2, axis=1)
        # Spatial Gaussian kernel weight.
        spatial_kernel = np.exp(-diff2 / (2 * sigma**2))
        # Exponential time decay.
        time_decay = np.exp(-(t_array[n] - t_array[: n + 1]) / tau_d)
        # Discrete approximation of the memory integral.
        mem_integral = dt * np.sum(
            diff * spatial_kernel[:, None] * time_decay[:, None], axis=0
        )
        mem_term = mem_integral / (tau_k * sigma**2)
    drift = -gradV(x_traj[n]) + mem_term
    noise = np.sqrt(2 * gamma * dt) * np.random.randn(2)
    x_traj[n + 1] = x_traj[n] + drift * dt + noise

# Compute the speed at each time step (approximate via finite differences)
speed = np.zeros(N + 1)
speed[0] = 0.0
speed[1:] = np.linalg.norm(x_traj[1:] - x_traj[:-1], axis=1) / dt

# -----------------------------
# Prepare for animation: two subplots (upper: 2d trajectory, lower: speed vs. time)

# Create grid for contour of the potential.
x_vals = np.linspace(-2, 2, 200)
y_vals = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)
for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        xy = np.array([X[j, i], Y[j, i]])
        Z[j, i] = V(xy)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 12), gridspec_kw={"height_ratios": [2, 1]}
)
fig.tight_layout(pad=3.0)

# Top subplot: 2d potential contour and trajectory.
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("2D Trajectory on a Ring of Three Wells")
# Plot the contour without labels.
contours = ax1.contour(X, Y, Z, levels=20, cmap="viridis")
# Initialize trajectory line and moving particle.
(traj_line,) = ax1.plot([], [], "b-", lw=1, label="Trajectory")
(ball,) = ax1.plot([], [], "ro", markersize=6, label="Particle")
ax1.legend()

# Bottom subplot: speed vs. time.
ax2.set_xlim(0, T)
ax2.set_ylim(0, np.max(speed) * 1.1)
ax2.set_xlabel("Time")
ax2.set_ylabel("Speed")
ax2.set_title("Particle Speed vs. Time")
(speed_line,) = ax2.plot([], [], "m-", lw=2, label="Speed")
ax2.legend()


def update(frame):
    n = frame
    # Update the 2d trajectory.
    traj_line.set_data(x_traj[: n + 1, 0], x_traj[: n + 1, 1])
    ball.set_data([x_traj[n, 0]], [x_traj[n, 1]])
    ax1.set_title(f"2D Trajectory, Time = {t_array[n]:.2f}")
    # Update the speed plot.
    speed_line.set_data(t_array[: n + 1], speed[: n + 1])
    ax2.set_xlim(0, t_array[n] if t_array[n] > 10 else 10)
    return traj_line, ball, speed_line


# Create animation: update every 100th time step for speed.
anim = FuncAnimation(fig, update, frames=range(0, N + 1, 100), interval=50, blit=True)

plt.show()
