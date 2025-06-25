# tempered_fractional_langevin_double_well.py
"""
Simulate and animate the overdamped tempered fractional Langevin equation
in a symmetric double-well potential V(x)=a x**4 − b x**2.

The stochastic driver is tempered fractional Gaussian noise (TFGn) with
Hurst exponent H and tempering parameter lambda.  A simple
spectral-factorisation method is used to generate a discrete TFGn
sequence.  For pedagogical clarity we ignore inertial terms and memory
integrals in the force; extensions to under-damped GLE form are noted in
the accompanying write-up.

Dependencies
------------
* numpy
* matplotlib (for the live animation and plotting)
* scipy (for power spectral density and statistical fitting)

Run
---
$ python tempered_fractional_langevin_double_well.py

The script will open four matplotlib windows:
1. Animated particle in the double-well potential
2. Static time-series plot of the trajectory
3. Power spectral density (PSD) of the increments Δx with 1/f^β fit
4. Histogram of dwell times with exponential fit

Uncomment the ``ani.save``, ``fig2.savefig``, ``fig3.savefig``, or ``fig4.savefig``
lines to export MP4 or PNG respectively (requires ffmpeg for the MP4).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import welch
from scipy.stats import expon

# ===================== USER-TUNABLE PARAMETERS ===================== #
N = 100000  # number of time steps
DT = 0.001  # time step (s)
H = 0.2  # Hurst exponent (0.5 gives classical white noise)
LAMBDA = 0.1  # tempering rate (1/s) – cut-off for long-range memory
SIGMA = 200.0  # noise amplitude (sqrt(DT) scaling already built in)
FIT_MIN_FREQ = 0.1  # minimum frequency (Hz) for 1/f fit

# Double-well potential V(x)=a x^4 − b x^2
A = 1.0
B = 1.0

# Starting position
X0 = -1.0

# -------------------- Parameter validation ---------------------- #
# Enforce valid Hurst exponent range for stationary fractional increments
if not (0 < H < 1):
    raise ValueError(
        f"Invalid Hurst exponent H={H}: must satisfy 0 < H < 1 for tempered fractional Gaussian noise."
    )


# =============== UTILITY: tempered fractional noise =============== #
def tempered_fGn(N, dt, H, lam):
    """Return a length-N array of tempered fractional Gaussian noise."""
    alpha = H + 0.5
    nfft = 2 * N
    freqs = np.fft.fftfreq(nfft, d=dt)
    omega = 2 * np.pi * freqs
    S = (lam**2 + omega**2) ** (-alpha)
    S[0] = 0.0
    g = np.random.normal(size=nfft) + 1j * np.random.normal(size=nfft)
    g[nfft // 2 + 1 :] = np.conj(g[1 : nfft // 2][::-1])
    y = np.fft.ifft(g * np.sqrt(S))
    return np.real(y[:N]) / np.sqrt(dt)


# -------------------- deterministic force ------------------------- #
def dVdx(x, a=A, b=B):
    return 4 * a * x**3 - 2 * b * x


# ========================= MAIN PROGRAM =========================== #
# Generate trajectory
t = np.arange(N) * DT
xi = SIGMA * tempered_fGn(N, DT, H, LAMBDA)
x = np.empty(N)
x[0] = X0
for n in range(1, N):
    x[n] = x[n - 1] - dVdx(x[n - 1]) * DT + xi[n - 1] * DT

# ======================= ANIMATION PLOT ========================== #
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.set_xlim(-2.2, 2.2)
# ax.set_ylim(-1.2, 2.5)
# ax.set_xlabel("x")
# ax.set_ylabel("V(x)")
# xs = np.linspace(-2.2, 2.2, 400)
# ax.plot(xs, A * xs**4 - B * xs**2, "k", lw=2)
# (particle,) = ax.plot([], [], "ro", ms=5)
# ax.set_title("Tempered Fractional Langevin Dynamics in a Double Well")


def init():
    particle.set_data([], [])
    return (particle,)


def update(frame):
    xf = x[frame]
    particle.set_data([xf], [A * xf**4 - B * xf**2])
    return (particle,)


# ani = FuncAnimation(
#     fig, update, frames=range(0, N, 10), init_func=init, interval=20, blit=True
# )
# plt.tight_layout()
# plt.show()
# # ani.save("tempered_fractional_langevin.mp4", dpi=150, fps=60)

# ==================== STATIC TIME SERIES PLOT ===================== #
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(t, x, lw=1)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Position x(t)")
ax2.set_title("Tempered Fractional Langevin Trajectory vs Time")
plt.tight_layout()
plt.show()
# fig2.savefig("trajectory_time_series.png", dpi=150)

# ============== POWER SPECTRAL DENSITY OF INCREMENTS ============= #
# Compute increments Δx and their PSD using Welch's method
dx = np.diff(x)
fs = 1.0 / DT
f_inc, Pxx_inc = welch(dx, fs=fs, nperseg=4096)

# Fit 1/f^β in log-log for frequencies above FIT_MIN_FREQ
mask = f_inc >= FIT_MIN_FREQ
logf = np.log10(f_inc[mask])
logP = np.log10(Pxx_inc[mask])
m, c = np.polyfit(logf, logP, 1)
beta = -m  # PSD ~ 1/f^β => logP = -β logf + const

# Plot PSD and fit line only over the fit range
gf = f_inc[mask]
Pfit = 10**c * gf**m

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.loglog(f_inc, Pxx_inc, label="PSD of Δx")
ax3.loglog(gf, Pfit, "r--", label=f"1/f^{beta:.2f} fit")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("PSD of Δx(t)")
ax3.set_title("Power Spectral Density of Increments Δx with 1/f^β Fit")
ax3.legend()
# Annotate exponent on plot in upper right
ax3.text(
    0.95,
    0.95,
    f"β = {beta:.2f}",
    transform=ax3.transAxes,
    ha="right",
    va="top",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"),
)
plt.tight_layout()
plt.show()

# ==================== DWELL TIME DISTRIBUTION ==================== #======= #
well_idx = np.sign(x)
crossings = np.where(well_idx[:-1] != well_idx[1:])[0] + 1
dwell_times = np.diff(crossings) * DT
# params = expon.fit(dwell_times, floc=0)
# scale = params[1]
# lambda_est = 1.0 / scale

fig4, ax4 = plt.subplots(figsize=(6, 4))
bins = np.linspace(0, dwell_times.max(), 50)
ax4.hist(dwell_times, bins=bins, density=True, alpha=0.6, label="Empirical")
t_vals = np.linspace(0, dwell_times.max(), 200)
# ax4.plot(
#     t_vals,
#     expon.pdf(t_vals, loc=0, scale=scale),
#     "r--",
#     lw=2,
#     label=f"Exponential fit (λ={lambda_est:.3f} s⁻¹)",
# )
ax4.set_xlabel("Dwell time (s)")
ax4.set_ylabel("Probability density")
ax4.set_title("Distribution of Dwell Times Between Wells")
ax4.legend()
plt.tight_layout()
plt.show()
# fig4.savefig("dwell_time_distribution.png", dpi=150)

if __name__ == "__main__":
    pass
