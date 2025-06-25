using DifferentialEquations
using ApproxFun  # for Fourier basis representation and operations
using CairoMakie

# -------------------------
# 1. Define system parameters (with reasonable defaults)
# -------------------------
β = 1.0                  # coupling for dx/dt = β * p (e.g., inverse mass, set to 1.0 by default)
γ = 1.0                  # damping coefficient
σ = 1.0                  # Gaussian kernel width (sigma)
τ_k = 1.0                  # bias kernel time-scale (appears as 1/τ_k factor)
L = 10.0                 # domain size (period length in each dimension)
N = 32                   # number of Fourier modes (truncation order in each dimension)
tspan = (0.0, 10.0)         # time span for simulation (start, end)
dt = 0.01                # integration time step (for fixed-step solver or initial suggestion)

# -------------------------
# 2. Setup Fourier basis for a periodic domain and define bias kernel in that basis
# -------------------------
# Define a periodic domain in [−L/2, L/2] for each spatial dimension (for smooth periodic wrap-around)
domain = PeriodicSegment(-L / 2, L / 2)
fourier_space = Fourier(domain)

# Use ApproxFun to construct a truncated Fourier approximation of a 1D Gaussian on the domain
# We sample the Gaussian at N points and use a Fourier transform to get coefficients.
x_nodes = points(fourier_space, N)                    # N equally spaced points in the domain
gauss_values = [exp(-x^2 / (2 * σ^2)) for x in x_nodes]    # samples of Gaussian exp(-(x^2)/(2σ^2)) at those points
T = ApproxFun.plan_transform(fourier_space, N)        # Fourier transform operator (values -> coeffs) for N modes
gauss_coeffs = T * gauss_values                       # compute Fourier coefficients of the Gaussian
base_gaussian = Fun(fourier_space, gauss_coeffs)      # Fun representing the Gaussian (1D) in Fourier basis with N modes

# Compute the spatial derivative of the Gaussian in the Fourier basis.
# (In Fourier space, differentiation corresponds to multiplying coefficients by i*k.)
Dx = Derivative(fourier_space)                        # first-derivative operator on the Fourier space
base_deriv = Dx * base_gaussian                       # Fun representing d/dx of the Gaussian (i.e., -x/σ^2 * exp(-x^2/(2σ^2)))

# Now `base_gaussian` corresponds to f(x) = exp(-x^2/(2σ^2)) and
# `base_deriv` corresponds to f'(x) = - (x/σ^2) * exp(-x^2/(2σ^2)),
# both represented with N Fourier modes on the periodic domain.

# -------------------------
# 3. Define functions to compute drift (deterministic part) and diffusion (stochastic part)
# -------------------------
# The state vector is u = [x1, x2, p1, p2] (position components followed by momentum components).
# We'll implement the system:
#    ẋ = β * p
#    ṗ = -∇V(x) ∘ p  - γ * p  + (bias force) + noise
# where ∇V(x) = x for V = 1/2||x||^2, and "∘" denotes elementwise multiplication.
# The bias force is -∇K(x,t) ∘ p, with K defined by the Gaussian kernel (in Fourier space).
# The noise term will be handled in the diffusion function (white noise in momentum).
function drift!(du, u, params, t)
    x1, x2, p1, p2 = u
    β, γ, σ, τ_k = params.β, params.γ, params.σ, params.τ_k
    gradV_x1 = x1
    gradV_x2 = x2

    diffx = x1 - x1
    diffy = x2 - x2
    if diffx > L / 2
        diffx -= L
    end
    if diffx < -L / 2
        diffx += L
    end
    if diffy > L / 2
        diffy -= L
    end
    if diffy < -L / 2
        diffy += L
    end
    # Evaluate the 1D Fun objects at these differences
    Kx_val = base_gaussian(diffx)   # Gaussian in x-direction
    Ky_val = base_gaussian(diffy)   # Gaussian in y-direction
    dKx_val = base_deriv(diffx)      # derivative in x-direction
    dKy_val = base_deriv(diffy)      # derivative in y-direction
    # Gradient of K: (∂K/∂x1, ∂K/∂x2)
    gradK_x1 = (1 / τ_k) * dKx_val * Ky_val   # ∂K/∂x1 at (x1,x2)
    gradK_x2 = (1 / τ_k) * Kx_val * dKy_val   # ∂K/∂x2 at (x1,x2)

    # Now assemble the time-derivatives:
    # Position derivatives:
    du[1] = β * p1               # ẋ1 = β * p1
    du[2] = β * p2               # ẋ2 = β * p2
    # Momentum derivatives:
    du[3] = -gradV_x1 * p1 - γ * p1 #!- gradK_x1 * p1   # ṗ1 = - (∂V/∂x1 * p1) - γ p1 - (∂K/∂x1 * p1)
    du[4] = -gradV_x2 * p2 - γ * p2 #!- gradK_x2 * p2   # ṗ2 = similarly for p2

    return
end

# Diffusion function for SDE (noise terms).
function diffusion!(du, u, params, t)
    β, γ = params.β, params.γ
    du .= 0.0
    du[3:4] .= sqrt(2γ)
    return
end

u0 = [0.0 0.0 0.0 0.0]  # combine into a 4-element vector

params = (β = β, γ = γ, σ = σ, τ_k = τ_k)

problem = SDEProblem(drift!, diffusion!, u0, tspan, params)

sol = solve(problem, EM(), dt = dt)

println("Final state at t=$(tspan[2]): ", sol(tspan[2]))

lines(sol.t, sol[4, :])
