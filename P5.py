import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve

# ============================================================
# 1) Physical constants and problem parameters
# ============================================================
hbar = 1.0
m = 1.0
L = 1.0
V0 = 0.5
omega = 4.0 * np.pi**2   # resonant with 1 -> 3

# ============================================================
# 2) Spatial grid
# ============================================================
N_x = 1000
x = np.linspace(0.0, L, N_x)
dx = x[1] - x[0]

x_int = x[1:-1]
N_int = len(x_int)

# ============================================================
# 3) Time grid
# ============================================================
T_drive = 2.0 * np.pi / omega
t_max = 20 * T_drive
dt = 1.0e-4
t = np.arange(0.0, t_max + dt, dt)
N_t = len(t)

# ============================================================
# 4) Initial wavefunction: ground state
# ============================================================
psi_int = np.sqrt(2.0 / L) * np.sin(np.pi * x_int / L)
psi_int = psi_int.astype(complex)

# Unperturbed eigenstates for population tracking
phi1 = np.sqrt(2.0 / L) * np.sin(1.0 * np.pi * x_int / L).astype(complex)
phi3 = np.sqrt(2.0 / L) * np.sin(3.0 * np.pi * x_int / L).astype(complex)

# ============================================================
# 5) Kinetic-energy operator
# ============================================================
coeff = hbar**2 / (2.0 * m * dx**2)

T = coeff * diags(
    diagonals=[
        -np.ones(N_int - 1),
         2.0 * np.ones(N_int),
        -np.ones(N_int - 1)
    ],
    offsets=[-1, 0, 1],
    format="csc",
    dtype=complex
)

I = identity(N_int, format="csc", dtype=complex)

# ============================================================
# 6) Helper functions
# ============================================================
def normalize(psi_interior):
    norm = np.sqrt(np.sum(np.abs(psi_interior)**2) * dx)
    return psi_interior / norm

def full_wavefunction(psi_interior):
    psi_full = np.zeros(N_x, dtype=complex)
    psi_full[1:-1] = psi_interior
    return psi_full

def store_observables(step, psi_interior,
                      prob_density_history, x_expectation,
                      P1_history, P3_history):
    psi_full = full_wavefunction(psi_interior)
    prob_density = np.abs(psi_full)**2

    prob_density_history[step] = prob_density
    x_expectation[step] = np.sum(x * prob_density) * dx

    c1 = np.sum(np.conj(phi1) * psi_interior) * dx
    c3 = np.sum(np.conj(phi3) * psi_interior) * dx

    P1_history[step] = np.abs(c1)**2
    P3_history[step] = np.abs(c3)**2

# ============================================================
# 7) Storage arrays
# ============================================================
prob_density_history = np.zeros((N_t, N_x), dtype=float)
x_expectation = np.zeros(N_t, dtype=float)
P1_history = np.zeros(N_t, dtype=float)
P3_history = np.zeros(N_t, dtype=float)

psi_int = normalize(psi_int)
store_observables(0, psi_int,
                  prob_density_history, x_expectation,
                  P1_history, P3_history)

# ============================================================
# 8) Time evolution using Crank-Nicolson
# ============================================================
for n in range(N_t - 1):
    t_mid = 0.5 * (t[n] + t[n + 1])

    V_mid = V0 * np.sin(np.pi * x_int / L) * np.sin(omega * t_mid)
    V_op = diags(V_mid, 0, format="csc", dtype=complex)

    H_mid = T + V_op

    A = I + 1j * dt * H_mid / (2.0 * hbar)
    B = I - 1j * dt * H_mid / (2.0 * hbar)

    rhs = B @ psi_int
    psi_int = spsolve(A, rhs)

    psi_int = normalize(psi_int)

    store_observables(n + 1, psi_int,
                      prob_density_history, x_expectation,
                      P1_history, P3_history)

# ============================================================
# 9) Plot probability density snapshots
# ============================================================
time_indices = np.linspace(0, N_t - 1, 5, dtype=int)

plt.figure(figsize=(10, 6))
for idx in time_indices:
    plt.plot(x, prob_density_history[idx], label=fr"$t = {t[idx]:.3f}$")

plt.title(r"Probability Density $|\psi(x,t)|^2$ at Different Times")
plt.xlabel("Position x")
plt.ylabel(r"$|\psi(x,t)|^2$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 10) Plot expectation value <x>
# ============================================================
plt.figure(figsize=(8, 6))
plt.plot(t, x_expectation)
plt.title(r"Expectation Value of Position $\langle x \rangle$ Over Time")
plt.xlabel("Time t")
plt.ylabel(r"$\langle x \rangle$")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 11) Plot populations P1(t) and P3(t)
# ============================================================
plt.figure(figsize=(8, 6))
plt.plot(t, P1_history, label=r"$P_1(t)$")
plt.plot(t, P3_history, label=r"$P_3(t)$")
plt.title(r"State Populations $P_1(t)$ and $P_3(t)$")
plt.xlabel("Time t")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 12) Optional heatmap
# ============================================================
plt.figure(figsize=(10, 6))
plt.imshow(
    prob_density_history,
    aspect="auto",
    origin="lower",
    extent=[x[0], x[-1], t[0], t[-1]]
)
plt.colorbar(label=r"$|\psi(x,t)|^2$")
plt.xlabel("Position x")
plt.ylabel("Time t")
plt.title(r"Time Evolution of Probability Density $|\psi(x,t)|^2$")
plt.tight_layout()
plt.show()
