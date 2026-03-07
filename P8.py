import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ============================================================
# Parameters
# ============================================================
N = 8          # Number of spins
J = 1.0        # Heisenberg coupling
S = 0.5        # Spin-1/2
dim = 2**N

# ============================================================
# Generate all basis states in the Sz product basis
# Each state is represented by +/-1, corresponding to Sz = +/-1/2
# ============================================================
basis_states = np.array(list(product([1, -1], repeat=N)), dtype=int)

# Map each basis state to its index for fast lookup
state_to_index = {tuple(state): idx for idx, state in enumerate(basis_states)}

# Total magnetization Mz for each basis state
# Since basis uses +/-1, multiply by S = 1/2
Mz_basis = S * np.sum(basis_states, axis=1)

# ============================================================
# Construct the Heisenberg Hamiltonian
#
# H = J sum_i [ S_i^z S_{i+1}^z + 1/2 (S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+) ]
#
# Periodic boundary conditions are enforced with (i+1) % N
# ============================================================
H = np.zeros((dim, dim), dtype=float)

for idx, state in enumerate(basis_states):
    for i in range(N):
        j = (i + 1) % N   # periodic boundary condition

        si = state[i]
        sj = state[j]

        # Diagonal term: S_i^z S_j^z
        H[idx, idx] += J * 0.25 * si * sj

        # Off-diagonal flip-flop term:
        # only acts on antiparallel neighbors
        if si != sj:
            flipped_state = state.copy()
            flipped_state[i] *= -1
            flipped_state[j] *= -1
            flipped_idx = state_to_index[tuple(flipped_state)]

            H[idx, flipped_idx] += 0.5 * J

# Optional symmetry cleanup against tiny numerical asymmetry
H = 0.5 * (H + H.T)

# ============================================================
# Diagonalize the Hamiltonian
# ============================================================
eigenvalues, eigenvectors = np.linalg.eigh(H)

# ============================================================
# Precompute magnetization expectation values in the energy eigenstates
#
# For each eigenstate |n>, compute:
#   <Mz>_n  = <n|Mz|n>
#   <Mz^2>_n = <n|Mz^2|n>
#
# Since Mz is diagonal in the basis, these are easy to evaluate from
# the eigenvector coefficients.
# ============================================================
probabilities = np.abs(eigenvectors)**2   # probabilities in basis states
Mz_eig = probabilities.T @ Mz_basis
Mz2_eig = probabilities.T @ (Mz_basis**2)

# ============================================================
# Thermodynamic functions
# Use shifted Boltzmann weights for numerical stability:
#   w_n = exp(-(E_n - E0)/T)
# This does not change normalized averages.
# ============================================================
E0 = eigenvalues[0]

def boltzmann_weights(T):
    return np.exp(-(eigenvalues - E0) / T)

def partition_function(T):
    w = boltzmann_weights(T)
    return np.sum(w)

def internal_energy(T):
    w = boltzmann_weights(T)
    Z = np.sum(w)
    return np.sum(eigenvalues * w) / Z

def specific_heat(T):
    w = boltzmann_weights(T)
    Z = np.sum(w)

    avg_E = np.sum(eigenvalues * w) / Z
    avg_E2 = np.sum((eigenvalues**2) * w) / Z

    return (avg_E2 - avg_E**2) / (T**2)

def magnetic_susceptibility(T):
    w = boltzmann_weights(T)
    Z = np.sum(w)

    avg_Mz = np.sum(Mz_eig * w) / Z
    avg_Mz2 = np.sum(Mz2_eig * w) / Z

    # Zero-field susceptibility (k_B = 1)
    return (avg_Mz2 - avg_Mz**2) / T

# ============================================================
# Temperature range
# ============================================================
temperatures = np.linspace(0.1, 5.0, 100)

C_V_list = []
chi_list = []
U_list = []

for T in temperatures:
    U_list.append(internal_energy(T))
    C_V_list.append(specific_heat(T))
    chi_list.append(magnetic_susceptibility(T))

# ============================================================
# Print a few spectral results
# ============================================================
print("Lowest 10 eigenvalues:")
for n, E in enumerate(eigenvalues[:10]):
    print(f"{n:2d}: {E:.8f}")

# ============================================================
# Plot specific heat and magnetic susceptibility
# ============================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(temperatures, C_V_list, linewidth=2)
plt.title("Specific Heat $C_V(T)$")
plt.xlabel("Temperature $T$")
plt.ylabel("$C_V$")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(temperatures, chi_list, linewidth=2)
plt.title("Magnetic Susceptibility $\\chi(T)$")
plt.xlabel("Temperature $T$")
plt.ylabel("$\\chi$")
plt.grid(True)

plt.tight_layout()
plt.show()
