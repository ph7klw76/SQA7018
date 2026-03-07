import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0
m = 1.0
omega = 1.0
x_max = 5.0

# Spatial discretization
N = 1000
x = np.linspace(-x_max, x_max, N)
dx = x[1] - x[0]

# Potential energy
V = 0.5 * m * omega**2 * x**2

# Construct the Hamiltonian matrix
diag = hbar**2 / (m * dx**2) + V
off_diag = -hbar**2 / (2 * m * dx**2) * np.ones(N - 1)
H = np.diag(diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)

# Solve the eigenvalue problem
eigenvalues, eigenvectors = np.linalg.eigh(H)

# Extract the first three eigenvalues and eigenvectors
E_numerical = eigenvalues[:3]
psi_numerical = eigenvectors[:, :3]

# Analytical solutions
n_values = np.array([0, 1, 2])
E_analytical = hbar * omega * (n_values + 0.5)

# Print numerical and analytical energies
print("Numerical Energies:")
for i, E in enumerate(E_numerical):
    print(f"E_{i} = {E:.5f}")

print("\nAnalytical Energies:")
for i, E in enumerate(E_analytical):
    print(f"E_{i} = {E:.5f}")

# Normalize the eigenfunctions
psi_numerical /= np.sqrt(dx)

# Plot the first three eigenfunctions
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(x, psi_numerical[:, i], label=f"n={i}")

plt.title("First Three Energy Eigenfunctions of Quantum Harmonic Oscillator")
plt.xlabel("Position x")
plt.ylabel("Wavefunction ψ_n(x)")
plt.legend()
plt.grid(True)
plt.show()
