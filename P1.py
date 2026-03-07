import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0  # Reduced Planck constant
m = 1.0     # Mass of the particle
L = 1.0     # Length of the box

# Spatial discretization
N = 1000   # Number of points
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Construct the Hamiltonian matrix
# Using finite difference: -hbar^2/(2m) * d^2/dx^2

# Kinetic energy operator
diag = np.full(N, -2.0)
off_diag = np.ones(N - 1)
laplacian = (np.diag(diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)) / dx**2

# Hamiltonian
H = -(hbar**2) / (2 * m) * laplacian

# Apply boundary conditions (wavefunction zero at x=0 and x=L)
# This is implicit in the finite difference method as we exclude the boundary points

# Solve the eigenvalue problem
eigenvalues, eigenvectors = np.linalg.eigh(H)

# Extract the first three eigenvalues and eigenvectors
E_numerical = eigenvalues[:3]
psi_numerical = eigenvectors[:, :3]

# Analytical solutions for comparison
n_values = np.array([1, 2, 3])
E_analytical = (n_values**2 * np.pi**2 * hbar**2) / (2 * m * L**2)

# Print the numerical and analytical energies
print("Numerical Energies:")
for i, E in enumerate(E_numerical):
    print(f"E_{i+1} = {E:.5f}")

print("\nAnalytical Energies:")
for i, E in enumerate(E_analytical):
    print(f"E_{i+1} = {E:.5f}")

# Normalize the eigenfunctions
psi_numerical = psi_numerical / np.sqrt(dx)

# Plot the first three eigenfunctions
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(x, psi_numerical[:, i], label=f"n={i+1}")

plt.title("First Three Energy Eigenfunctions of a Particle in a Box")
plt.xlabel("Position x")
plt.ylabel("Wavefunction ψ_n(x)")
plt.legend()
plt.grid(True)
plt.show()
