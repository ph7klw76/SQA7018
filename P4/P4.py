import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Domain discretization
N = 50
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Number of unknowns
N_unknowns = N * N

# Initialize the coefficient matrix and right-hand side vector
A = lil_matrix((N_unknowns, N_unknowns))
b = np.zeros(N_unknowns)

# Map 2D indices to 1D indices
def idx(i, j):
    return i + j * N

# Set up the equations
for j in range(N):
    for i in range(N):
        k = idx(i, j)
        if i == 0:
            # Left boundary
            A[k, k] = 1
            b[k] = 0
        elif i == N - 1:
            # Right boundary
            A[k, k] = 1
            b[k] = np.sin(np.pi * y[j])
        elif j == 0 or j == N - 1:
            # Top and bottom boundaries
            A[k, k] = 1
            b[k] = 0
        else:
            # Interior points
            A[k, idx(i+1, j)] = 1 / dx**2
            A[k, idx(i-1, j)] = 1 / dx**2
            A[k, idx(i, j+1)] = 1 / dy**2
            A[k, idx(i, j-1)] = 1 / dy**2
            A[k, k] = -2 * (1 / dx**2 + 1 / dy**2)
            b[k] = 0

# Convert A to CSR format for efficient solving
A_csr = A.tocsr()

# Solve the linear system
phi = spsolve(A_csr, b)

# Reshape phi back to 2D
phi_2d = phi.reshape((N, N))

# Plot the potential
X, Y = np.meshgrid(x, y)
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, phi_2d.T, 50, cmap='viridis')
plt.colorbar(cp)
plt.title('Potential φ(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
