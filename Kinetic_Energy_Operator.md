# The Kinetic Energy Operator in Quantum Mechanics

The kinetic energy operator in quantum mechanics is a crucial part of the Hamiltonian operator, representing the kinetic energy of a particle. Here's a detailed breakdown of its role and how the mathematics translates into the code:

---

## Kinetic Energy Operator in Quantum Mechanics

The kinetic energy operator in one dimension is defined as:

$$
\hat{T} = -\frac{\hbar^2}{2m} \frac{d^2}{dx^2},
$$

where:
- $\hbar$: Reduced Planck's constant.
- $m$: Mass of the particle.
- $\frac{d^2}{dx^2}$: The second derivative with respect to position, representing how the wavefunction $\psi(x)$ changes curvature.

In the Schrödinger equation, this operator appears as part of the Hamiltonian, which governs the total energy of the system:

$$
\hat{H} \psi(x) = E \psi(x), \quad \hat{H} = \hat{T} + \hat{V},
$$

where:
- $\hat{T}$: Kinetic energy operator.
- $\hat{V}$: Potential energy operator.

In the problem of the particle in a box with an infinite potential well, the potential energy $V(x)$ inside the box is zero. Thus, the Hamiltonian simplifies to:

$$
\hat{H} = -\frac{\hbar^2}{2m} \frac{d^2}{dx^2}.
$$

---

## Discretizing the Kinetic Energy Operator

To solve the Schrödinger equation numerically, we approximate the second derivative $\frac{d^2}{dx^2}$ using finite difference methods.

### 1. Finite Difference Approximation

The second derivative of a function $\psi(x)$ at a grid point $x_i$ can be approximated using the central difference formula:

$$
\frac{d^2 \psi}{dx^2} \bigg|_{x_i} \approx \frac{\psi_{i-1} - 2\psi_i + \psi_{i+1}}{\Delta x^2},
$$

where:
- $\psi_i = \psi(x_i)$: Value of the wavefunction at the $i$-th grid point.
- $\Delta x$: Spacing between adjacent grid points.

This formula results in a tridiagonal matrix when applied to all grid points in the domain.

---

### 2. Constructing the Laplacian Matrix

The finite difference approximation for the second derivative leads to the Laplacian operator (a matrix representation of $\frac{d^2}{dx^2}$):

$$
L = \frac{1}{\Delta x^2} 
\begin{bmatrix}
-2 & 1 & 0 & \cdots & 0 \\
1 & -2 & 1 & \cdots & 0 \\
0 & 1 & -2 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & -2
\end{bmatrix}.
$$

Here:
- The main diagonal contains $-2$ (representing the $-2\psi_i$ term).
- The sub- and superdiagonals contain $1$ (representing the $\psi_{i-1}$ and $\psi_{i+1}$ terms).

---

### In the Code:

```python
import numpy as np

# Parameters
N = 100  # Number of grid points
dx = 0.01  # Grid spacing

# Construct Laplacian matrix
diag = np.full(N, -2.0)          # Main diagonal
off_diag = np.ones(N - 1)        # Off-diagonals
laplacian = (np.diag(diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)) / dx**2
```
This code constructs the Laplacian matrix, which is then used to represent the kinetic energy operator in the numerical solution of the Schrödinger equation.

- **diag**: Creates the main diagonal ($-2$).
- **off_diag**: Creates the sub- and superdiagonals ($1$).
- **laplacian**: Combines them into a tridiagonal matrix and scales by $\frac{1}{\Delta x^2}$.

---

## 3. Including the Kinetic Energy Prefactor

The kinetic energy operator:

$$
\hat{T} = -\frac{\hbar^2}{2m} \frac{d^2}{dx^2}
$$

introduces a prefactor:

$$
\hat{T} = -\frac{\hbar^2}{2m} L,
$$

where $L$ is the Laplacian matrix.

### In the Code:

```python
H = -(hbar**2) / (2 * m) * laplacian
```

# Understanding the Second Derivative Approximation

The second derivative of a function $\psi(x)$ is central to many physical problems, including the Schrödinger equation. Using the finite difference method, we approximate the second derivative at a point $x_i$ as:

![image](https://github.com/user-attachments/assets/49e905fe-a2c3-46ab-a403-f28b406424ed)



### Here:

- $\psi_{i-1}$: The value of the function at the point to the left of $x_i$.
- $\psi_i$: The value of the function at $x_i$ (the point of interest).
- $\psi_{i+1}$: The value of the function at the point to the right of $x_i$.
- $\Delta x$: The spacing between adjacent grid points.

This formula comes from a Taylor expansion around $x_i$ and is accurate to second order ($O(\Delta x^2)$).

---

## Representing the Second Derivative as a Matrix

For a discretized domain with $N$ points, the second derivative operator is represented by a tridiagonal matrix acting on a vector of function values. Let's build this step by step:

### 1. Matrix Form of the Second Derivative

For a grid with $N$ points, let:

$$
\psi = [\psi_1, \psi_2, \dots, \psi_N]^T
$$

represent the wavefunction values at the grid points. The second derivative operator can be written as a matrix multiplication:

$$
\frac{d^2 \psi}{dx^2} \approx \frac{1}{\Delta x^2} 
\begin{bmatrix}
-2 & 1 & 0 & 0 & \cdots & 0 \\
1 & -2 & 1 & 0 & \cdots & 0 \\
0 & 1 & -2 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1 & -2 & 1 \\
0 & 0 & \cdots & 0 & 1 & -2
\end{bmatrix}
\begin{bmatrix}
\psi_1 \\
\psi_2 \\
\psi_3 \\
\vdots \\
\psi_{N-1} \\
\psi_N
\end{bmatrix}.
$$

---

### 2. Components of the Matrix

1. **Main Diagonal (-2):**
   - Represents the $-2\psi_i$ term for each grid point.

2. **Off-Diagonals (1):**
   - Represent the $\psi_{i-1}$ and $\psi_{i+1}$ contributions from the neighboring points.

3. **Boundary Conditions:**
   - At the edges ($x=0$ and $x=L$), the wavefunction $\psi(x)$ is zero for an infinite potential well.
   - This means the boundary points are implicitly excluded from the matrix.


![image](https://github.com/user-attachments/assets/37a2325a-5c19-433d-8a95-6eb9d2cf5ed4)

## Solution
```python
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
```


