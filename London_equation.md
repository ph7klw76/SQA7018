# Illustrative Example: Solving a 2D Version of the London Equation

Below is an illustrative (and somewhat simplified) example of how you might solve a 2D version of the London equation:

$$
\nabla \times j_s = -\frac{n_s e^2}{m} B \quad \longleftrightarrow \quad \nabla^2 B = \frac{1}{\lambda_L^2} B
$$

using a finite-difference approach in Python, then plotting the solution over a 2D mesh. We will focus on one component of the magnetic field (e.g., $B_z$) in the $(x, y)$ plane, which is common in illustrative 2D superconductivity problems. This example is meant to demonstrate the numerical setup and solution procedure; in real applications, boundary conditions and geometry may be more sophisticated.

---

## 1. Background: From the Second London Equation to a PDE

### 1.1 Second London Equation (Vector Form)

$$
\nabla \times j_s = -\frac{n_s e^2}{m} B.
$$

### 1.2 Relation to Ampère’s Law
Ampère’s law (neglecting displacement current) gives:

$$
\nabla \times B = \mu_0 j_s \quad \Rightarrow \quad j_s = \frac{1}{\mu_0} \nabla \times B.
$$

### 1.3 Combine Both
Taking the curl of $j_s$ on the right side and using:

$$
\nabla \times (\nabla \times B) = \nabla (\nabla \cdot B) - \nabla^2 B,
$$

and noting $\nabla \cdot B = 0$ (since there are no magnetic monopoles), one obtains:

$$
\nabla^2 B = \mu_0 \frac{n_s e^2}{m} B = \frac{1}{\lambda_L^2} B,
$$

where:

$$
\lambda_L^2 = \frac{m}{\mu_0 n_s e^2}
$$

is the London penetration depth.

In a 2D geometry where the field is primarily in the $z$-direction ($B = B_z(x, y) \hat{z}$), the PDE simplifies to:

$$
\frac{\partial^2 B_z}{\partial x^2} + \frac{\partial^2 B_z}{\partial y^2} = \frac{1}{\lambda_L^2} B_z.
$$

That is the equation we will solve numerically.

---

## 2. Setting Up the Finite-Difference Scheme

### 2.1 Discretization
Let:

$$
x_i = i \Delta x, \quad \text{for } i = 0, 1, \dots, N_x - 1,
$$

$$
y_j = j \Delta y, \quad \text{for } j = 0, 1, \dots, N_y - 1.
$$

We store $B_z$ in a 2D array $B[i, j] = B_z(x_i, y_j)$. The Laplacian $\nabla^2 B_z$ at grid point $(i, j)$ can be approximated by the standard central difference:

$$
\nabla^2 B_z \big|_{(i, j)} \approx \frac{B_{i+1, j} - 2 B_{i, j} + B_{i-1, j}}{\Delta x^2} + \frac{B_{i, j+1} - 2 B_{i, j} + B_{i, j-1}}{\Delta y^2}.
$$

Hence, our PDE in discrete form is:

$$
\frac{B_{i+1, j} - 2 B_{i, j} + B_{i-1, j}}{\Delta x^2} + \frac{B_{i, j+1} - 2 B_{i, j} + B_{i, j-1}}{\Delta y^2} = \frac{1}{\lambda_L^2} B_{i, j}.
$$

We can rearrange this to solve for $B_{i, j}$ iteratively:

$$
B_{i, j} = \frac{\frac{B_{i+1, j} + B_{i-1, j}}{\Delta x^2} + \frac{B_{i, j+1} + B_{i, j-1}}{\Delta y^2}}{\frac{2}{\Delta x^2} + \frac{2}{\Delta y^2} - \frac{1}{\lambda_L^2}}.
$$

---

### 2.2 Boundary Conditions
We must specify boundary conditions. Typical scenarios might be:
- $B_z$ fixed on one boundary (e.g., a constant external field on the left edge).
- $B_z = 0$ on far edges, simulating the field having decayed to zero at large distances.
- Or periodic boundary conditions in some directions (less common for a Meissner problem).

In the demo below, we will assume:
- $B_z = B_0$ at $x = 0$ (left boundary),
- $B_z = 0$ at $x = L_x$ (right boundary), $y = 0$ (bottom), and $y = L_y$ (top).

You can adapt this to suit your specific physical scenario.

---

## 3. Example Python Code (Iterative Relaxation)

Below is a Python implementation of the finite-difference method to solve the 2D London equation. The code sets up a 2D mesh, applies the boundary conditions, and solves for $B_z$ iteratively.

```python
import numpy as np
import matplotlib.pyplot as plt

def solve_london_2d(nx=50, ny=50, 
                    lx=5.0, ly=5.0, 
                    lambda_L=1.0, 
                    b_left=1.0, max_iter=50000, tol=2e-3):
    """
    Solve the 2D London-type equation:
        d^2 B/dx^2 + d^2 B/dy^2 = (1/lambda_L^2)*B
    on a rectangular grid (0..lx) x (0..ly) 
    using finite differences + iterative relaxation.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in x and y directions.
    lx, ly : float
        Physical size of the domain in x and y directions.
    lambda_L : float
        London penetration depth (assumed constant).
    b_left : float
        Boundary condition: Bz at x=0.
    max_iter : int
        Maximum number of relaxation iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    B : 2D numpy array
        Numerical solution for Bz on the (nx x ny) grid.
    X, Y : 2D numpy arrays
        Meshgrid coordinates for plotting.
    """
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    # Precompute denominators to speed up iteration
    denom = (2.0/dx**2 + 2.0/dy**2 - 1.0/lambda_L**2)

    # Initialize B array with zeros
    B = np.zeros((nx, ny), dtype=float)

    # Boundary conditions
    # B(x=0,y) = b_left, B=0 on other boundaries
    B[0, :] = b_left     # left boundary
    B[-1, :] = 0.0       # right boundary
    B[:, 0] = 0.0        # bottom boundary
    B[:, -1] = 0.0       # top boundary

    # Iterative relaxation (Gauss-Seidel or Jacobi)
    for iteration in range(max_iter):
        max_diff = 0.0

        # Make a copy of B if using Jacobi; Gauss-Seidel can update in-place
        B_new = B.copy()

        # Update interior points (i=1..nx-2, j=1..ny-2)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # Laplacian part
                lap = ((B[i+1, j] + B[i-1, j]) / dx**2 +
                       (B[i, j+1] + B[i, j-1]) / dy**2)
                # Solve the PDE at (i,j)
                B_new[i, j] = lap / denom

                diff = abs(B_new[i, j] - B[i, j])
                if diff > max_diff:
                    max_diff = diff

        B = B_new

        # Check for convergence
        if max_diff < tol:
            print(f"Converged after {iteration} iterations.")
            break
    else:
        print(f"Warning: Did not fully converge after {max_iter} iterations.")

    # Create mesh grids for plotting
    x_vals = np.linspace(0, lx, nx)
    y_vals = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    return B, X, Y

# --- Run the solver and plot the results ---
if __name__ == "__main__":
    nx, ny = 60, 60
    lx, ly = 6.0, 6.0
    lambda_L = 1.0
    b_left = 1.0

    B, X, Y = solve_london_2d(nx, ny, lx, ly, lambda_L, b_left)

    # Plot
    plt.figure(figsize=(6,5))
    cp = plt.contourf(X, Y, B, 50, cmap='viridis')
    plt.colorbar(cp, label='Bz Field')
    plt.title("2D London Equation Solution (Bz)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
```
![image](https://github.com/user-attachments/assets/56e3dfe5-0e37-4bd2-a103-d91a829f494d)

# Explanation of the Key Parts

## Discretization:
- We divide $[0, L_x]$ into $n_x - 1$ segments of size $\Delta x$. Similarly for $[0, L_y]$.
- We store $B_{i,j} \equiv B_z(x_i, y_j)$ in a 2D array.

## Boundary Conditions:
- `B[0, :] = b_left` enforces $B_z = b_{\text{left}}$ at $x = 0$.
- `B[-1, :] = 0, B[:, 0] = 0, B[:, -1] = 0` set $B_z = 0$ on the other three boundaries.

## Iteration:
- We use a simple Jacobi (or Gauss-Seidel) method to update interior points based on the finite-difference stencil of the PDE.
- The loop continues until either the solution converges (i.e., updates become smaller than a chosen tolerance `tol`) or we reach `max_iter`.

## Plotting:
- We use `matplotlib.pyplot.contourf` to produce a filled contour plot of the solution $B_z(x, y)$.

---

# 4. Interpreting the Results

- In a typical superconducting scenario, you would see $B_z$ decaying from its boundary value ($b_{\text{left}}$) toward zero within the bulk of the domain, reflecting the Meissner screening.
- The characteristic decay length is related to $\lambda_L$. A smaller $\lambda_L$ forces the field to drop off more steeply.

---

# 5. Reconstructing $j_s$ (Optional)

Once you have $B_z$, you could approximately reconstruct the supercurrent $j_s$ via:

$$
j_s = \frac{1}{\mu_0} \nabla \times B.
$$

In 2D with $B = B_z \hat{z}$, the curl in the plane yields:

$$
(\nabla \times B)_x = \frac{\partial B_z}{\partial y}, \quad (\nabla \times B)_y = -\frac{\partial B_z}{\partial x}.
$$

Hence, if you store $B[i,j]$, you can finite-difference to get:

$$
j_{s,x} \approx \frac{1}{\mu_0} \frac{B_{i,j+1} - B_{i,j-1}}{2 \Delta y}, \quad
j_{s,y} \approx -\frac{1}{\mu_0} \frac{B_{i+1,j} - B_{i-1,j}}{2 \Delta x}.
$$

This allows you to visualize current streamlines in the superconductor.

---

# 6. Notes and Extensions

### Dimensional Consistency:
- Make sure that the constants, e.g., $\mu_0$, $\lambda_L$, $n_s$, etc., are in correct SI units if you introduce them explicitly. This example keeps them abstracted or set to 1 for demonstration.

### 3D Extension:
- In realistic superconductors, you might want to solve in 3D. The approach is similar but with a 3D grid and higher memory/compute demands.

### Nontrivial Geometries:
- In practice, superconductors may have cylindrical or spherical geometries, requiring either:
  - (a) a more complex mesh, or
  - (b) re-casting the PDE in cylindrical/spherical coordinates.

### Higher-Order Methods:
- The simple second-order central difference can be replaced by higher-order finite differences, finite elements, or spectral methods for improved accuracy.

### Comparisons to Analytical Solutions:
- Whenever possible, compare your numerical solution to known 1D or 2D analytical solutions (e.g., exponential decay in the simplest slab geometry) to validate correctness.

---

## 6. Example Python Code (Direct Sparse Solve)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def index(i, j, nx):
    """ Map 2D grid index (i, j) to 1D index k. """
    return j*nx + i

def solve_london_2d_direct(nx=50, ny=50,
                           lx=5.0, ly=5.0,
                           lambda_L=1.0,
                           b_left=1.0):
    """
    Solve (d^2 B/dx^2 + d^2 B/dy^2) = (1/lambda_L^2)* B
    in 2D using a direct sparse solver.

    Dirichlet boundary conditions:
      B = b_left at x=0
      B = 0      at x=lx
      B = 0      at y=0
      B = 0      at y=ly
    """
    dx = lx / (nx-1)
    dy = ly / (ny-1)

    # We want:  Lap(B) - (1/lambda_L^2)*B = 0
    # or equivalently Lap(B) = (1/lambda_L^2)*B
    # We'll code it as: Lap(B) - alpha^2 * B = 0 with alpha^2 = 1/lambda_L^2
    alpha2 = 1.0 / lambda_L**2

    # Number of unknowns in interior + boundary
    N = nx * ny

    # Sparse matrix in "list of lists" format
    A = lil_matrix((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)

    # Helper constants
    c_x = 1.0 / dx**2
    c_y = 1.0 / dy**2
    # Discretization: (B_{i+1,j} + B_{i-1,j} - 2*B_{i,j})/dx^2 + ...
    #                (B_{i,j+1} + B_{i,j-1} - 2*B_{i,j})/dy^2
    #                - alpha2 * B_{i,j} = 0

    for j in range(ny):
        for i in range(nx):
            k = index(i, j, nx)

            # Check boundary
            if i == 0:
                # x=0 boundary => B = b_left
                A[k, k] = 1.0
                b[k] = b_left
            elif i == nx-1 or j == 0 or j == ny-1:
                # other boundaries => B=0
                A[k, k] = 1.0
                b[k] = 0.0
            else:
                # interior point
                # Laplacian contribution
                A[k, k]     = -2.0*(c_x + c_y) - alpha2
                A[k, k+1]   = c_x   # i+1
                A[k, k-1]   = c_x   # i-1
                A[k, k+nx]  = c_y   # j+1
                A[k, k-nx]  = c_y   # j-1
                # b[k] stays zero

    # Convert to CSR and solve
    A_csr = A.tocsr()
    B_vec = spsolve(A_csr, b)

    # Reshape solution into 2D
    B_2D = B_vec.reshape((ny, nx))  # note shape=(ny,nx) if we used j*nx + i

    # Create mesh grids for plotting
    x_vals = np.linspace(0, lx, nx)
    y_vals = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    return B_2D, X, Y

# --- Example usage ---
if __name__ == "__main__":
    nx, ny = 60, 60
    lx, ly = 6.0, 6.0
    lambda_L = 1.0
    b_left = 1.0

    B, X, Y = solve_london_2d_direct(nx, ny, lx, ly, lambda_L, b_left)

    plt.figure(figsize=(6,5))
    cp = plt.contourf(X, Y, B, 50, cmap='viridis')
    plt.colorbar(cp, label='Bz')
    plt.title("2D London Equation (Direct Sparse Solve)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
```

# 6.1. The PDE and Boundary Conditions

We consider the partial differential equation (PDE):

$$
\frac{\partial^2 B}{\partial x^2} + \frac{\partial^2 B}{\partial y^2} = \frac{1}{\lambda_L^2} B,
$$

on a rectangular domain $0 \leq x \leq L_x$, $0 \leq y \leq L_y$. The boundary conditions are:

- $B = b_{\text{left}}$ on $x = 0$,
- $B = 0$ on the other three sides: $x = L_x$, $y = 0$, $y = L_y$.

We discretize the domain into an $(n_x \times n_y)$ grid and approximate derivatives with finite differences.

---

# 6.2. Method2: Direct Sparse Solve

In the first approach:

```python
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
```

The code assembles a large sparse linear system that represents the PDE on the discrete grid:

$$
A B_{\text{vec}} = b,
$$

where:

- $B_{\text{vec}}$ is the vector of unknowns $B_{i,j}$ (flattened from 2D to 1D).
- $A$ is a sparse matrix encoding the finite-difference stencil and boundary conditions.
- $b$ is the right-hand side vector (including boundary values).

---

## 6.3 Construction of the Sparse Matrix

### Numbering the Grid Points:
A helper function, `index(i, j, nx)`, maps the 2D coordinates $(i, j)$ to a single index $k$. In the code:

```python
def index(i, j, nx):
    return i * nx + j
```

This means $k$ goes from $0$ to $N-1$, where $N = n_x \times n_y$.

---

## Filling the Matrix $A$:

- **Boundary Points**:  
  If $(i, j)$ is on a boundary, we enforce $B_{i,j} = \text{constant}$.  
  The code does this by setting:
  $$
  A[k, k] = 1, \quad b[k] = \text{boundary value} \quad \text{(Dirichlet condition)}.
  $$

- **Interior Points**:  
  If $(i, j)$ is an interior point, we use the finite-difference approximation for the Laplacian:
  $$
  \frac{B_{i+1, j} - 2 B_{i, j} + B_{i-1, j}}{\Delta x^2} + 
  \frac{B_{i, j+1} - 2 B_{i, j} + B_{i, j-1}}{\Delta y^2} - 
  \alpha^2 B_{i, j} = 0,
  $$
  where $\alpha^2 = 1 / \lambda_L^2$.

  Hence, we place the corresponding coefficients in the matrix $A$ and $b$. For example:
  - $A[k, k]$ gets $-2 (c_x + c_y) - \alpha^2$,
  - $A[k, k \pm 1]$ gets $c_x$, corresponding to $\pm 1$ step in the $x$-direction,
  - $A[k, k \pm n_x]$ gets $c_y$, corresponding to $\pm 1$ step in the $y$-direction.

---

### Sparsity:
Notice that each row in $A$ has only about 5 nonzero entries (the center point, plus up to 4 neighbors), so $A$ is very sparse.

---

## 6.4 Solving the System

After filling the matrix:

```python
from scipy.sparse import lil_matrix
import numpy as np

A = lil_matrix((N, N), dtype=np.float64)
b = np.zeros(N, dtype=np.float64)
```
We set the boundary or interior coefficients. Then:

- **Convert $A$ to CSR format** (more efficient for linear algebra):
  
```python
  A_csr = A.tocsr()
 ```

Solve directly using SciPy’s sparse solver

```python
from scipy.sparse.linalg import spsolve
B_vec = spsolve(A_csr, b)
 ```

This call attempts to factorize $A$ and solve the system in a single step.

Reshape `B_vec` back into a 2D array for plotting:

```python
B_2D = B_vec.reshape((ny, nx))
```

This approach is called a **direct method** because it (conceptually) inverts the matrix $A$. In practice, SciPy might use a factorization like LU or a similar technique optimized for sparse matrices.

---

## 7.1 Pros and Cons of the Direct Method

### Pros:
- If the matrix is not too large (i.e., if $n_x \times n_y$ is manageable), the direct solve typically converges in one step (no iteration needed).
- It’s systematically robust: for well-posed, nonsingular systems, it will give an answer that meets numerical precision.

### Cons:
- As the grid gets very fine (e.g., thousands in each dimension), the matrix size becomes huge, and factorization can be time-consuming and memory-intensive.
- If you only need an approximate solution or have a large 3D mesh, a direct factorization might be too expensive.

# Iterative Relaxation (Jacobi/Gauss–Seidel)

In contrast, the second code snippet uses an **iterative approach**. It does not build a global matrix $A$. Instead, it uses the finite-difference stencil to update the solution in place in multiple sweeps over the grid.

---

## 8.1 Core Idea of Iterative Relaxation

1. Initialize $B_{i,j} = 0$ for interior points (or some initial guess).
2. Enforce boundary conditions.
3. Iteratively update the interior points based on their neighbors until the solution converges (or reaches a maximum iteration count).

For the PDE:

$$
\nabla^2 B - \alpha^2 B = 0,
$$

the finite-difference rearrangement at an interior point $(i, j)$ often looks like:

$$
B_{i,j}^{\text{new}} = \frac{\frac{B_{i+1,j} + B_{i-1,j}}{\Delta x^2} + \frac{B_{i,j+1} + B_{i,j-1}}{\Delta y^2}}{\frac{2}{\Delta x^2} + \frac{2}{\Delta y^2} - \alpha^2}.
$$

Then we replace $B_{i,j}$ by $B_{i,j}^{\text{new}}$. This is repeated iteratively until changes become small.

---

## 8.2 Implementation Details

1. **Loop** over `max_iter` times.
2. For each interior grid point:
   - Calculate the Laplacian approximation using neighbors:  
     `lap = (B[i+1,j] + ... ) / dx^2 + ... / dy^2`.
   - Solve for `B_new[i,j] = lap / denom`.
   - Track `max_diff = abs(B_new[i,j] - B[i,j])` to check convergence.
3. If `max_diff < tol` at the end of an iteration, we assume convergence.

No large matrix or direct solve is used. Instead, each iteration gradually relaxes the solution to the correct PDE solution.

---

### 8.2.1 Pros and Cons of the Iterative Method

**Pros**:
- Uses very little memory compared to storing a large matrix.
- Easy to implement for large 2D or 3D problems and can be parallelized or combined with advanced iterative solvers (e.g., multigrid).
- You can stop early if a rough approximation suffices.

**Cons**:
- It can converge slowly or even diverge if the update scheme is not stable (e.g., if denominators are negative).
- Needs many iterations to get high accuracy (especially near boundary layers).

---

## 9. Summary of the Differences

### **Matrix Assembly + Direct Solve**:
- We explicitly construct the global sparse linear system that represents the finite-difference discretization.
- Then we call a direct (or sometimes iterative) sparse solver from SciPy.
- **Pros**: Typically converges in one shot, up to machine precision, if the system is well-conditioned.

### **Iterative Relaxation**:
- We do not build a big matrix. Instead, we apply local update formulas over and over.
- **Pros**: Convergence depends on the chosen scheme (Jacobi, Gauss–Seidel, SOR, etc.), the relaxation factors, and tolerance.
- **Cons**: May be easier to implement in some contexts but can be slow or tricky to converge if the PDE or boundary conditions are stiff.

**Both methods discretize the same PDE and boundary conditions.** One goes the matrix-based route; the other uses an iterative in-place update. Both yield the solution $B(x, y)$, just by different numerical strategies.

---

## 10. When to Choose One vs. the Other

1. **Direct Solve**:
   - Often preferable for moderate grid sizes where factorization is feasible.
   - It’s a straightforward “assemble and solve” approach, typically with a high reliability of convergence.

2. **Iterative Solve**:
   - Better for very large problems (e.g., 3D with millions of unknowns) or for special PDEs (like Poisson’s equation) where more advanced iterative methods (multigrid, conjugate gradient, etc.) can be highly efficient and use less memory than a direct factorization.

In practice, many real-world PDE solvers rely on advanced iterative or multigrid methods that converge faster than basic Gauss–Seidel, while direct methods are used for smaller or special-structure problems.

---

## Final Takeaway

1. **Matrix-based direct approach**:
   - Build the linear system explicitly, then solve via `spsolve`.
   - Great for smaller to medium grids, yields a (near-)exact solution with one solve.

2. **Iterative approach**:
   - Use repeated updates from the finite-difference stencil.
   - More memory-friendly for large grids, but can require careful tuning of iteration parameters (relaxation factor, tolerance, etc.) and can converge slowly.

Both implement the same numerical approximation (central finite differences) to the London equation:

$$
\nabla^2 B - \frac{1}{\lambda_L^2} B = 0,
$$
and both yield solutions that match well—given enough iterations in the iterative version or correct setup in the direct version. The choice depends on problem size, available memory, and desired accuracy or speed.


# Conclusion

This Python snippet demonstrates how to solve a 2D London-type PDE on a rectangular domain using finite differences. It yields a spatial map of $B_z$, showing exponential-like decay from a boundary value, capturing the essence of the Meissner screening phenomenon in superconductors. More complex boundary conditions or geometries can be implemented similarly, reflecting real-world superconducting problems.


