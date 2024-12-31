# 1. Introduction

Superconductors in an external magnetic field exhibit macroscopic quantum phenomena, including magnetic flux quantization and vortex formation. To simulate these effects, one can use the Time-Dependent Ginzburg-Landau (TDGL) equations—a phenomenological PDE system describing how the superconducting order parameter evolves in space and time when magnetic fields and thermal effects are present.

We focus on a 2D geometry for simplicity and assume a uniform magnetic field $B_0 \hat{z}$. We want to track how the complex order parameter $\psi(r, t)$ changes over time, forming vortex patterns or reaching a steady-state configuration.

---

# 2. Time-Dependent Ginzburg-Landau (TDGL) Equations

## 2.1 Background

The static Ginzburg-Landau theory uses a free energy functional:

$$
F[\psi, A] = \int \left( 
\alpha |\psi|^2 + 
\frac{\beta}{2} |\psi|^4 + 
\frac{1}{2m} |(-i\hbar \nabla - 2e A) \psi|^2 + 
\frac{|\mathbf{B}|^2}{2\mu_0} 
\right) d^3r,
$$

where $A$ is the electromagnetic vector potential and $\mathbf{B} = \nabla \times A$.

The Time-Dependent version adds a dynamical equation for $\psi(r, t)$. A simplified form often used is:

$$
\frac{\partial \psi}{\partial t} = -\Gamma \left[ 
\alpha \psi + \beta |\psi|^2 \psi - 
\frac{\hbar^2}{2m} \left( \nabla - i\frac{2e}{\hbar} A \right)^2 \psi 
\right],
$$

where $\Gamma$ (often written as $\gamma$ in code) is a relaxation parameter. In our code, we set $\Gamma = \gamma$. The complex amplitude $\psi$ can vary in both magnitude (Cooper-pair density) and phase (related to supercurrent flow).

---

## 2.2 Specific TDGL Equation in the Code

For convenience, we adopt dimensionless constants or simple scalings for $\hbar$, $e$, and $m$. The discrete PDE in code form is:

$$
\frac{\partial \psi}{\partial t} = 
-\gamma \left[ 
\alpha \psi + \beta |\psi|^2 \psi - 
\frac{\hbar^2}{2m} \left( D^\wedge \right)^2 \psi 
\right],
$$

where the covariant Laplacian $D^\wedge^2$ encodes $\nabla - i\frac{2e}{\hbar} A$ in a gauge-invariant way.

---

# 3. Gauge Invariance and Link Variables

In the presence of a magnetic vector potential $A(r)$, naive finite differences of $\nabla \psi$ plus $-i\frac{2e}{\hbar}A\psi$ can break gauge invariance if not implemented carefully. A standard remedy in lattice gauge theory is to use link variables.

### Link Variable $U_x(i, j)$

On the lattice edge from $(i, j)$ to $(i+1, j)$, the link variable is:

$$
U_x(i, j) = \exp \left[ i \phi_x(i, j) \right],
$$

where:

$$
\phi_x(i, j) \approx \frac{2e}{\hbar} \int_{x_i}^{x_i + \Delta x} A_x(x, y_j) \, dx.
$$

### Link Variable $U_y(i, j)$

Similarly, for the edge from $(i, j)$ to $(i, j+1)$:

$$
U_y(i, j) = \exp \left[ i \phi_y(i, j) \right].
$$

By multiplying the wavefunction $\psi$ at neighboring sites with the appropriate phase factor from the link variable, one can mimic the continuous covariant derivative $D^\wedge = -i\hbar \nabla - 2eA$ on a discrete grid. This ensures that if you perform a gauge transformation on $\psi$, the link variables transform correspondingly, preserving gauge invariance of the physical results.

---

# 4. Numerical Implementation

## 4.1 Building Uniform-Field Link Variables

Our code sets a uniform magnetic field $B_0 \hat{z}$. We choose a symmetric gauge in continuous form:

$$
A_x(x, y) = -\frac{1}{2} B_0 y, \quad A_y(x, y) = +\frac{1}{2} B_0 x.
$$

For each grid point $(i, j)$ at coordinate $(x_i, y_j)$, we compute:

$$
U_x(i, j) = \exp \left[ i \left( \frac{2e}{\hbar} \right) A_x(i, j) \Delta x \right],
$$

$$
U_y(i, j) = \exp \left[ i \left( \frac{2e}{\hbar} \right) A_y(i, j) \Delta y \right].
$$

Thus, each horizontal link $U_x$ and vertical link $U_y$ is precomputed and stored in arrays. The code snippet:

```python
Ux[i, j] = np.exp(1j * phi_x)
Uy[i, j] = np.exp(1j * phi_y)
```
# 4. Numerical Implementation

## 4.2 Covariant Laplacian in 2D

We define forward and backward difference operators that include link variables:

$$
D_+^x \psi(i, j) = \frac{U_x(i, j) \psi(i+1, j) - \psi(i, j)}{\Delta x},
$$

$$
D_-^x \psi(i, j) = \frac{\psi(i, j) - U_x(i-1, j)^* \psi(i-1, j)}{\Delta x}.
$$

(Note $U^*$ is the complex conjugate, appropriate for traveling in the reverse direction on the lattice link.)

We then apply these again to form:

$$
D_+^x (D_-^x \psi)(i, j) + D_+^y (D_-^y \psi)(i, j).
$$

This combination is the discrete version of the covariant Laplacian $D^\wedge^2$. In the code, we handle periodic boundary conditions by using NumPy’s `roll` function to wrap indices around the domain edges.

---

## 4.3 Time-Stepping Method

We use a forward Euler approach:

$$
\psi^{(n+1)} = \psi^{(n)} + \Delta t \frac{\partial \psi}{\partial t},
$$

where:

$$
\frac{\partial \psi}{\partial t} = -\gamma \left[ 
\alpha \psi + \beta |\psi|^2 \psi - \frac{\hbar^2}{2m} D^\wedge^2 \psi 
\right].
$$

Hence:

$$
\psi^{(n+1)} = \psi^{(n)} - \Delta t \gamma \left[
\alpha \psi^{(n)} + \beta |\psi^{(n)}|^2 \psi^{(n)} - \left(\frac{\hbar^2}{2m}\right) D^\wedge^2 \psi^{(n)}
\right].
$$

Though explicit Euler is the simplest method, it can be conditionally stable—so $\Delta t$ must be sufficiently small to avoid divergence.

---

# 5. The Code Workflow

## Initialization

A 2D array $\psi[i, j]$ is populated with random noise to break symmetry, ensuring vortex states can emerge.

---

## Build Link Variables

For each $(i, j)$, compute $\exp(i \cdot \text{phase})$ from the local gauge potential $A_x, A_y$. Store in $U_x, U_y$.

---

## Covariant Laplacian

Implement the discrete operators $D_+^x, D_-^x, D_+^y, D_-^y$. Sum them to get $D^\wedge^2 \psi$.

---

## TDGL Time Step

At each step:
1. Compute the nonlinear terms $\alpha \psi + \beta |\psi|^2 \psi$ and the covariant Laplacian terms.
2. Update $\psi$ with a forward Euler rule.

---

## Animation

Using `matplotlib.animation.FuncAnimation`, we repeatedly call an update function. Between frames, we apply multiple Euler steps. At each frame, we plot $|\psi|$.

This sequence reveals how $\psi$ evolves to form vortex cores (where $\psi \approx 0$) or other patterns typical of Type II superconductors in an external magnetic field.

---

# 6. Interpreting the Results

## 6.1 Vortex Lattices

For sufficiently large $\kappa = \lambda / \xi$ (the Ginzburg-Landau parameter) and moderate field $B_0$, the system may form an Abrikosov vortex lattice. In 2D simulations, one sees bright (large $|\psi|$) regions separated by small “holes” (vortex cores where $|\psi| \approx 0$) arranged in a roughly triangular pattern.

---

## 6.2 Parameter Sensitivity

- $\alpha < 0$ is essential to favor a superconducting (nonzero $\psi$) state.
- $\beta > 0$ ensures the amplitude of $\psi$ stays finite.
- $\gamma$ dictates how fast the system relaxes.
- $B_0$: Increasing the magnetic field typically raises the vortex density.
- $\Delta t$: Must be chosen small enough for numerical stability in forward Euler.

---

# 7. Extensions and Improvements

## Implicit or Semi-Implicit Methods

Forward Euler can suffer from instabilities if time steps are not sufficiently small. More advanced PDE solvers (e.g., Crank-Nicolson) can provide better stability.

---

## Self-Consistent Magnetic Field

In the code, $A$ is fixed for a uniform $B_0$. For a fully self-consistent solution, one must solve $\nabla \times B = \mu_0 J$ in tandem, updating $A$ each step.

---

## Three-Dimensional Systems

The method extends to 3D with more complexity in boundary conditions, link variables, and memory usage.

---

## Pinning Potentials

Real superconductors often have impurities or engineered defects that pin vortices. One can add local variations in $\alpha(r)$ or other parameters to simulate pinning landscapes.

---

# 8. Conclusion

This 2D TDGL code demonstrates a gauge-invariant numerical approach to simulating superconductivity in the presence of a uniform magnetic field. By combining:

- Phenomenological PDE (Time-Dependent Ginzburg-Landau),
- Link-variable discretization for the covariant derivative,
- Forward Euler time integration, and
- Matplotlib animation to visualize $|\psi|$,

we capture key physics of Type II superconductors, including vortex nucleation and flux-line lattices. While simplified, the methodology serves as a foundation for more advanced simulations—incorporating self-consistent fields, pinning, and 3D geometries—to explore the rich macroscopic quantum behavior of superconducting materials.

