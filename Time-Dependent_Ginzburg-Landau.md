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

![image](https://github.com/user-attachments/assets/c0f557cc-89f8-4873-aad6-0e335ae2be22)


where the covariant Laplacian

![image](https://github.com/user-attachments/assets/9f6a58d8-f7fb-42b9-b817-5d22542f0d68)  encodes ![image](https://github.com/user-attachments/assets/0f9f3ef1-d637-4767-a387-1023df61a359)  in a gauge-invariant way.

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

(Note ![image](https://github.com/user-attachments/assets/d7a819f0-f7dc-488e-ad3e-e951faa45821)
is the complex conjugate, appropriate for traveling in the reverse direction on the lattice link.)

We then apply these again to form:

![image](https://github.com/user-attachments/assets/96be708c-70fd-4b66-8f5e-72069e1873f8)


This combination is the discrete version of the covariant Laplacian ![image](https://github.com/user-attachments/assets/d7a819f0-f7dc-488e-ad3e-e951faa45821). In the code, we handle periodic boundary conditions by using NumPy’s `roll` function to wrap indices around the domain edges.

---

## 4.3 Time-Stepping Method

We use a forward Euler approach:

$$
\psi^{(n+1)} = \psi^{(n)} + \Delta t \frac{\partial \psi}{\partial t},
$$

where:

![image](https://github.com/user-attachments/assets/298872c4-7475-4d12-8b58-0647018e7e06)


Hence:

![image](https://github.com/user-attachments/assets/4f73090d-66db-4976-8691-6625697e24e7)


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

Implement the discrete operators ![image](https://github.com/user-attachments/assets/662b7721-b4d6-40c0-bf5c-161a6950c37d)
. Sum them to get 
![image](https://github.com/user-attachments/assets/69376033-5ed4-465d-9793-5bce2ca07c85)


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


Code
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

###############################################################################
# 1. Link Variable Construction for a Uniform Magnetic Field
###############################################################################

def build_uniform_field_links(nx, ny, dx, B0):
    """
    Build the 2D link variables Ux, Uy for a uniform magnetic field B0 in the z-direction.
    Using (for example) the symmetric gauge:
        A_x = -0.5 * B0 * y
        A_y = +0.5 * B0 * x
    then each link variable is exp[i * (2e/hbar) * integral(A · dl)] ~ exp[i ...].
    """
    # For demonstration, let 2 e / hbar = 1.0
    e2_hbar = 1.0  
    phase_factor = e2_hbar * dx

    x_vals = (np.arange(nx) - 0.5*nx) * dx
    y_vals = (np.arange(ny) - 0.5*ny) * dx

    Ux = np.ones((nx, ny), dtype=np.complex128)
    Uy = np.ones((nx, ny), dtype=np.complex128)

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            Axi = -0.5 * B0 * y   # A_x at (x, y)
            Ayi =  0.5 * B0 * x   # A_y at (x, y)
            phi_x = phase_factor * Axi
            phi_y = phase_factor * Ayi
            Ux[i, j] = np.exp(1j * phi_x)
            Uy[i, j] = np.exp(1j * phi_y)

    return Ux, Uy

###############################################################################
# 2. Covariant Laplacian Operator (Discrete)
###############################################################################

def covariant_laplacian(psi, Ux, Uy, dx):
    """
    Discrete gauge-invariant Laplacian: D^2 psi = D_+x(D_-x(psi)) + D_+y(D_-y(psi)),
    where D_± accounts for link variables Ux, Uy in x- and y-directions.
    """
    nx, ny = psi.shape

    # Shift helpers for periodic BC
    def ip1(i): return (i + 1) % nx
    def im1(i): return (i - 1) % nx
    def jp1(j): return (j + 1) % ny
    def jm1(j): return (j - 1) % ny

    # Forward difference in x
    Dpx_psi = (Ux * np.roll(psi, -1, axis=0) - psi) / dx
    # Backward difference in x
    Ux_im1 = np.roll(Ux, +1, axis=0).conjugate()
    Dmx_psi = (psi - Ux_im1 * np.roll(psi, +1, axis=0)) / dx
    Dpx_Dmx = (Ux * np.roll(Dmx_psi, -1, axis=0) - Dmx_psi) / dx

    # Forward difference in y
    Dpy_psi = (Uy * np.roll(psi, -1, axis=1) - psi) / dx
    # Backward difference in y
    Uy_jm1 = np.roll(Uy, +1, axis=1).conjugate()
    Dmy_psi = (psi - Uy_jm1 * np.roll(psi, +1, axis=1)) / dx
    Dpy_Dmy = (Uy * np.roll(Dmy_psi, -1, axis=1) - Dmy_psi) / dx

    return Dpx_Dmx + Dpy_Dmy

###############################################################################
# 3. TDGL Euler Step
###############################################################################

def tdgl_euler_step(psi, Ux, Uy, alpha, beta, dx, dt, 
                    hbar=1.0, mass=1.0, gamma=0.2):
    """
    One forward-Euler step of:
      dpsi/dt = -gamma [ alpha psi + beta|psi|^2 psi - (hbar^2 / 2m)*D^2 psi ].
    """
    lap_psi = covariant_laplacian(psi, Ux, Uy, dx)
    nonlinear_term = alpha * psi + beta * (np.abs(psi)**2) * psi
    lap_term = -(hbar**2 / (2.0*mass)) * lap_psi

    dpsi_dt = -gamma * (nonlinear_term + lap_term)
    psi_new = psi + dt * dpsi_dt
    return psi_new

###############################################################################
# 4. Initialization
###############################################################################

def initialize_psi(nx, ny, noise_amp=0.1):
    """
    Initialize psi with small random noise for real and imaginary parts.
    """
    real_part = noise_amp*(np.random.rand(nx, ny) - 0.5)
    imag_part = noise_amp*(np.random.rand(nx, ny) - 0.5)
    return real_part + 1j * imag_part

###############################################################################
# 5. Main Simulation + Animation
###############################################################################

def run_tdgl_simulation_anim(nx=64, ny=64, dx=0.5,
                             alpha=-1.0, beta=1.0,
                             B0=1.0,
                             dt=0.05, nsteps=200,
                             output_interval=5,
                             hbar=1.0, mass=1.0, gamma=0.2,
                             noise_amp=0.1):
    """
    Run a 2D time-dependent Ginzburg-Landau simulation with link variables for a uniform field,
    producing an animation of |psi| vs. time.

    Parameters:
      nx, ny: grid dimensions
      dx: spatial step size (assuming dx=dy)
      alpha, beta: GL coefficients
      B0: external uniform field
      dt: time step
      nsteps: total integration steps
      output_interval: number of steps between animation frames
      hbar, mass, gamma: physical or dimensionless parameters
      noise_amp: amplitude for initial random perturbation
    Returns:
      anim: a matplotlib.animation.FuncAnimation object
    """
    # 1) Build link variables
    Ux, Uy = build_uniform_field_links(nx, ny, dx, B0)

    # 2) Initialize psi
    psi = initialize_psi(nx, ny, noise_amp)

    # 3) Setup figure
    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_title("2D TDGL Simulation: |psi|")
    extent = [0, nx*dx, 0, ny*dx]

    # Initial field for |psi|
    im = ax.imshow(np.abs(psi).T, origin='lower', extent=extent,
                   cmap='magma', vmin=0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("|psi|", rotation=90)
    
    # We will create an init function for FuncAnimation (for blitting):
    def init():
        im.set_data(np.abs(psi).T)
        im.set_clim(0, np.max(np.abs(psi)))  # adjust color scale if needed
        return [im]

    # The update function for each frame:
    def update(frame):
        nonlocal psi
        # Do several Euler steps before showing the next frame
        for _ in range(output_interval):
            psi = tdgl_euler_step(psi, Ux, Uy, alpha, beta, dx, dt,
                                  hbar=hbar, mass=mass, gamma=gamma)

        # Update image
        mod_psi = np.abs(psi)
        im.set_data(mod_psi.T)
        im.set_clim(0, np.max(mod_psi))  # Auto-adjust color scale if desired
        return [im]

    # Number of frames in the animation
    n_frames = nsteps // output_interval

    # Create FuncAnimation
    anim = FuncAnimation(fig, update, frames=n_frames,
                         init_func=init, interval=100, blit=True)
    return anim

###############################################################################
# 6. Main
###############################################################################

if __name__ == "__main__":
    # Example parameters
    nx, ny = 200, 200
    dx = 0.5
    alpha = -1.0   # superconducting regime
    beta = 1.0
    B0 = 1.0       # uniform field
    dt = 0.05
    nsteps = 600   # total time steps
    output_interval = 1  # steps between animation frames
    gamma = 0.2
    noise_amp = 0.1

    anim = run_tdgl_simulation_anim(nx, ny, dx,
                                    alpha, beta,
                                    B0, dt, nsteps,
                                    output_interval,
                                    gamma=gamma, noise_amp=noise_amp)

    # 7) Show animation in a window:
    plt.show()

    # If you want to save to an mp4 file (requires ffmpeg installed):
    # anim.save("tdgl_sim.mp4", writer="ffmpeg", fps=10)
```


![image](https://github.com/user-attachments/assets/2ac8795c-ee42-45cb-b1f5-62f5d733f513)
