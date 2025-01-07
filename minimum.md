
# Principle of Least (Stationary) Action

How the Principle of Least Action (or, more precisely, the Principle of Stationary Action) provides a conceptual and mathematical bridge to quantum mechanics. We will explore:

![image](https://github.com/user-attachments/assets/1d7d96b5-ee6e-4f8b-b356-4432ca0d946f)
![image](https://github.com/user-attachments/assets/cc4af8ee-d5ef-48aa-b428-bc10af0de3b0)
![image](https://github.com/user-attachments/assets/42c40a18-e6e8-435b-82af-bb85767e5085)
![image](https://github.com/user-attachments/assets/cf7dba65-55a7-4526-b8e3-ef96eb634677)
![image](https://github.com/user-attachments/assets/5692803b-0c60-4620-b553-28777e17b2fa)
![image](https://github.com/user-attachments/assets/748a417c-365a-42de-a9a0-1d36d9b61017)
![image](https://github.com/user-attachments/assets/a56d29a0-3982-40f8-a058-00fe18bbc2be)
![image](https://github.com/user-attachments/assets/ab95f473-b2e7-4b99-b276-f315db1dfb0a)
![image](https://github.com/user-attachments/assets/9667498c-e614-4b07-b7b4-42f3394b49a5)
![image](https://github.com/user-attachments/assets/c87df723-ac81-4f2c-90f8-999208b12e1d)
![image](https://github.com/user-attachments/assets/3de4d774-cf86-4f08-a81c-41b6ae6832d7)
![image](https://github.com/user-attachments/assets/76ef91be-50d6-43eb-9413-8ddf14a1d8bb)
![image](https://github.com/user-attachments/assets/823b0c12-f7d3-45a3-a39b-5e3357f713f2)
![image](https://github.com/user-attachments/assets/f4bc66b9-edbc-4936-b3cc-3102f76a8a36)
![image](https://github.com/user-attachments/assets/2fdbf3aa-8b1f-40fe-9d76-1a4695e3be7a)
![image](https://github.com/user-attachments/assets/11ed29ed-dbda-4c23-8b58-42f90a863bbe)
![image](https://github.com/user-attachments/assets/19643d2b-40e2-4330-8cf2-a52f8eefa8f3)
![image](https://github.com/user-attachments/assets/3967f8f0-30ea-4d7e-84fa-979936ea1f35)
![image](https://github.com/user-attachments/assets/8f00de74-cfe9-4027-81c6-347182c10d48)
![image](https://github.com/user-attachments/assets/9c3ace3d-9983-4e06-9966-680ed9a14a31)
![image](https://github.com/user-attachments/assets/b22e2730-893d-4285-9179-c2f763403a1a)
![image](https://github.com/user-attachments/assets/936078c5-89bd-4478-bd6a-d7b9a9357f25)
![image](https://github.com/user-attachments/assets/be47941c-2750-40a0-9895-57ec08098c6e)
![image](https://github.com/user-attachments/assets/4fd56dc5-d362-4a96-9bcc-92f44dd182ca)
![image](https://github.com/user-attachments/assets/7788e325-94bc-4bee-b53d-00060b8cd719)
![image](https://github.com/user-attachments/assets/b7c2ea85-9094-4a71-b58f-38c55403e398)
![image](https://github.com/user-attachments/assets/0d09cb20-2bce-4f21-9b78-0e88170335d3)
![image](https://github.com/user-attachments/assets/c548b9db-6868-4eb3-aebd-8ba1b7ecf115)
![image](https://github.com/user-attachments/assets/c435437d-c6bd-4abd-bde0-82f72f3ceb65)
![image](https://github.com/user-attachments/assets/c6ae570c-9742-49e9-9613-30da1fddbe90)
![image](https://github.com/user-attachments/assets/714f8558-0d25-4d05-911f-e60c3cf16078)
![image](https://github.com/user-attachments/assets/74c8d2be-d395-4430-9a3f-9648ec7ae5b9)
![image](https://github.com/user-attachments/assets/ce91394d-f91d-484c-aed3-73c69ed8427d)
![image](https://github.com/user-attachments/assets/1125f46a-a4a3-4461-88e9-b2d5ff83e68e)
![image](https://github.com/user-attachments/assets/f69c7df4-434d-4740-a704-32261ec81b12)
![image](https://github.com/user-attachments/assets/415fea2d-bc83-451f-a1b4-6460982b2dd3)
![image](https://github.com/user-attachments/assets/6dea8ffe-bce6-43bc-8a6b-17c6ea0a8853)
![image](https://github.com/user-attachments/assets/6df900f5-addc-4751-8308-b73d42ab3713)
```python
"""
Animate the 1D Free-Particle Quantum Evolution via a (Discrete) Path-Integral Approach.

We simulate a Gaussian wave packet using the principle of stationary action
generalized to quantum mechanics (path integral). Each short time step
is propagated by the free-particle kernel K(x_f, x_i; dt).


Requirements:
    - numpy
    - matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def free_propagator_matrix(x_grid, dt, hbar, mass):
    """
    Build the discrete propagator matrix K_{f,i} for a free particle
    from x_i to x_f over a time step dt, in 1D.

    K(x_f, x_i; dt) = sqrt(m / (2 pi i hbar dt)) * exp[i m (x_f - x_i)^2 / (2 hbar dt)]

    We incorporate 'dx' into K so that the matrix-vector product
    automatically approximates the integral:
        psi_new(x_f) = sum_{x_i} K(x_f, x_i; dt) * psi_old(x_i).

    Parameters
    ----------
    x_grid : 1D numpy array
        Grid of x-coordinates.
    dt : float
        Time step.
    hbar : float
        Reduced Planck's constant.
    mass : float
        Particle mass.

    Returns
    -------
    K : 2D numpy array (complex)
        Propagator matrix of shape (Nx, Nx).
    """
    Nx = len(x_grid)
    dx = x_grid[1] - x_grid[0]  # assume uniform spacing
    # Constant prefactor (with dx included, so the matrix multiplication is a direct sum).
    prefactor = np.sqrt(mass / (2.0j * np.pi * hbar * dt)) * dx

    # Construct the full Nx x Nx matrix
    Xf = x_grid.reshape(-1, 1)   # column
    Xi = x_grid.reshape(1, -1)   # row
    # (x_f - x_i)^2
    diff_sq = (Xf - Xi)**2

    # Phase factor = exp[i m (x_f - x_i)^2 / (2 hbar dt)]
    phase_factor = np.exp(1j * mass * diff_sq / (2.0 * hbar * dt))

    # Combine
    K = prefactor * phase_factor
    return K

def initial_gaussian_packet(x_grid, x0=0.0, p0=0.0, sigma=0.5, hbar=1.0):
    """
    Create an initial Gaussian wave packet:
        psi(x, 0) ~ exp[-(x - x0)^2 / (4 sigma^2)] * exp(i p0 x / hbar)

    Not normalized by default; we'll normalize later.

    Parameters
    ----------
    x_grid : 1D numpy array
        Grid of x-coordinates.
    x0 : float
        Initial center of the wave packet.
    p0 : float
        Initial momentum.
    sigma : float
        Width parameter of the Gaussian.
    hbar : float
        Reduced Planck's constant.

    Returns
    -------
    psi : 1D numpy array (complex)
        The (unnormalized) initial wavefunction values on x_grid.
    """
    x_rel = x_grid - x0
    # Gaussian part
    gauss = np.exp(- (x_rel**2) / (4.0 * sigma**2))
    # Plane-wave factor for momentum p0
    plane = np.exp(1j * p0 * x_grid / hbar)
    psi = gauss * plane
    return psi

def normalize_wavefunction(psi, x_grid):
    """
    Normalize the wavefunction so that sum(|psi|^2) * dx = 1.
    """
    dx = x_grid[1] - x_grid[0]
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm

def run_simulation():
    """
    Runs the path-integral-like free-particle simulation and returns:
    - x_grid
    - prob_snapshots (list of |psi|^2 arrays at each timestep)
    - dt, so we know the time increments
    """
    # --- Physical Parameters ---
    hbar = 1.0       # Planck's constant / 2pi
    mass = 1.0       # mass of the particle
    x_min, x_max = -10.0, 10.0
    Nx = 500         # number of spatial points
    dt = 0.2        # time step
    N_time_steps = 100  # total steps to evolve
    x0 = -3.0        # initial center of wave packet
    p0 = 5.0         # initial momentum
    sigma = 0.5      # initial width of wave packet

    # --- Prepare the spatial grid ---
    x_grid = np.linspace(x_min, x_max, Nx)

    # --- Build the free propagator matrix K ---
    K = free_propagator_matrix(x_grid, dt, hbar, mass)

    # --- Initialize the wavefunction psi(x,0) ---
    psi = initial_gaussian_packet(x_grid, x0, p0, sigma, hbar)
    psi = normalize_wavefunction(psi, x_grid)

    # We will store wavefunction snapshots at each time step for animation
    psi_snapshots = [psi.copy()]

    # --- Time evolution loop ---
    for step in range(N_time_steps):
        psi = K @ psi  # matrix multiplication for the new wavefunction
        # Re-normalize to avoid numeric drift
        psi = normalize_wavefunction(psi, x_grid)
        psi_snapshots.append(psi.copy())

    # Convert wavefunction snapshots into probability distributions
    prob_snapshots = [np.abs(psi_t)**2 for psi_t in psi_snapshots]

    return x_grid, prob_snapshots, dt

def main():
    # Run the simulation first
    x_grid, prob_snapshots, dt = run_simulation()
    x_min, x_max = x_grid[0], x_grid[-1]

    # --- Create an animation ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(x_min, x_max)

    # For aesthetics, let's guess a y-limit by looking at the maximum of the initial distribution
    # multiplied by a small factor to accommodate spreading
    max_prob_initial = np.max(prob_snapshots[0])
    ax.set_ylim(0, max_prob_initial * 1.2)

    # Plot an empty line initially; we'll update it in the animation
    line, = ax.plot([], [], 'b-', lw=2)

    def init():
        """Initialize the background of the animation."""
        line.set_data([], [])
        return (line,)

    def update(frame):
        """Update the line plot for frame index."""
        prob = prob_snapshots[frame]
        line.set_data(x_grid, prob)
        ax.set_title(f"Time = {frame*dt:.2f}")
        return (line,)

    # Create the animation and store reference in a variable 'anim'
    anim = animation.FuncAnimation(
        fig, update,
        frames=len(prob_snapshots),
        init_func=init,
        blit=True,
        interval=100  # in milliseconds, adjust speed of animation
    )

    # Save the animation as a file (optional)
    anim.save("free_particle_simulation.gif", writer="imagemagick", fps=30)

    # Show the animation
    plt.show()

if __name__ == "__main__":
    main()
```
![image](https://github.com/user-attachments/assets/e9864316-03f1-4ae4-bc35-055771589143)
![image](https://github.com/user-attachments/assets/3c9d5dd0-570e-440f-956f-6c8f5e36692d)





































