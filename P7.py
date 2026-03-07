import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100
density = 0.8
L = np.sqrt(N / density)  # Box size
dt = 0.005
num_steps = 5000
epsilon = 1.0
sigma = 1.0
mass = 1.0

# Initialize positions and velocities
positions = np.random.rand(N, 2) * L
velocities = np.random.randn(N, 2)
velocities -= velocities.mean(axis=0)  # Zero total momentum

# Periodic boundary conditions
def apply_pbc(r):
    return r - L * np.round(r / L)

# Compute forces
def compute_forces(positions):
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    virial = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r_vec = apply_pbc(r_vec)
            r = np.linalg.norm(r_vec)
            if r < 3 * sigma:
                r6 = (sigma / r)**6
                r12 = r6 * r6
                force_scalar = 48 * epsilon * (r12 - 0.5 * r6) / r**2
                force_vec = force_scalar * r_vec
                forces[i] += force_vec
                forces[j] -= force_vec
                potential_energy += 4 * epsilon * (r12 - r6)
                virial += np.dot(force_vec, r_vec)
    return forces, potential_energy, virial

# Equilibration
kinetic_energy_list = []
potential_energy_list = []
temperature_list = []
pressure_list = []

forces, potential_energy, virial = compute_forces(positions)
for step in range(num_steps):
    # Verlet integration
    positions += velocities * dt + 0.5 * forces / mass * dt**2
    positions = positions % L  # Apply periodic boundaries

    forces_new, potential_energy, virial = compute_forces(positions)
    velocities += 0.5 * (forces + forces_new) / mass * dt
    forces = forces_new

    # Compute energies
    kinetic_energy = 0.5 * mass * np.sum(velocities**2)
    total_energy = kinetic_energy + potential_energy

    # Temperature and pressure
    temperature = (2 * kinetic_energy) / (N * 2)
    pressure = density * temperature + virial / (2 * L**2)

    # Store data
    kinetic_energy_list.append(kinetic_energy)
    potential_energy_list.append(potential_energy)
    temperature_list.append(temperature)
    pressure_list.append(pressure)

    # Optional: Print progress
    # if step % 1000 == 0:
    #     print(f"Step {step}/{num_steps}")

# Compute radial distribution function g(r)
def compute_rdf(positions, bins, dr):
    rdf = np.zeros(bins)
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r_vec = apply_pbc(r_vec)
            r = np.linalg.norm(r_vec)
            bin_index = int(r / dr)
            if bin_index < bins:
                rdf[bin_index] += 2
    # Normalize
    density = N / L**2
    shell_volume = 2 * np.pi * np.arange(1, bins + 1) * dr * dr
    ideal_gas = density * shell_volume
    rdf /= N * ideal_gas
    return rdf

bins = 100
dr = (L / 2) / bins
rdf = compute_rdf(positions, bins, dr)
r_values = dr * np.arange(1, bins + 1)

# Plot radial distribution function
plt.figure(figsize=(8, 6))
plt.plot(r_values, rdf)
plt.title('Radial Distribution Function g(r)')
plt.xlabel('Distance r')
plt.ylabel('g(r)')
plt.grid(True)
plt.show()

# Plot temperature and pressure over time
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(temperature_list)
plt.title('Temperature Over Time')
plt.xlabel('Time Step')
plt.ylabel('Temperature')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(pressure_list)
plt.title('Pressure Over Time')
plt.xlabel('Time Step')
plt.ylabel('Pressure')
plt.grid(True)

plt.tight_layout()
plt.show()

