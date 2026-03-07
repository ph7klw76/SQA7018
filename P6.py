import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) Physical constants
# =========================
hbar = 1.0
m = 1.0
a_s = 0.005
g = 4 * np.pi * hbar**2 * a_s / m

# =========================
# 2) Trap quench parameters
#    Prepare in one trap, evolve in another
# =========================
omega_initial = 1.0
omega_final = 0.7   # quench to a weaker trap -> breathing mode

# =========================
# 3) Spatial grid
# =========================
N = 256
x = np.linspace(-10, 10, N)
y = np.linspace(-10, 10, N)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# =========================
# 4) Initial wavefunction
#    Ground state of the INITIAL trap
# =========================
psi_0 = (m * omega_initial / (np.pi * hbar))**0.5 * np.exp(
    -m * omega_initial * (X**2 + Y**2) / (2 * hbar)
)
psi = psi_0.copy().astype(np.complex128)

# Normalize initial wavefunction
norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
psi /= norm

# Initial density
initial_density = (np.abs(psi)**2).copy()

# =========================
# 5) Quenched harmonic potential
#    This is the trap used during time evolution
# =========================
V = 0.5 * m * omega_final**2 * (X**2 + Y**2)

# =========================
# 6) Time discretization
# =========================
dt = 0.01
t_max = 20.0
N_t = int(t_max / dt)

# =========================
# 7) Fourier-space variables
# =========================
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)
K_squared = KX**2 + KY**2

# Kinetic evolution operator
expK = np.exp(-1j * hbar * K_squared * dt / (2 * m))

# =========================
# 8) Snapshot settings
# =========================
snapshot_interval = 100   # save every 1.0 time unit
snapshots_density = []
snapshots_delta = []
snapshot_times = []

# Store initial snapshot
snapshots_density.append(initial_density.copy())
snapshots_delta.append(np.zeros_like(initial_density))
snapshot_times.append(0.0)

# =========================
# 9) Arrays to track scalar quantities over time
# =========================
times = []
rms_radius = []
peak_density = []

def compute_rms_radius(psi, X, Y, dx, dy):
    density = np.abs(psi)**2
    r2_mean = np.sum((X**2 + Y**2) * density) * dx * dy
    return np.sqrt(r2_mean)

def compute_peak_density(psi):
    return np.max(np.abs(psi)**2)

# Store initial scalar values
times.append(0.0)
rms_radius.append(compute_rms_radius(psi, X, Y, dx, dy))
peak_density.append(compute_peak_density(psi))

# =========================
# 10) Time evolution
#     Split-step Fourier method
# =========================
for time_idx in range(1, N_t + 1):
    # Half-step: potential + nonlinear term in real space
    psi *= np.exp(-1j * (V + g * np.abs(psi)**2) * dt / (2 * hbar))

    # Full-step: kinetic term in Fourier space
    psi_k = np.fft.fft2(psi)
    psi_k *= expK
    psi = np.fft.ifft2(psi_k)

    # Half-step: potential + nonlinear term in real space
    psi *= np.exp(-1j * (V + g * np.abs(psi)**2) * dt / (2 * hbar))

    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    psi /= norm

    # Current time
    current_time = time_idx * dt

    # Track scalar quantities
    times.append(current_time)
    rms_radius.append(compute_rms_radius(psi, X, Y, dx, dy))
    peak_density.append(compute_peak_density(psi))

    # Save snapshots
    if time_idx % snapshot_interval == 0 or time_idx == N_t:
        density = (np.abs(psi)**2).copy()
        delta_density = density - initial_density

        snapshots_density.append(density)
        snapshots_delta.append(delta_density)
        snapshot_times.append(current_time)

# =========================
# 11) Plot density change snapshots
#     Δn(x,y,t) = n(x,y,t) - n(x,y,0)
# =========================
num_snapshots = len(snapshots_delta)
ncols = 3
nrows = int(np.ceil(num_snapshots / ncols))

# Common symmetric color scale for all delta-density plots
delta_max = max(np.max(np.abs(delta)) for delta in snapshots_delta)

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.array(axes).reshape(-1)

for i, ax in enumerate(axes):
    if i < num_snapshots:
        contour = ax.contourf(
            X, Y, snapshots_delta[i],
            levels=50,
            cmap='seismic',
            vmin=-delta_max,
            vmax=delta_max
        )
        ax.set_title(f"ΔDensity at t = {snapshot_times[i]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(contour, ax=ax)
    else:
        ax.axis("off")

plt.suptitle("Density Change During Breathing Mode", fontsize=16)
plt.tight_layout()
plt.show()

# =========================
# 12) Plot actual density snapshots
#     Useful alongside delta-density plots
# =========================
density_min = min(np.min(d) for d in snapshots_density)
density_max = max(np.max(d) for d in snapshots_density)

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.array(axes).reshape(-1)

for i, ax in enumerate(axes):
    if i < num_snapshots:
        contour = ax.contourf(
            X, Y, snapshots_density[i],
            levels=50,
            cmap='viridis',
            vmin=density_min,
            vmax=density_max
        )
        ax.set_title(f"Density at t = {snapshot_times[i]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(contour, ax=ax)
    else:
        ax.axis("off")

plt.suptitle("Condensate Density During Breathing Mode", fontsize=16)
plt.tight_layout()
plt.show()

# =========================
# 13) Plot scalar quantity over time
#     RMS radius is the cleanest breathing-mode observable
# =========================
plt.figure(figsize=(8, 5))
plt.plot(times, rms_radius, linewidth=2)
plt.xlabel("Time")
plt.ylabel(r"$R_{\mathrm{rms}} = \sqrt{\langle x^2 + y^2 \rangle}$")
plt.title("Breathing Mode: RMS Radius vs Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 14) Optional extra scalar:
#     Peak density vs time
# =========================
plt.figure(figsize=(8, 5))
plt.plot(times, peak_density, linewidth=2)
plt.xlabel("Time")
plt.ylabel("Peak Density")
plt.title("Peak Density vs Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 15) Final density and final density change
# =========================
final_density = np.abs(psi)**2
final_delta = final_density - initial_density

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, final_density, levels=50, cmap='viridis')
plt.colorbar(label='Density')
plt.title('Final Condensate Density')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, final_delta, levels=50, cmap='seismic')
plt.colorbar(label='Density Change')
plt.title('Final Density Change: n(x,y,t) - n(x,y,0)')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
