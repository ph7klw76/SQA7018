import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ---- 1. Parameters ----
L      = 1.0           # m
k      = 100.0         # W/m-K
N      = 10            # number of CVs
T_left = 100.0         # °C (Dirichlet)
T_right = 200.0        # °C (Dirichlet)
# Heat source as a *vectorised* function so S(x_nodes) returns an array
S      = lambda x: np.zeros_like(x)   # W/m^3 (zero everywhere)

# ---- 2. Mesh ----
dx = L / N
x_nodes = (np.arange(N) + 0.5) * dx  # cell centres

# ---- 3. Coefficient arrays ----
aW = np.full(N, k/dx)
aE = np.full(N, k/dx)
aP = aW + aE
b  = -S(x_nodes) * dx                # vector RHS (size N)

# ---- 4. Boundary modifications ----
# West boundary (i = 0) -> T_left is known, move to RHS
aW[0] = 0.0
b[0] += (k/dx) * T_left
# East boundary (i = N-1) -> T_right is known
aE[-1] = 0.0
b[-1] += (k/dx) * T_right

# ---- 5. Assemble tri‑diagonal matrix ----
A = diags(diagonals=[-aW[1:], aP, -aE[:-1]], offsets=[-1, 0, 1], format='csr')

# ---- 6. Solve ----
T = spsolve(A, b)

# ---- 7. Verification & plot ----
x_exact = np.linspace(0.0, L, 100)
T_exact = T_left + (T_right - T_left) * x_exact / L

plt.plot(x_nodes, T, marker='o', label='FVM')
plt.plot(x_exact, T_exact, linestyle='--', label='Analytic')
plt.xlabel('x [m]')
plt.ylabel('Temperature [°C]')
plt.legend()
plt.show()
