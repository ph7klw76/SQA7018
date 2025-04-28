# Real-World Problem Scenario

You’re operating a drone-based delivery system. A small package must be launched from ground level ($y = 0$) and land exactly on a platform 200 m away (also at $y = 0$). The drone always gives the package an initial speed of 70 m/s, but the package experiences quadratic air drag (i.e., drag $\propto v^2$).


![image](https://github.com/user-attachments/assets/7879aa88-6235-495d-9ea0-b1679471eb2d)

---

## Physics Background

### No Air Resistance

- Closed-form range:

$$
R(\theta) = \frac{v_0^2}{g} \sin(2\theta)
$$

You’d solve:

$$
\sin(2\theta) = \frac{gR}{v_0^2}
$$

for $\theta$ analytically.

---

### With Quadratic Drag

$$
\begin{cases}
m \frac{dv_x}{dt} = -b \sqrt{v_x^2 + v_y^2} \, v_x \\
m \frac{dv_y}{dt} = -mg - b \sqrt{v_x^2 + v_y^2} \, v_y
\end{cases}
$$

There is no closed-form solution for $R(\theta)$ → you must numerically integrate the trajectory. Given m=0.5 kg and b=0.001 kg/m

---

## Why Newton–Raphson?

Define:

$$
f(\theta) = R_{\text{num}}(\theta) - 200\ \text{m}
$$

We want $\theta$ such that $f(\theta) = 0$.

Newton–Raphson iterates:

$$
\theta_{n+1} = \theta_n - \frac{f(\theta_n)}{f'(\theta_n)}
$$

Since $f'(\theta)$ isn’t analytic, we approximate it by finite differences:

$$
f'(\theta) \approx \frac{f(\theta + h) - f(\theta - h)}{2h}
$$

where $h$ is a small step size.

---

## Improved Task Description

### Trajectory Simulation

Write a routine that, given an angle $\theta$, integrates the above ODEs (e.g., via Runge–Kutta) to compute the horizontal range $R_{\text{num}}(\theta)$.

---

### Root Finding

Implement Newton–Raphson to solve $f(\theta) = 0$.  
Use a finite‐difference step $h$ (you choose a suitable value) to approximate $f'(\theta)$.

---

## Deliverables

- The launch angle $\theta$ (in degrees and radians) that yields exactly 200 meters.
- Your choice of $h$, stopping criterion (e.g., $|\Delta\theta|$ or $|f(\theta)|$ tolerance), and maximum iterations.
- A brief discussion of convergence behavior and numerical robustness.

---

## Bonus

Compare your result to the drag-free solution and comment on the impact of air resistance.


# Solution

Imagine you are designing a drone-based delivery system. Your task: launch a package from the ground (initial  
$y=0$) and have it land exactly 200 meters away (also at $y=0$).

Easy, right? Well, not quite — the package suffers quadratic air drag (forces proportional to $v^2$), making the problem much trickier than a simple physics textbook projectile.

You know:

- Initial speed: $v_0 = 70\ \text{m/s}$
- Mass: $m = 0.5\ \text{kg}$
- Drag coefficient: $b = 0.001\ \text{kg/m}$
- Gravity: $g = 9.81\ \text{m/s}^2$

You need to find the launch angle $\theta$ that lets the package reach exactly 200 meters.

Since air drag complicates things (no clean formulas!), we'll combine numerical integration with the Newton–Raphson method to find the answer.

---

## Step 1: Simulate the Flight with Euler's Method

First, to compute how far the package flies for a given $\theta$, we use Euler integration. It's a simple (first-order) method that updates the package's position and velocity over tiny time steps $\Delta t$.

Here's the physics:

- Drag force:  
  $$F_{\text{drag}} = b v^2, \quad \text{with} \quad v = \sqrt{v_x^2 + v_y^2}$$

- Horizontal acceleration:  
  $$a_x = -k v v_x$$

- Vertical acceleration:  
  $$a_y = -g - k v v_y$$

where:

$$k = \frac{b}{m}$$

Each step updates:

$$
\begin{aligned}
x_{\text{new}} &= x + v_x \Delta t \\
y_{\text{new}} &= y + v_y \Delta t \\
v_{x,\text{new}} &= v_x + a_x \Delta t \\
v_{y,\text{new}} &= v_y + a_y \Delta t
\end{aligned}
$$

We simulate until the package crosses back to $y=0$ (landing).

---

## Step 2: Define the Root-Finding Problem

We define:

$$f(\theta) = R(\theta) - 200$$

where $R(\theta)$ is the simulated horizontal range for launch angle $\theta$.

We want:

$$f(\theta) = 0$$

meaning the projectile lands exactly 200 meters away.

---

## Step 3: Use Newton–Raphson Iteration

Newton–Raphson is a powerful method for finding roots. It updates guesses using:

$$
\theta_{n+1} = \theta_n - \frac{f(\theta_n)}{f'(\theta_n)}
$$

However, $f(\theta)$ isn't a simple formula, so we approximate the derivative $f'(\theta)$ by finite differences:

$$
f'(\theta) \approx \frac{f(\theta + h) - f(\theta - h)}{2h}
$$

where $h$ is a small step (e.g., $10^{-4}$).

---

## Step 4: Find All Launch Angles

Because quadratic drag creates multiple possible launch angles (one low, one high), we:

- Scan a coarse grid of angles between $0^\circ$ and $90^\circ$,
- Look for sign changes in $f(\theta)$ (root brackets),
- Use Newton–Raphson starting near each sign change to converge to the correct launch angle.

---

# Full Code Implementation

Here's a clean version of the code:

```python
import numpy as np

# Parameters
g  = 9.81                 # m/s²
v0 = 70.0                 # m/s
m  = 0.50                 # kg
b  = 0.001                # kg/m
k  = b / m
X_TARGET = 200.0          # m

# Euler integration step
def euler_step(x, y, vx, vy, dt):
    v  = np.hypot(vx, vy)
    ax = -k * v * vx
    ay = -g - k * v * vy
    return (
        x  + vx * dt,
        y  + vy * dt,
        vx + ax * dt,
        vy + ay * dt,
    )

# Range for given launch angle
def range_for_angle(theta, dt=2e-4):
    x, y = 0.0, 0.0
    vx, vy = v0 * np.cos(theta), v0 * np.sin(theta)

    while True:
        x_new, y_new, vx, vy = euler_step(x, y, vx, vy, dt)
        if y_new <= 0.0 and y > 0.0:
            frac = y / (y - y_new)
            return x + frac * (x_new - x)
        x, y = x_new, y_new

# Root function
def f(theta):
    return range_for_angle(theta) - X_TARGET

def fprime(theta, h=1e-4):
    return (range_for_angle(theta + h) - range_for_angle(theta - h)) / (2*h)

# Newton-Raphson solver
def newton(theta0, tol=1e-3, max_iter=25):
    theta = theta0
    for _ in range(max_iter):
        fn = f(theta)
        if abs(fn) < tol:
            return theta
        theta -= fn / fprime(theta)
    raise RuntimeError("Newton did not converge")

# Bracket and refine roots
grid = np.linspace(0.02, np.pi/2 - 0.02, 400)
vals = [f(th) for th in grid]

roots = []
for th1, th2, v1, v2 in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
    if v1 * v2 < 0:
        theta_mid = 0.5 * (th1 + th2)
        roots.append(newton(theta_mid))

roots.sort()

# Output
print("\nLaunch angles (Euler dt = 2×10⁻⁴ s)\n")
for i, th in enumerate(roots, 1):
    print(f"θ{i}: {th: .8f} rad  =  {np.degrees(th): .5f}°   "
          f"(range = {range_for_angle(th):.3f} m)")
```

## Step 5: Interpretation and Realism

- **Two solutions**: Generally, one low-angle fast trajectory and one high-angle slow trajectory.

- **Convergence**: Newton–Raphson is extremely fast — usually 3–5 iterations.

- **Robustness**: A tiny step size $h \sim 10^{-4}$ and a time step $\Delta t = 2 \times 10^{-4}$ ensure good numerical accuracy.

- **Realistic modeling**: Drag at high speeds is significant. Ignoring it would lead to a major overshoot.

---
![image](https://github.com/user-attachments/assets/0e44d564-5ef1-49b2-bf3d-73b6d485594b)

# Final Thoughts

This problem beautifully blends classical mechanics, numerical methods, and algorithmic thinking. In real-world drone or rocket launches, quadratic drag (and even more complex aerodynamics) must be considered carefully.

Here, using Newton–Raphson with finite differences and Euler integration provides a simple yet highly effective solution.

