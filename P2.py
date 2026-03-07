import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
m1 = m2 = 1.0  # Masses
L1 = L2 = 1.0  # Lengths
g = 9.81       # Acceleration due to gravity

# Equations of motion
def equations(t, y):
    theta1, theta2, omega1, omega2 = y
    delta = theta1 - theta2

    denominator1 = (2 * m1 + m2 - m2 * np.cos(2 * delta))
    denominator2 = (L2 / L1) * denominator1

    dydt = np.zeros_like(y)
    dydt[0] = omega1
    dydt[1] = omega2

    dydt[2] = (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 
               2 * np.sin(delta) * m2 * (omega2**2 * L2 + omega1**2 * L1 * np.cos(delta))) / (L1 * denominator1)

    dydt[3] = (2 * np.sin(delta) * (omega1**2 * L1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) +
               omega2**2 * L2 * m2 * np.cos(delta))) / (L2 * denominator1)
    return dydt

# Initial conditions
theta1_0 = np.pi / 2
theta2_0 = np.pi / 2
omega1_0 = 0.0
omega2_0 = 0.0
y0 = [theta1_0, theta2_0, omega1_0, omega2_0]

# Time span
t_span = (0, 20)
t_eval = np.linspace(*t_span, 2000)

# Solve the system
sol = solve_ivp(equations, t_span, y0, t_eval=t_eval)

# Extract solutions
theta1 = sol.y[0]
theta2 = sol.y[1]

# Plot angles as functions of time
plt.figure(figsize=(10, 6))
plt.plot(sol.t, theta1, label=r'$\theta_1$')
plt.plot(sol.t, theta2, label=r'$\theta_2$')
plt.title('Double Pendulum Angles Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)
plt.show()

# Animate the double pendulum
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)

x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2 * (L1 + L2), 2 * (L1 + L2))
ax.set_ylim(-2 * (L1 + L2), 2 * (L1 + L2))
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=2)

def animate(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    return line,

ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=20, blit=True)

plt.title('Double Pendulum Animation')
plt.show()

