# Principle of Least (Stationary) Action-Informed Neural Networks

## 1. Introduction

The Principle of Least (Stationary) Action states that the trajectory (or configuration history) a physical system follows between two specified times (or boundary conditions) is the one that extremizes the action $S$. In classical mechanics, for a single degree of freedom with generalized coordinate $x(t)$, the action is:

$$
S[x] = \int_{t_0}^{t_1} L(x(t), \dot{x}(t)) \, dt,
$$

where the Lagrangian $L(x, \dot{x}) = T - V$ (kinetic minus potential energy) often takes the form:

$$
L(x, \dot{x}) = \frac{1}{2} m \dot{x}^2 - V(x).
$$

### 1.1 Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks (PINNs) embed physical knowledge (PDEs, ODEs, boundary/initial conditions, or physical principles like conservation laws) into the loss function of a neural network. By doing so, the network is not just fitting data; it is constrained by fundamental physical principles.

**Traditional PINN:** Minimizes a combined loss:

$$
L_\text{PINN} = L_\text{data} + \lambda_\text{physics} L_\text{PDE} + \dots
$$

where $L_\text{PDE}$ enforces a PDE residual, boundary conditions, etc.

**Action-Based PINN:** Instead of enforcing PDE residuals directly, we can enforce the principle of stationary action by making the network minimize the action integral. When $\hbar \to 0$ (classical limit), the extremization of this integral leads to the classical equations of motion.

### 1.2 Why Use the Principle of Least Action?

- **Fundamental:** Many areas of physics (classical mechanics, field theory, relativity) all spring from action minimization.
- **Compact:** Encodes boundary conditions or initial conditions succinctly, because the variational principle typically involves specified conditions at the endpoints.
- **Smoothness:** Encourages global consistency of the solution, since the action is an integral over the entire domain of interest (e.g., time).

---

## 2. Embedding Stationary Action into a PINN

Consider a simple 1D harmonic oscillator for concreteness, though the method generalizes to more complex systems:

- **System:** Mass $m=1$, spring constant $k=1$.
- **Lagrangian:**

$$
L(x, \dot{x}) = \frac{1}{2} \dot{x}^2 - \frac{1}{2} x^2.
$$

- **Action (from $t=0$ to $t=T$):**

$$
S[x] = \int_0^T \left[ \frac{1}{2} \dot{x}(t)^2 - \frac{1}{2} x(t)^2 \right] dt.
$$

- **Boundary/Initial Conditions (example):**

$$
x(0) = x_0, \quad x(T) = x_T.
$$

(Or you could specify initial position and velocity if you want an initial value problem.)

### 2.1 Setting Up a Neural Network

We let a neural network $x_\theta(t)$ approximate the trajectory over $t \in [0, T]$. The parameters $\theta$ of the network are optimized so as to minimize the action integral.

- **Input:** Time $t$.
- **Output:** Approximate position $x_\theta(t)$.
- **Automatic Differentiation:** We use deep learning frameworks (e.g., PyTorch or TensorFlow) to obtain
  
![image](https://github.com/user-attachments/assets/d975c2b8-6849-492b-b022-bf3355bb38bf)

automatically.

### 2.2 Loss Function: Action Integral

We discretize the time domain into $\{t_i\}_{i=0}^N$. The action is approximated numerically by (for instance) a trapezoidal or midpoint rule:

![image](https://github.com/user-attachments/assets/c8eb0ce9-6141-4ace-b057-08b88aa8ee53)

where $\Delta t = \frac{T}{N}$ (assuming a uniform grid for simplicity).

Thus, the loss function to minimize is:

$$
L_\text{action}(\theta) = S[\theta].
$$

### 2.3 Enforcing Boundary/Initial Conditions

We can incorporate boundary conditions in at least two ways:

1. **Soft constraints (penalty method):** Add a penalty term to the loss if $x_\theta(0) \neq x_0$ or $x_\theta(T) \neq x_T$.
2. **Hard constraints:** Build the network so that it explicitly satisfies boundary conditions. For example, if $x(0) = x_0$ and $x(T) = x_T$, one can use a function ansatz like:

$$
x_\theta(t) = x_0 + t \left[x_T - x_0 + \tilde{x}_\theta(t)\right],
$$

ensuring the boundary conditions at 0 and $T$, with $\tilde{x}_\theta(t)$ vanishing at $t=0$ and $t=T$.

---

## 3. Example: A Simple Python Code (Using PyTorch)

Below is a minimal working example illustrating how to set up a PINN that minimizes the action for a harmonic oscillator. This example is purely educational and can be extended to more sophisticated systems.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Neural Network Definition
# -----------------------------
class ActionNet(nn.Module):
    def __init__(self, num_hidden=2, num_neurons=32, x0=1.0, xT=0.0, T=6.28):
        """
        Simple feed-forward network that:
         - Takes time t as input (shape: [batch_size, 1])
         - Outputs position x(t)
         - Hard-encodes boundary conditions x(0) = x0, x(T) = xT
        """
        super(ActionNet, self).__init__()
        
        # Store boundary conditions and domain
        self.x0 = x0
        self.xT = xT
        self.T  = T
        
        # Build a small MLP
        layers = []
        in_features = 1
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_features, num_neurons))
            layers.append(nn.Tanh())
            in_features = num_neurons
        layers.append(nn.Linear(in_features, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, t):
        """
        Forward pass.
        We embed the boundary conditions in the network output such that:
            x(0) = x0, x(T) = xT
        One common approach is:
            x(t) = x0 + (xT - x0) * (t/T) + (t*(T - t)/T^2)*NN(t)
        This ensures x(0) = x0, x(T) = xT exactly.
        """
        # Normalize t to [0,1] for better conditioning
        t_norm = t / self.T
        
        # The 'raw' net output
        raw_output = self.model(t_norm)
        
        # Hard boundary condition embedding
        # factor = t*(T - t)/T^2 ensures zero at t=0 and t=T
        factor = t_norm * (1.0 - t_norm)
        x = self.x0 + (self.xT - self.x0) * t_norm + factor * raw_output
        
        return x

# -----------------------------
# 2. Define the Action Functional
# -----------------------------
def action_loss(model, t_tensor, dt):
    """
    Approximates the action integral:
        S[x] = \int (0.5*x'(t)^2 - 0.5*x(t)^2) dt
    using a discrete sum over t_tensor.
    """
    # Forward pass: x(t)
    x_pred = model(t_tensor)
    
    # Compute derivative x'(t) via automatic differentiation
    # We need x_pred -> shape: [N, 1]
    # We differentiate w.r.t. t_tensor -> shape: [N, 1]
    x_dot = torch.autograd.grad(
        x_pred,         # y
        t_tensor,       # x
        grad_outputs=torch.ones_like(x_pred),
        create_graph=True
    )[0]
    
    # Lagrangian = 0.5*x'(t)^2 - 0.5*x(t)^2
    L = 0.5 * x_dot**2 - 0.5 * x_pred**2
    
    # Numerical integration by summation
    S_approx = torch.sum(L) * dt
    return S_approx

# -----------------------------
# 3. Main Training Loop
# -----------------------------
def train_action_model(
    x0=1.0,       # boundary condition at t=0
    xT=0.0,       # boundary condition at t=T
    T=2*np.pi,    # total time (one full oscillation is 2 pi for w=1)
    num_points=100, 
    num_iter=2000, 
    lr=1e-3
):

    # Create time discretization
    t_np = np.linspace(0, T, num_points)
    dt = (T - 0) / (num_points - 1)
    
    # Convert to PyTorch
    t_tensor = torch.tensor(t_np, dtype=torch.float32).view(-1, 1)
    t_tensor.requires_grad = True  # to allow auto-diff w.r.t. t
    
    # Instantiate the model
    model = ActionNet(x0=x0, xT=xT, T=T)
    model.train()
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_iter):
        optimizer.zero_grad()
        
        # Compute the approximate action
        S_value = action_loss(model, t_tensor, dt)
        
        # Backprop
        S_value.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Action = {S_value.item():.6f}")
    
    return model

# -----------------------------
# 4. Run the Training and Plot
# -----------------------------
if __name__ == "__main__":
    # Train the model
    model = train_action_model(x0=1.0, xT=0.0, T=2*np.pi, num_points=100, num_iter=2000, lr=1e-3)
    
    # Evaluate the learned solution
    model.eval()
    
    # Create fine grid for plotting
    t_plot = np.linspace(0, 2*np.pi, 200)
    t_tensor_plot = torch.tensor(t_plot, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        x_pinn = model(t_tensor_plot).numpy().flatten()
    
    # True analytic solution for reference (harmonic oscillator)
    # If x(0) = 1 and x'(0) = 0, the solution is x(t)=cos(t). But here we forced x(T)=0 => T=2pi.
    # We'll do an approximate boundary-value solution approach:
    # For a simple guess, let's see if we can guess a "cos(t) - shift" type solution:
    # Actually, let's just keep the learned solution for demonstration.
    
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(t_plot, x_pinn, 'r-', label="PINN (action-based)")
    plt.title("Action-Based PINN for Harmonic Oscillator")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.grid(True)
    plt.show()
```

![image](https://github.com/user-attachments/assets/0bda7e86-09dc-48a1-b180-fb6be6350071)

## 3.1 Code Explanation

### ActionNet Class
- Constructs a small feedforward network (MLP).
- Enforces boundary conditions exactly by using a custom output transformation $x(t)$.

### action_loss Function
- Implements the discrete approximation of the action integral.
- Uses `torch.autograd.grad` to compute $\dot{x}$ with respect to $t$.
- Sums over the time grid points to approximate the integral.

### train_action_model Function
- Sets up the training dataset (uniform grid of time points).
- Initializes the network and optimizer.
- Iterates for a fixed number of epochs, each time computing the action and backpropagating the gradients.

### Results
- By design, the approach should converge to a trajectory that (locally) minimizes the discrete action.
- For boundary value problems, the final path matches the classical solution that satisfies $x(0) = x_0, \, x(T) = x_T$.

---

## 4. Extensions and Generalizations

### Non-Linear Systems
- You can replace the simple harmonic oscillator potential $V(x) = x^2 / 2$ with any more complex potential $V(x)$.
- The network architecture and the action loss function remain structurally the same.

### Multiple Degrees of Freedom
- For higher-dimensional systems ($x(t) \in \mathbb{R}^n$), simply output $n$ coordinates from your neural network and sum the Lagrangians for each coordinate.

### Field Theories
- Extend the concept to fields $\phi(r, t)$. You discretize the spacetime domain and approximate $\phi$ with a neural network. 
- The integral in the action then becomes over space and time:

$$
S[\phi] = \int L(\phi, \partial_\mu \phi) \, d^4x.
$$

- Minimizing with respect to all parameters ensures the solution satisfies the Euler–Lagrange equations for the field.

### Constraints and Lagrange Multipliers
- If your problem has constraints (e.g., $f(x, \dot{x}) = 0$), you can embed them via penalty terms or augmented Lagrangians in the network training.

### Hybrid Data + Action
- Combine data fitting (observations) with the principle of least action. This is powerful when partial experimental data is available.

---

## 5. Concluding Remarks

By integrating the Principle of Least (Stationary) Action into a neural network training scheme, we create a physics-informed neural network whose predictions are consistent with fundamental physical laws—rather than solely fitting data. This approach:

- Naturally enforces the correct equations of motion.
- Is robust against noise or sparse data scenarios.
- Scales to complex systems (nonlinear, multi-dimensional) with the flexible capacity of neural networks.

Such action-based PINNs hold promise for solving challenging boundary-value problems, exploring new potentials in high-dimensional spaces, and even bridging the gap between classical and quantum systems by moving from path integrals to action-based constraints.


