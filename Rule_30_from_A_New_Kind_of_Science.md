# Exploring Cellular Automaton via Rule 30 from "A New Kind of Science"

Stephen Wolfram’s A New Kind of Science presents a thought-provoking study into simple computational rules that lead to complex behaviors. One key aspect of his exploration is the concept of one-dimensional cellular automata. Cellular automata (CA) are computational systems that evolve over discrete time steps, driven by simple, local rules. Despite their simplicity, these systems often exhibit surprisingly rich and complex behaviors.

Among the many rules described by Wolfram, Rule 30 has gained attention for its ability to generate intricate, chaotic patterns from simple initial conditions. In this technical blog, we'll explore what Rule 30 is, how it works, and why it matters in both mathematical theory and practical applications.

## What is a Cellular Automaton?
A cellular automaton (CA) is a discrete computational system made up of an array (or grid) of cells, each of which evolves in "generations" according to a specific set of rules. The concept of cellular automata, introduced by mathematician John von Neumann in the mid-20th century, offers a powerful model for simulating complex, emergent behavior from relatively simple initial conditions. Cellular automata can take different forms, such as one-dimensional (a line of cells), two-dimensional (like Conway’s Game of Life), or higher-dimensional systems.

One-Dimensional Cellular Automaton
In a one-dimensional CA, cells are arranged in a line, and each cell can exist in one of two possible states: 'on' (represented as 1) or 'off' (represented as 0). The state of each cell evolves based on its current state and the states of its immediate neighbors. This evolution happens at discrete time steps, creating a "generation" each time the cells update.

Examples of One-Dimensional Cellular Automata
### Example 1: Simple Rule with Two States
Consider a row of cells with two possible states: 0 (off) and 1 (on). Each cell’s state in the next generation depends on its state and the states of its left and right neighbors. Suppose the rule states:

If a cell and its neighbors are all 1, the cell will become 0 (off) in the next generation.
Otherwise, the cell remains 1 (on).
This rule would produce a pattern that resembles flipping specific cells off while keeping others on, resulting in a repetitive pattern.

### Example 2: Initial Conditions and Rule Application
Suppose we start with a simple initial row:
Initial state: 0001000 (only one cell is 'on')

Applying a rule, say Rule 90 (a commonly studied CA), would result in the following evolution over generations:

Generation 1: 0011100

Generation 2: 0110011

Generation 3: 1100001

Here, Rule 90 is defined such that a cell's new state is the XOR (exclusive OR) of its left and right neighbors. This rule creates interesting symmetrical and fractal-like patterns.

Combinatorial Nature of Rules
For each cell in a one-dimensional CA, its next state depends on the states of itself and its two nearest neighbors (left and right). There are 8 possible combinations for these three cells (2³ = 8). Therefore, the number of possible rules that can be defined for a one-dimensional CA is 2⁸ = 256. Each rule specifies the output state for all 8 combinations of parent states, forming a comprehensive rule set.

For example, the possible combinations of three cells (left-center-right) and their binary representations are:

111, 110, 101, 100, 011, 010, 001, 000

Any rule will specify whether each combination results in an 'on' (1) or 'off' (0) state in the next generation. Rule numbers range from 0 to 255, corresponding to different binary outputs.

## The Mechanics of Rule 30
Among the 256 possible rules for one-dimensional cellular automata, Rule 30 is of particular interest because it produces complex, chaotic patterns despite its simplicity. Let's dive deeper into the mechanics of how Rule 30 operates and how it maps cell states.

The Rule Specification
Rule 30 is defined by the following outputs for each of the 8 possible parent combinations:
Parent Combination    → New State
111                   → 0 (off)
110                   → 0 (off)
101                   → 0 (off)
100                   → 1 (on)
011                   → 1 (on)
010                   → 1 (on)
001                   → 1 (on)
000                   → 0 (off)
These outputs can be expressed as a binary number: 00011110, which corresponds to the decimal number 30. This binary representation gives Rule 30 its name.

Step-by-Step Evolution Example
Initial State
Assume a row of 80 cells where all cells are initially 'off' (0) except for a single 'on' (1) cell in the center. This forms the initial state:

Generation 0: ................................................*.................................................

(Here, . represents an 'off' cell for easier visualization, and * represents an 'on' cell.)

Applying Rule 30
For each cell, we look at the state of the cell and its two nearest neighbors to determine the cell’s state in the next generation. Let’s walk through how the states change:

The initial 'on' cell in the middle (surrounded by two 'off' cells) corresponds to the parent combination 000, which maps to 0 (off) under Rule 30.
However, the cells adjacent to the initial 'on' cell will have parent combinations 010 and 001, which map to 1 (on).
After applying Rule 30, we get:

Generation 1: ..............................................***................................................

Continuing the Evolution
As we continue applying Rule 30, the pattern evolves as follows:

Generation 2: ...............................................................................................

Generation 3: ............................................*****.............................................

Generation 4: ..............................................................................................

This process repeats, and the pattern grows outward, creating a structure that appears chaotic and unpredictable, yet deterministic.

## Properties of Rule 30
Chaotic Behavior
Rule 30 is known for generating complex, seemingly random patterns from simple starting conditions. This chaotic behavior has made it a useful example in studies of randomness and complexity.

Deterministic Nature
Despite its chaotic output, Rule 30 is entirely deterministic. Given the same initial state, it will always produce the same pattern, making it a prime example of how simple rules can yield complex emergent behavior.

Applications of Rule 30
Random Number Generation
Rule 30’s chaotic behavior has practical applications in generating pseudo-random numbers. It has been used in certain cryptographic applications and as part of random number generation algorithms.

Mathematical Modeling and Complexity Theory
Rule 30 has been studied extensively in the context of complexity theory. It demonstrates how complex patterns can arise from simple, deterministic rules, offering insights into emergent complexity and chaotic systems.

Simulation of Natural Phenomena
Cellular automata like Rule 30 can be used to simulate certain natural phenomena and processes, such as fluid dynamics, traffic flow, and growth patterns in biology. The simple, rule-based approach allows for exploration of complex systems using minimal computational overhead.

Education and Research
Rule 30 serves as an excellent teaching tool for exploring topics such as deterministic chaos, emergent complexity, and computational theory. It is often used in educational settings to illustrate fundamental concepts in computer science and mathematics.

## Implementation of Rule 30
Here’s a Python program to generate and display the first few rows of cells governed by Rule 30, starting with a single ‘on’ cell in the center:

```python
def generate_rule30(num_generations=20, row_width=80):
    # Create initial row with a single ‘on’ cell in the center
    current_row = [' ' for _ in range(row_width)]
    current_row[row_width // 2] = '*'  # Middle cell ‘on’

    # Rule 30 mapping based on 3-cell parent states (left, center, right)
    rule30 = {
        '***': ' ',  # 111 → 0
        '** ': ' ',  # 110 → 0
        '* *': ' ',  # 101 → 0
        '*  ': '*',  # 100 → 1
        ' **': '*',  # 011 → 1
        ' * ': '*',  # 010 → 1
        '  *': '*',  # 001 → 1
        '   ': ' '   # 000 → 0
    }

    # Display initial row
    print(''.join(current_row))

    # Generate subsequent generations
    for _ in range(num_generations - 1):
        new_row = [' ' for _ in range(row_width)]
        for i in range(1, row_width - 1):
            # Get states of left, center, and right cells
            parent_state = ''.join(current_row[i - 1:i + 2])
            # Determine new state using Rule 30
            new_row[i] = rule30[parent_state]
        # Update current row and print it
        current_row = new_row
        print(''.join(current_row))

# Call the function to display rows generated by Rule 30
generate_rule30()
```
## Explanation of the Program
The initial state consists of a row of 80 cells with a single ‘on’ cell (represented by *) in the middle.
The rule30 dictionary maps each possible combination of parent states to the new state of the cell, based on the definitions of Rule 30.
The program displays each row as it evolves over the specified number of generations.

## Why Does Rule 30 Matter?
Randomness and Complexity: Despite being a simple deterministic rule, Rule 30 generates patterns that appear random. This has made it useful in fields such as cryptography and random number generation.
Simple Rules, Complex Behavior: Rule 30 exemplifies how complex, unpredictable behavior can emerge from simple deterministic rules. This has implications in mathematics, physics, and the study of complex systems.
Theoretical Importance: Wolfram used Rule 30 and similar cellular automata to explore fundamental questions about computation, determinism, and the nature of complexity.


Rule 30 serves as a powerful example of how simple, deterministic rules can produce complex and seemingly chaotic behavior. Whether viewed as a mathematical curiosity, a tool for studying complex systems, or a source of randomness, Rule 30 captures the essence of emergent complexity—a theme that continues to fascinate scientists, mathematicians, and computer scientists alike. 

## Advanced Stuff Beware

A more rigorious code to implement such and animate it is as below


![image](https://github.com/user-attachments/assets/f9ee1d0e-e396-4ee3-90e3-4d0bf6a600b1)

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def rule30(left, center, right):
    """Apply Rule 30 to determine the next state of a cell."""
    return left ^ (center | right)

def initialize_grid(size):
    """Initialize the grid with a single cell in the middle set to 1."""
    grid = np.zeros(size, dtype=int)
    grid[size // 2] = 1
    return grid

def update_grid(grid):
    """Update the grid according to Rule 30."""
    new_grid = np.zeros_like(grid)
    for i in range(1, len(grid) - 1):
        new_grid[i] = rule30(grid[i - 1], grid[i], grid[i + 1])
    return new_grid

def generate_automaton(size, generations):
    """Generate the cellular automaton for a given number of generations."""
    grid = initialize_grid(size)
    automaton = [grid]
    for _ in range(generations - 1):
        grid = update_grid(grid)
        automaton.append(grid)
    return np.array(automaton)

def plot_automaton(automaton):
    """Plot the cellular automaton."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(automaton, cmap='binary', interpolation='nearest')
    ax.set_title('Cellular Automaton - Rule 30')
    ax.set_xlabel('Cell Index')
    ax.set_ylabel('Generation')
    plt.show()

def animate_automaton(automaton):
    """Animate the cellular automaton."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Cellular Automaton - Rule 30')
    ax.set_xlabel('Cell Index')
    ax.set_ylabel('Generation')

    def update(frame):
        ax.clear()
        ax.imshow(automaton[:frame+1], cmap='binary', interpolation='nearest')
        ax.set_title('Cellular Automaton - Rule 30')
        ax.set_xlabel('Cell Index')
        ax.set_ylabel('Generation')

    ani = animation.FuncAnimation(fig, update, frames=len(automaton), repeat=False)
    return ani

# Parameters
size = 101  # Size of the grid (number of cells)
generations = 100  # Number of generations

# Generate and plot the automaton
automaton = generate_automaton(size, generations)
plot_automaton(automaton)

# Generate and display the animation
ani = animate_automaton(automaton)
plt.show()
```

## Very advanced stuff. Beware

One of the use of such method is to simulate the spread as disease as shown below

![image](https://github.com/user-attachments/assets/19ddcd44-300b-47a1-bb58-5c7c27b23b89)

![image](https://github.com/user-attachments/assets/6a6e3d90-9f37-4f7e-a338-71c297a075eb)

The simulation presented offers a comprehensive model for understanding disease spread by incorporating geographic considerations, population movement dynamics, vaccination strategies, and boundary constraints. It uses a 2D grid to represent a geographical area, where each cell can hold an individual. Individuals exist in various states such as susceptible (healthy but at risk), infected (capable of spreading disease), recovered (with temporary immunity), immune (with long-term immunity), asymptomatic (carriers without symptoms but still contagious), and vaccinated (with partial protection from infection). The boundaries between regions can be porous, semi-porous, or non-porous, influencing how individuals move between areas, thereby simulating the impact of geographical restrictions or policy measures like quarantines and lockdowns.

The simulation starts with an initial state of the population distributed across the grid, and over generations, individuals interact based on probabilities of infection, movement, and vaccination rollout. Movement is influenced by boundaries: porous boundaries allow free movement, semi-porous boundaries allow movement with a 50% probability, and non-porous boundaries do not permit movement at all. Vaccination reduces the susceptibility of individuals, effectively limiting the spread of infection in regions where it is implemented.

The first visualization demonstrates the initial state of the population, where the grid is divided into regions by boundaries. Black lines represent non-porous boundaries, orange lines denote semi-porous boundaries, and gray lines indicate porous boundaries. The distribution of individuals in various states can be observed, with clusters of susceptible, infected, recovered, immune, asymptomatic, and vaccinated individuals. The boundaries clearly influence disease transmission; regions with non-porous boundaries tend to confine the spread of infection, while areas with porous boundaries experience more widespread disease spread due to unrestricted movement.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

SUSCEPTIBLE, INFECTED, RECOVERED, IMMUNE, ASYMPTOMATIC, VACCINATED = 0, 1, 2, 3, 4, 5
POROUS, SEMI_POROUS, NON_POROUS = 0, 1, 2  # Boundary types

def initialize_population(grid_size, p_infected=0.01, p_asymptomatic=0.01, p_vaccinated=0.0, density=0.5):
    """Initialize the 2D population grid with susceptible, vaccinated, infected, and asymptomatic individuals based on density."""
    grid = np.full(grid_size, -1, dtype=int)  # -1 indicates an unoccupied cell
    num_individuals = int(grid_size[0] * grid_size[1] * density)
    positions = np.random.choice(grid_size[0] * grid_size[1], size=num_individuals, replace=False)
    x_indices = positions // grid_size[1]
    y_indices = positions % grid_size[1]
    grid[x_indices, y_indices] = SUSCEPTIBLE

    infected_indices = np.random.choice([False, True], size=num_individuals, p=[1-p_infected, p_infected])
    asymptomatic_indices = np.random.choice([False, True], size=num_individuals, p=[1-p_asymptomatic, p_asymptomatic])
    vaccinated_indices = np.random.choice([False, True], size=num_individuals, p=[1-p_vaccinated, p_vaccinated])

    grid[x_indices[infected_indices], y_indices[infected_indices]] = INFECTED
    grid[x_indices[asymptomatic_indices], y_indices[asymptomatic_indices]] = ASYMPTOMATIC
    grid[x_indices[vaccinated_indices], y_indices[vaccinated_indices]] = VACCINATED

    return grid

def get_neighbors(grid, x, y):
    """Get the list of neighbors for the cell (x, y) in the grid."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                neighbors.append(grid[nx, ny])
    return neighbors

def is_movement_allowed(x, y, new_x, new_y, boundary_map):
    """Check if movement is allowed based on boundary type."""
    # Ensure indices are within the boundary map limits
    if 0 <= min(x, new_x) < boundary_map.shape[0] and 0 <= min(y, new_y) < boundary_map.shape[1]:
        boundary_type = boundary_map[min(x, new_x), min(y, new_y)]
        if boundary_type == NON_POROUS:
            return False
        elif boundary_type == SEMI_POROUS:
            return np.random.rand() < 0.5  # 50% chance of crossing
    return True  # Porous boundary allows movement


def update_population(grid, infection_prob=0.2, recovery_time=5, immunity_time=10, vaccination_prob=0.01, generation=0, vaccination_start=20):
    """Update the 2D population grid according to the disease spread rules and vaccination rollout."""
    new_grid = np.copy(grid)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] in (INFECTED, ASYMPTOMATIC):
                # Recover with a certain probability
                if np.random.rand() < 1/recovery_time:
                    new_grid[x, y] = RECOVERED
            elif grid[x, y] == RECOVERED:
                # Gain immunity with a certain probability
                if np.random.rand() < 1/immunity_time:
                    new_grid[x, y] = IMMUNE
            elif grid[x, y] == SUSCEPTIBLE:
                # Vaccination rollout starts after a certain number of generations
                if generation >= vaccination_start and np.random.rand() < vaccination_prob:
                    new_grid[x, y] = VACCINATED
                else:
                    # Check for infection from neighbors
                    neighbors = get_neighbors(grid, x, y)
                    if any(neighbor in (INFECTED, ASYMPTOMATIC) for neighbor in neighbors):
                        if np.random.rand() < infection_prob:
                            new_grid[x, y] = INFECTED if np.random.rand() > 0.5 else ASYMPTOMATIC
            elif grid[x, y] == VACCINATED:
                # Vaccinated individuals have a lower chance of becoming infected (simulating vaccine effectiveness)
                neighbors = get_neighbors(grid, x, y)
                if any(neighbor in (INFECTED, ASYMPTOMATIC) for neighbor in neighbors):
                    if np.random.rand() < infection_prob * 0.1:  # Reduced infection probability for vaccinated individuals
                        new_grid[x, y] = INFECTED if np.random.rand() > 0.5 else ASYMPTOMATIC
    return new_grid

def move_individuals(grid, boundary_map, move_prob=0.1):
    """Randomly move individuals within the grid while considering boundaries."""
    new_grid = np.copy(grid)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] != -1 and np.random.rand() < move_prob:  # Only move occupied cells
                new_x = x + np.random.choice([-1, 0, 1])
                new_y = y + np.random.choice([-1, 0, 1])
                if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:
                    if is_movement_allowed(x, y, new_x, new_y, boundary_map):
                        new_grid[x, y], new_grid[new_x, new_y] = new_grid[new_x, new_y], new_grid[x, y]
    return new_grid

def generate_population(grid_size, generations, p_infected=0.01, p_asymptomatic=0.01, p_vaccinated=0.0, infection_prob=0.2, recovery_time=5, immunity_time=10, move_prob=0.1, vaccination_prob=0.01, vaccination_start=20, density=0.5, boundary_map=None):
    """Generate the disease spread for a given number of generations."""
    grid = initialize_population(grid_size, p_infected, p_asymptomatic, p_vaccinated, density)
    population = [grid]
    for gen in range(generations - 1):
        grid = update_population(grid, infection_prob, recovery_time, immunity_time, vaccination_prob, generation=gen, vaccination_start=vaccination_start)
        grid = move_individuals(grid, boundary_map, move_prob)
        population.append(grid)
    return np.array(population)

def plot_population(population, boundary_map):
    """Plot the disease spread over a 2D grid with boundaries."""
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap('viridis', 6)
    ax.imshow(population[0], cmap=cmap, interpolation='nearest')
    ax.set_title('Disease Spread Simulation with Geography, Boundaries, Immunity, Movement, Asymptomatic Carriers, and Vaccination')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Draw boundaries
    for x in range(boundary_map.shape[0]):
        for y in range(boundary_map.shape[1]):
            if boundary_map[x, y] == NON_POROUS:
                color = 'k'  # Black for non-porous
                linewidth = 3
            elif boundary_map[x, y] == SEMI_POROUS:
                color = 'orange'  # Orange for semi-porous
                linewidth = 2
            else:  # POROUS
                color = 'gray'  # Gray for porous
                linewidth = 1
            # Draw boundary lines around the cell
            ax.plot([y, y + 1], [x, x], color=color, linewidth=linewidth)  # Top border
            ax.plot([y, y + 1], [x + 1, x + 1], color=color, linewidth=linewidth)  # Bottom border
            ax.plot([y, y], [x, x + 1], color=color, linewidth=linewidth)  # Left border
            ax.plot([y + 1, y + 1], [x, x + 1], color=color, linewidth=linewidth)  # Right border

    # Create legend
    legend_labels = ['Susceptible', 'Infected', 'Recovered', 'Immune', 'Asymptomatic', 'Vaccinated']
    colors = [cmap(i) for i in range(6)]
    patches = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(6)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.show()


def animate_population(population):
    """Animate the disease spread over the 2D grid."""
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap('viridis', 6)
    ax.set_title('Disease Spread Simulation with Geography, Boundaries, Immunity, Movement, Asymptomatic Carriers, and Vaccination')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Create legend
    legend_labels = ['Susceptible', 'Infected', 'Recovered', 'Immune', 'Asymptomatic', 'Vaccinated']
    colors = [cmap(i) for i in range(6)]
    patches = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(6)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    def update(frame):
        ax.clear()
        ax.imshow(population[frame], cmap=cmap, interpolation='nearest')
        ax.set_title('Disease Spread Simulation with Geography, Boundaries, Immunity, Movement, Asymptomatic Carriers, and Vaccination')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ani = animation.FuncAnimation(fig, update, frames=len(population), repeat=False)
    return ani

# Parameters
grid_size = (50, 50)  # Size of the grid (20x20 geographical area)
generations = 100
boundary_map = np.random.choice([POROUS, SEMI_POROUS, NON_POROUS], size=(grid_size[0] - 1, grid_size[1] - 1))

# Generate population
population = generate_population(grid_size, generations, p_vaccinated=0.05, vaccination_prob=0.02, vaccination_start=10, density=0.6, boundary_map=boundary_map)

# Plot population
plot_population(population, boundary_map)


# Animate population
ani = animate_population(population)
plt.show()
```



