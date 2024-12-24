# ---------------------------
# IMMEDIATE LOCAL SEARCH
# ---------------------------
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile

# ---------------------------
# Simulation Parameters
# ---------------------------
GRID_SIZE = 100
INITIAL_NUM_BACTERIA = 25
NUM_NUTRIENT = 10
BITES_NUTRIENT = 100
DEATH_PROBABILITY = 0.01     # Death chance after MAX_STEPS
REPRODUCE_PROBABILITY = 0.5
MOVE_TOWARD_NUTRIENT_PROB = 0.8
MAX_STEPS = 500
NUM_SIMULATIONS = 10

# ---------------------------
# Bacterium Class
# ---------------------------
class Bacterium:
    """
    Represents a single bacterium with position and survival/reproduction logic.
    
    Attributes:
        x (int): X-position of the bacterium in the grid.
        y (int): Y-position of the bacterium in the grid.
        steps (int): The total steps the bacterium has survived.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.steps = 0

    def random_move(self):
        """
        Moves the bacterium randomly in both x and y directions (including diagonals).
        Wraps around the grid if it moves beyond boundaries using modulo operation.
        """
        self.x = (self.x + rd.choice([-1, 0, 1])) % GRID_SIZE
        self.y = (self.y + rd.choice([-1, 0, 1])) % GRID_SIZE

    def move_towards_nutrient(self, grid):
        """
        Tries to move to an adjacent cell (up/down/left/right) that has a positive
        nutrient value (> 0). If none of the four neighbors has nutrient, it moves randomly.
        """
        best_move = None
        best_distance = float('inf')

        # Check only 4 orthogonal neighbors for nutrient presence
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = (self.x + dx) % GRID_SIZE
            ny = (self.y + dy) % GRID_SIZE
            if grid[nx, ny] > 0:  # There's nutrient in this neighbor
                distance = abs(nx - self.x) + abs(ny - self.y)
                if distance < best_distance:
                    best_distance = distance
                    best_move = (nx, ny)

        if best_move:
            self.x, self.y = best_move
        else:
            self.random_move()

    def move(self, grid):
        """
        Decides whether to move toward nutrient (with probability MOVE_TOWARD_NUTRIENT_PROB)
        or move randomly. Then increments the bacterium's 'steps' counter.
        """
        if rd.random() < MOVE_TOWARD_NUTRIENT_PROB:
            self.move_towards_nutrient(grid)
        else:
            self.random_move()

        # Increment the step counter once per move
        self.steps += 1

    def check_death(self):
        """
        Returns True if the bacterium dies (after it has lived more than MAX_STEPS).
        """
        if self.steps > MAX_STEPS:
            return (rd.random() < DEATH_PROBABILITY)
        return False

    def reproduce(self, new_bacteria_list):
        """
        With REPRODUCE_PROBABILITY, spawns a new bacterium in one of the 9 surrounding cells
        (including diagonals).
        """
        if rd.random() < REPRODUCE_PROBABILITY:
            new_x = (self.x + rd.choice([-1, 0, 1])) % GRID_SIZE
            new_y = (self.y + rd.choice([-1, 0, 1])) % GRID_SIZE
            new_bacteria_list.append(Bacterium(new_x, new_y))


# ---------------------------
# Simulation Logic
# ---------------------------
def initialize_grid():
    """
    Initialize the grid with random nutrient placements.
    
    Returns:
        np.ndarray: A 2D grid of shape (GRID_SIZE, GRID_SIZE), where each cell
                    indicates how many "bites" of nutrient are present.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

    for _ in range(NUM_NUTRIENT):
        x = rd.randint(0, GRID_SIZE - 1)
        y = rd.randint(0, GRID_SIZE - 1)
        grid[x, y] = BITES_NUTRIENT

    return grid


def run_simulation():
    """
    Runs a single simulation, returning a list of the bacteria count at each step.
    """
    # Create the nutrient grid
    grid = initialize_grid()

    # Initialize bacteria population at random positions
    bacteria_list = [
        Bacterium(rd.randint(0, GRID_SIZE - 1), rd.randint(0, GRID_SIZE - 1))
        for _ in range(INITIAL_NUM_BACTERIA)
    ]

    population_history = []

    # Main simulation loop: runs until there are no bacteria left
    while bacteria_list:
        population_history.append(len(bacteria_list))
        new_bacteria_list = []

        for bacterium in bacteria_list:
            # Move the bacterium
            bacterium.move(grid)

            # Check for death (only after max steps)
            if bacterium.check_death():
                # Bacterium dies; skip adding it to the new list
                continue

            # If bacterium is alive, see if there's nutrient at its location
            if grid[bacterium.x, bacterium.y] > 0:
                # Consume one "bite"
                grid[bacterium.x, bacterium.y] -= 1

                # Attempt reproduction
                bacterium.reproduce(new_bacteria_list)

            # Keep the bacterium
            new_bacteria_list.append(bacterium)

        # Update the population for the next iteration
        bacteria_list = new_bacteria_list

    return population_history

@profile
def run_multiple_simulations(num_sims):
    """
    Runs multiple simulations in sequence and returns their population histories.
    """
    results = [run_simulation() for _ in range(num_sims)]
    return results


def plot_results(simulations):
    """
    Plots each simulation's population history on the same figure.
    """
    plt.figure(figsize=(10, 6))
    for i, sim in enumerate(simulations, start=1):
        plt.plot(sim, label=f"Simulation {i}")
    plt.title("Bacteria Population Over Time")
    plt.xlabel("Step")
    plt.ylabel("Number of Bacteria")
    plt.legend()
    plt.show()


# ---------------------------
# Main Entry Point
# ---------------------------

all_sims = run_multiple_simulations(NUM_SIMULATIONS)
plot_results(all_sims)
