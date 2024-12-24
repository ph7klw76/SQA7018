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
# These constants control the simulation's behavior and environment.
GRID_SIZE = 100                  # Size of the square grid.
INITIAL_NUM_BACTERIA = 25        # Initial number of bacteria.
NUM_NUTRIENT = 10                # Number of nutrient cells in the grid.
BITES_NUTRIENT = 100             # How many "bites" a nutrient cell contains.
DEATH_PROBABILITY = 0.01         # Probability of death after MAX_STEPS.
REPRODUCE_PROBABILITY = 0.5      # Probability that a bacterium reproduces when it eats.
MOVE_TOWARD_NUTRIENT_PROB = 0.8  # Probability of bacteria moving toward a nutrient cell.
MAX_STEPS = 500                  # Maximum steps a bacterium before Death Probability Initiated
NUM_SIMULATIONS = 10             # Number of simulations to run for statistical comparison.

# ---------------------------
# Bacterium Class
# ---------------------------
class Bacterium:
    """
    Represents a single bacterium in the simulation.

    Attributes:
        x (int): X-position on the grid.
        y (int): Y-position on the grid.
        steps (int): Tracks how many steps the bacterium has taken.
    """
    def __init__(self, x, y):
        self.x = x  # Initial x-coordinate
        self.y = y  # Initial y-coordinate
        self.steps = 0  # Tracks survival time in steps

    def random_move(self):
        """
        Moves the bacterium randomly to one of its 9 possible adjacent cells.
        Wraps around if it goes off the grid using modulo arithmetic.
        """
        self.x = (self.x + rd.choice([-1, 0, 1])) % GRID_SIZE
        self.y = (self.y + rd.choice([-1, 0, 1])) % GRID_SIZE

    def move_towards_nutrient(self, grid):
        """
        Attempts to move towards a nutrient-containing neighboring cell.
        If no such cell exists, the bacterium moves randomly.
        """
        best_move = None  # Stores the best movement towards nutrients
        best_distance = float('inf')  # Minimal distance to nutrient found

        # Loop over orthogonal neighbors to find nutrients
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = (self.x + dx) % GRID_SIZE
            ny = (self.y + dy) % GRID_SIZE
            if grid[nx, ny] > 0:  # Check if nutrient exists in the neighboring cell
                distance = abs(nx - self.x) + abs(ny - self.y)
                if distance < best_distance:
                    best_distance = distance
                    best_move = (nx, ny)

        if best_move:
            self.x, self.y = best_move  # Move to the best nutrient cell
        else:
            self.random_move()  # No nutrient found, move randomly

    def move(self, grid):
        """
        Executes a move based on probabilities:
        - Moves toward nutrient with MOVE_TOWARD_NUTRIENT_PROB probability.
        - Otherwise moves randomly.
        """
        if rd.random() < MOVE_TOWARD_NUTRIENT_PROB:
            self.move_towards_nutrient(grid)
        else:
            self.random_move()

        # Increment steps to track survival duration
        self.steps += 1

    def check_death(self):
        """
        Determines whether the bacterium dies due to aging or randomness.
        Returns True if the bacterium should die.
        """
        if self.steps > MAX_STEPS:
            return rd.random() < DEATH_PROBABILITY
        return False

    def reproduce(self, new_bacteria_list):
        """
        Attempts to reproduce with a fixed probability.
        Spawns a new bacterium in a random adjacent cell.
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
    Initializes the grid with nutrients randomly distributed.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)  # Grid initialized with zeros

    for _ in range(NUM_NUTRIENT):
        x = rd.randint(0, GRID_SIZE - 1)
        y = rd.randint(0, GRID_SIZE - 1)
        grid[x, y] = BITES_NUTRIENT  # Place nutrients randomly in the grid

    return grid


def run_simulation():
    """
    Runs a single simulation of bacteria survival and reproduction.
    Tracks population size over time.
    """
    grid = initialize_grid()  # Nutrient grid setup
    bacteria_list = [Bacterium(rd.randint(0, GRID_SIZE - 1), rd.randint(0, GRID_SIZE - 1))
                     for _ in range(INITIAL_NUM_BACTERIA)]

    population_history = []  # Tracks population size at each step

    while bacteria_list:
        population_history.append(len(bacteria_list))
        new_bacteria_list = []

        for bacterium in bacteria_list:
            bacterium.move(grid)  # Bacterium moves on the grid

            if bacterium.check_death():
                continue  # Bacterium dies, do not add to next iteration

            if grid[bacterium.x, bacterium.y] > 0:  # Check for nutrient
                grid[bacterium.x, bacterium.y] -= 1  # Consume nutrient
                bacterium.reproduce(new_bacteria_list)  # Attempt reproduction

            new_bacteria_list.append(bacterium)  # Keep alive bacteria

        bacteria_list = new_bacteria_list  # Update bacteria for next step

    return population_history


@profile
def run_multiple_simulations(num_sims):
    """
    Executes multiple independent simulations for statistical analysis.
    """
    return [run_simulation() for _ in range(num_sims)]


def plot_results(simulations):
    """
    Visualizes the population dynamics across multiple simulations.
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
# Executes the full simulation suite and plots the results.
if __name__ == "__main__":
    all_sims = run_multiple_simulations(NUM_SIMULATIONS)
    plot_results(all_sims)
