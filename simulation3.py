# ---------------------------
# GLOBAL EXHAUSIVE SEARCH
# ---------------------------

import random as rd
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile

# ---------------------------
# Simulation Parameters
# ---------------------------
GRID_SIZE = 100              # The environment is a 100 x 100 grid
INITIAL_NUM_BACTERIA = 25
NUM_NUTRIENT = 10            # Number of nutrient-rich cells to place randomly
BITES_NUTRIENT = 100         # Each nutrient cell starts with this many "bites"
DEATH_PROBABILITY = 0.01     # Per-step chance of death, but only after MAX_STEPS
REPRODUCE_PROBABILITY = 0.5  # Chance that a bacterium reproduces each step it feeds
MOVE_TOWARD_NUTRIENT_PROB = 0.8
MAX_STEPS = 500              # After 500 steps, a bacterium can die with DEATH_PROBABILITY
NUM_SIMULATIONS = 10         # Number of simulation runs

# ---------------------------
# Bacterium Class
# ---------------------------
class Bacterium:
    """
    Represents a single bacterium with position, movement, reproduction, and death logic.

    Attributes:
        x (int): Current x-coordinate in the grid (0 <= x < GRID_SIZE).
        y (int): Current y-coordinate in the grid (0 <= y < GRID_SIZE).
        steps (int): How many steps this bacterium has been alive.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.steps = 0

    def random_move(self):
        """
        Moves the bacterium to a randomly chosen adjacent (or same) cell
        in both x and y directions (including diagonals).

        The modulo operation ensures wrapping around the grid edges
        so that the grid is effectively 'toroidal'.
        """
        self.x = (self.x + rd.choice([-1, 0, 1])) % GRID_SIZE
        self.y = (self.y + rd.choice([-1, 0, 1])) % GRID_SIZE

    def move_towards_closest_nutrient(self, grid):
        """
        Moves the bacterium one step (up/down/left/right) closer to the globally nearest
        nutrient cell on the entire grid. The distance metric used here is
        the Manhattan distance.

        Steps:
          1. Find all cells (nx, ny) in the grid for which grid[nx, ny] > 0.
          2. Choose the cell that is closest by Manhattan distance:
                 d = |nx - x| + |ny - y|
          3. Move one step in whichever orthogonal direction reduces this distance.
             If the bacterium is already at that cell, it won't move horizontally
             or vertically.

        If there are no nutrients at all on the grid, the bacterium moves randomly.
        """
        # Step 1: Get positions of all nutrient cells
        nutrient_positions = np.argwhere(grid > 0)

        if len(nutrient_positions) == 0:
            # No nutrients at all; move randomly
            self.random_move()
            return

        # Step 2: Find the globally closest nutrient cell by Manhattan distance
        # Note: If you want to account for wrapping distances, you would need a
        # different formula, but here we assume direct Manhattan distance.
        distances = [
            np.sqrt(abs(nx - self.x)**2 + abs(ny - self.y)**2)
            for (nx, ny) in nutrient_positions
        ]
        min_index = np.argmin(distances)
        target_x, target_y = nutrient_positions[min_index]
        # This is the absolute closest nutrient cell on the grid

        # Step 3: Move one step closer to (target_x, target_y).
        # (with probability MOVE_TOWARD_NUTRIENT_PROB) or move randomly.
        # We reduce the Manhattan distance by adjusting x or y by 1 in the
        # appropriate direction. We do not move diagonally here.
        if rd.random() < MOVE_TOWARD_NUTRIENT_PROB:
            if target_x < self.x:
                self.x -= 1
            elif target_x > self.x:
                self.x += 1
            elif target_y < self.y:
                self.y -= 1
            elif target_y > self.y:
                self.y += 1
            else:
                # Already at the nutrient cell; no movement needed
                pass
        else:
            self.random_move()

        # Apply modulo if you want wrapping behavior
        self.x %= GRID_SIZE
        self.y %= GRID_SIZE

    def move(self, grid):
        """
        Decides whether to attempt moving towards a nutrient cell
        Then increments the 'steps' counter to reflect aging.
        """

        self.move_towards_closest_nutrient(grid)
        self.steps += 1

    def check_death(self):
        """
        Returns True if the bacterium dies.
        The bacterium can die only after it has exceeded MAX_STEPS in age.
        Once it passes that threshold, each subsequent step has DEATH_PROBABILITY chance.
        """
        if self.steps > MAX_STEPS:
            return rd.random() < DEATH_PROBABILITY
        return False

    def reproduce(self, new_bacteria_list):
        """
        With a probability REPRODUCE_PROBABILITY, spawns a new bacterium in one of the
        9 surrounding cells (including diagonals).

        Args:
            new_bacteria_list (list): A list to which the new bacterium (if any) is appended.
        """
        if rd.random() < REPRODUCE_PROBABILITY:
            # Random offset in x and y, including diagonals
            new_x = (self.x + rd.choice([-1, 0, 1])) % GRID_SIZE
            new_y = (self.y + rd.choice([-1, 0, 1])) % GRID_SIZE
            new_bacteria_list.append(Bacterium(new_x, new_y))

# ---------------------------
# Simulation Logic
# ---------------------------
def initialize_grid():
    """
    Creates and returns a 2D NumPy array of shape (GRID_SIZE, GRID_SIZE).
    Each cell starts at 0. Then we randomly place NUM_NUTRIENT cells, each
    loaded with BITES_NUTRIENT 'bites'.

    Returns:
        np.ndarray: The 2D array representing how many bites of nutrient
                    each cell currently holds.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    for _ in range(NUM_NUTRIENT):
        x = rd.randint(0, GRID_SIZE - 1)
        y = rd.randint(0, GRID_SIZE - 1)
        grid[x, y] = BITES_NUTRIENT  # Place a nutrient cell
    return grid

def run_simulation():
    """
    Runs a single simulation of the bacterial colony, returning a list that
    tracks the population size (number of living bacteria) at each simulation step.

    Steps:
      1. Create a grid with randomly placed nutrient cells.
      2. Initialize bacteria in random positions.
      3. While there are bacteria alive:
          a) Record current population size.
          b) For each bacterium:
             - Move according to the movement logic (random or global-nutrient-seeking).
             - Potentially die if it exceeds MAX_STEPS and a random check passes.
             - If it survives and there's nutrient at its position, consume it
               and possibly reproduce.
    """
    grid = initialize_grid()

    # Randomly create the initial bacteria population
    bacteria_list = [
        Bacterium(rd.randint(0, GRID_SIZE - 1), rd.randint(0, GRID_SIZE - 1))
        for _ in range(INITIAL_NUM_BACTERIA)
    ]

    population_history = []

    while bacteria_list:
        population_history.append(len(bacteria_list))
        new_bacteria_list = []

        for bacterium in bacteria_list:
            # Move the bacterium (either random or truly global-nutrient-seeking)
            bacterium.move(grid)

            # Check if the bacterium dies (only after it has exceeded MAX_STEPS)
            if bacterium.check_death():
                continue  # Skip adding it to the next generation

            # If alive, see if there's nutrient at current position
            if grid[bacterium.x, bacterium.y] > 0:
                # Consume one "bite" of nutrient
                grid[bacterium.x, bacterium.y] -= 1
                # Attempt to reproduce
                bacterium.reproduce(new_bacteria_list)

            # Keep the bacterium for the next iteration
            new_bacteria_list.append(bacterium)

        # Update the population for the next step
        bacteria_list = new_bacteria_list

    return population_history

@profile
def run_multiple_simulations(num_sims):
    """
    Runs the entire simulation multiple times, storing the population history for each run.
    The @profile decorator from memory_profiler will measure memory usage of this function.

    Args:
        num_sims (int): How many separate simulations to run.

    Returns:
        List[List[int]]: A list of lists, where each sub-list is the population
                         history from one simulation.
    """
    return [run_simulation() for _ in range(num_sims)]

def plot_results(simulations):
    """
    Given a list of population histories (one per simulation),
    plot them all on the same figure to compare outcomes.

    Args:
        simulations (List[List[int]]): Each sub-list is the population size at each step
                                       for one simulation run.
    """
    plt.figure(figsize=(10, 6))
    for i, sim_data in enumerate(simulations, start=1):
        plt.plot(sim_data, label=f"Simulation {i}")
    plt.title("Bacteria Population Over Time (Global Nutrient Seeking)")
    plt.xlabel("Step")
    plt.ylabel("Number of Bacteria")
    plt.legend()
    plt.show()

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    # Run multiple simulations sequentially and capture their population histories
    all_sims = run_multiple_simulations(NUM_SIMULATIONS)

    # Plot the results to visualize how population changes over time
    plot_results(all_sims)
