# ---------------------------
# ARBITARY NEIGBOURHOOD SEARCH
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
        in both the x and y directions (including diagonals).
        
        The modulo operation ensures wrapping around the grid edges
        so that the grid is effectively 'toroidal'.
        """
        self.x = (self.x + rd.choice([-1, 0, 1])) % GRID_SIZE
        self.y = (self.y + rd.choice([-1, 0, 1])) % GRID_SIZE

    def get_neighbors(self, neighbor=1):
        """
        Returns a list of all relative (dx, dy) positions within a specified 'neighbor' radius,
        excluding the cell where the bacterium currently is (dx=0, dy=0).
        
        By default, neighbor=1 means we look at all cells in a 3x3 block (including diagonals).
        If neighbor=2, we consider a 5x5 block, etc.

        Args:
            neighbor (int): The search 'radius' around the bacterium.

        Returns:
            List[Tuple[int, int]]: A list of (dx, dy) offsets from the current position.
        """
        neighbors = []
        for dx in range(-neighbor, neighbor + 1):
            for dy in range(-neighbor, neighbor + 1):
                # Exclude the current cell (0, 0) to avoid moving onto itself
                if not (dx == 0 and dy == 0):
                    neighbors.append((dx, dy))
        return neighbors

    def move_towards_nutrient(self, grid, search_range=2):
        """
        Attempts to move the bacterium toward a more nutrient-rich cell within a local area.
        - We look at all cells within +/-search_range in x and y (including diagonals).
        - Among those, we pick the cell with the highest nutrient, breaking ties by picking
          the cell that is closer in Manhattan distance (with probability MOVE_TOWARD_NUTRIENT_PROB) 
          or randomnly

        If no nutrient cells are found in this local region, the bacterium moves randomly.

        Args:
            grid (np.ndarray): A 2D array storing the nutrient 'bites' at each position.
            search_range (int): How far (in x and y) to look for nutrient cells.
        """
        neighbors = self.get_neighbors(neighbor=search_range)
        best_move = None
        best_distance = float('inf')
        max_nutrient = 0  # Track the best (highest) nutrient found

        # Examine all nearby cells
        for dx, dy in neighbors:
            nx = (self.x + dx) % GRID_SIZE
            ny = (self.y + dy) % GRID_SIZE

            # Only consider cells with any nutrient
            if grid[nx, ny] > 0:
                # Manhattan distance to that cell from current position
                distance = np.sqrt(abs(dx)**2 + abs(dy)**2)
                
                # If we find a cell with strictly higher nutrient, prefer it
                # If nutrients are the same, prefer the closer cell
                if (grid[nx, ny] > max_nutrient) or (
                    grid[nx, ny] == max_nutrient and distance < best_distance
                ):
                    max_nutrient = grid[nx, ny]
                    best_distance = distance
                    best_move = (nx, ny)

        # Move to the best nutrient cell if found; otherwise, move randomly
        if best_move:
            if rd.random() < MOVE_TOWARD_NUTRIENT_PROB:
                self.x, self.y = best_move  # Move to the best nutrient cell
            else:
                self.random_move()
        else:
            self.random_move()

    def move(self, grid):
        """
        Decides whether to attempt moving towards a nutrient cell 
        Then increments the 'steps' counter to reflect aging.
        """
        self.move_towards_nutrient(grid, search_range=2)
        # Each time the bacterium moves, it has aged by one step
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
        9 surrounding cells (including diagonals). This allows for local expansion of the colony.

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
        grid[x, y] = BITES_NUTRIENT  # Place a nutrient cell with BITES_NUTRIENT bites
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
             - Move according to the movement logic (random or nutrient-seeking).
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
            # Move the bacterium (either random or nutrient-seeking)
            bacterium.move(grid)

            # Check if the bacterium dies (only after it has exceeded MAX_STEPS)
            if bacterium.check_death():
                continue  # Skip adding it to the next generation

            # If alive, see if there's nutrient at current position
            if grid[bacterium.x, bacterium.y] > 0:
                # Consume one "bite" of nutrient
                grid[bacterium.x, bacterium.y] -= 1
                # Attempt to reproduce (potentially adding an offspring to new_bacteria_list)
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
    results = [run_simulation() for _ in range(num_sims)]
    return results


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
    plt.title("Bacteria Population Over Time")
    plt.xlabel("Step")
    plt.ylabel("Number of Bacteria")
    plt.legend()
    plt.show()


# ---------------------------
# Main Entry Point
# ---------------------------

# Run multiple simulations sequentially and capture their population histories
all_sims = run_multiple_simulations(NUM_SIMULATIONS)

# Plot the results to visualize how population changes over time
plot_results(all_sims)

