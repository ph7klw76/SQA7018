import random as rd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Simulation Parameters
# ---------------------------
GRID_SIZE = 100          # Size of the environment grid: 100 x 100
INITIAL_NUM_BACTERIA = 25
NUM_NUTRIENT = 10
BITES_NUTRIENT = 100     # Each nutrient has this many "bites" before it is fully consumed
DEATH_PROBABILITY = 0.01 # 1% chance of death, but only after MAX_STEPS for each bacterium
REPRODUCE_PROBABILITY = 0.5
MOVEMENT_PROBABILITY = 0.8  # Probability of moving toward nutrient vs. random movement
MAX_STEPS = 500
NUM_SIMULATION = 10      # Number of simulations to run for statistical variability

# ---------------------------
# Nutrient Class
# ---------------------------
class Nutrient:
    """
    Represents a nutrient source in the grid.
    
    Attributes:
        x (int): X-position of the nutrient in the grid.
        y (int): Y-position of the nutrient in the grid.
        bites (int): Remaining "bite" capacity of the nutrient.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.bites = BITES_NUTRIENT

    def consume(self):
        """
        Decrease the nutrient's bites by 1. Return True if the nutrient is fully consumed.
        """
        self.bites -= 1
        # If bites drop to zero or below, the nutrient is depleted
        if self.bites <= 0:
            return True
        return False


# ---------------------------
# Bacterium Class
# ---------------------------
class Bacterium:
    """
    Represents a single bacterium with position and survival/reproduction logic.
    
    Attributes:
        x (int): X-position of the bacterium in the grid.
        y (int): Y-position of the bacterium in the grid.
        steps (int): The total steps the bacterium has survived so far.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.steps = 0

    def random_move(self):
        """
        Moves the bacterium randomly in both x and y directions (including diagonals).
        Wraps around the grid if it moves beyond the boundaries using modulo operation.
        """
        self.x = (self.x + rd.choice([-1, 0, 1])) % GRID_SIZE
        self.y = (self.y + rd.choice([-1, 0, 1])) % GRID_SIZE

    def move_towards_nutrient(self, grid):
        """
        Moves the bacterium one step closer to any adjacent nutrient cell.
        If no adjacent nutrient is found, it moves randomly.
        
        Note: This function checks only the 4 orthogonal neighbors (up, down, left, right).
              If a neighbor has a positive value (indicating nutrient bites), we move there.
        """
        best_move = None
        best_distance = float('inf')

        # Check the 4 orthogonal directions for potential nutrient presence
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = (self.x + dx) % GRID_SIZE
            ny = (self.y + dy) % GRID_SIZE

            # If this cell has nutrient (> 0), consider it
            if grid[nx, ny] > 0:
                distance = abs(nx - self.x) + abs(ny - self.y)
                if distance < best_distance:
                    best_move = (nx, ny)
                    best_distance = distance

        if best_move:
            # Move to the best cell found
            self.x, self.y = best_move
        else:
            # No adjacent nutrient cell found; move randomly
            self.random_move()

    def move(self, nutrient_locations, grid):
        """
        Decides whether to move towards nutrient or randomly,
        then increments the bacterium's life steps.
        """
        # With MOVEMENT_PROBABILITY chance, try moving toward a nutrient
        if np.random.rand() < MOVEMENT_PROBABILITY and nutrient_locations:
            self.move_towards_nutrient(grid)
        else:
            self.random_move()

        # Increase step counter each time the bacterium moves
        self.steps += 1

    def check_death(self):
        """
        Returns True if the bacterium dies.
        Bacterium becomes death-eligible only after MAX_STEPS.
        """
        return np.random.rand() < DEATH_PROBABILITY

    def reproduce(self, bacteria_list):
        """
        With REPRODUCE_PROBABILITY chance, creates a new bacterium in a neighboring cell.
        
        Args:
            bacteria_list (list): The current list of bacteria. The offspring is appended here.
        """
        if np.random.rand() < REPRODUCE_PROBABILITY:
            new_x = (self.x + rd.choice([-1, 0, 1])) % GRID_SIZE
            new_y = (self.y + rd.choice([-1, 0, 1])) % GRID_SIZE
            bacteria_list.append(Bacterium(new_x, new_y))


# ---------------------------
# Helper Functions
# ---------------------------
def place_nutrients(num_nutrient, grid_size):
    """
    Randomly place a specified number of nutrients in the grid.
    
    Args:
        num_nutrient (int): Number of nutrient sources to create.
        grid_size (int): Dimension of the square grid.
    
    Returns:
        list: A list of Nutrient objects placed randomly.
    """
    nutrient_locations = []
    for _ in range(num_nutrient):
        x = rd.randint(0, grid_size - 1)
        y = rd.randint(0, grid_size - 1)
        nutrient_locations.append(Nutrient(x, y))
    return nutrient_locations


def run_simulation():
    """
    Runs a single simulation of bacteria in a 100x100 grid with randomly placed nutrients.
    
    Logic Flow:
    1. Place nutrients and initialize a bacteria list.
    2. Build a grid indicating how many 'bites' each cell has (0 if none).
    3. While there are bacteria alive:
       - Record current population count.
       - For each bacterium:
         a) Move it (towards nutrient or randomly).
         b) Check if it should die (only if steps > MAX_STEPS).
         c) If not dead, check if it is on a nutrient cell; if so, consume & reproduce.
    4. Returns a list of population sizes at each simulation step until all bacteria die.
    """
    # 1. Place nutrients
    nutrient_locations = place_nutrients(NUM_NUTRIENT, GRID_SIZE)

    # 2. Initialize bacteria with random positions
    bacteria_list = [
        Bacterium(rd.randint(0, GRID_SIZE - 1), rd.randint(0, GRID_SIZE - 1))
        for _ in range(INITIAL_NUM_BACTERIA)
    ]

    # Create a grid to represent the "bites" available in each cell
    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    # Fill grid with the initial nutrient bites
    for nutrient in nutrient_locations:
        grid[nutrient.x, nutrient.y] = nutrient.bites

    # Keep track of population over time
    population = []

    # 3. Main simulation loop
    while bacteria_list:
        population.append(len(bacteria_list))

        new_bacteria_list = []
        for bacterium in bacteria_list:
            # Move the bacterium and increment its step counter
            bacterium.move(nutrient_locations, grid)

            # Only after MAX_STEPS does the bacterium risk death
            if bacterium.steps > MAX_STEPS:
                if bacterium.check_death():
                    # Bacterium dies and is thus skipped
                    continue

            # If bacterium is still alive, check if it is on a nutrient
            for nutrient in nutrient_locations:
                if (bacterium.x, bacterium.y) == (nutrient.x, nutrient.y):
                    # Consume one "bite" from the nutrient
                    if nutrient.consume():
                        # If nutrient is fully consumed, remove from the list
                        nutrient_locations.remove(nutrient)
                        # Clear the cell in the grid
                        grid[nutrient.x, nutrient.y] = 0

                    # Attempt reproduction in a neighboring cell
                    bacterium.reproduce(new_bacteria_list)

            # Keep the bacterium in the new list if it is alive
            new_bacteria_list.append(bacterium)

        # Update the list of bacteria for the next iteration
        bacteria_list = new_bacteria_list

    return population


def run_multiple_simulations(num_simulations):
    """
    Runs the simulation multiple times and collects population data.
    
    Args:
        num_simulations (int): Number of simulations to run.
    
    Returns:
        list of lists: Each sublist contains population data over time for one simulation.
    """
    all_simulations = [run_simulation() for _ in range(num_simulations)]
    return all_simulations


def plot_simulations(simulations):
    """
    Plots the population curve (number of bacteria vs. simulation steps) for each run.
    
    Args:
        simulations (list of lists): Each sublist is one run's population data over time.
    """
    for i, sim in enumerate(simulations):
        plt.plot(range(len(sim)), sim, label=f"Simulation {i+1}")

    plt.title("Bacteria Number vs. Time Steps for Multiple Simulations")
    plt.xlabel("Steps")
    plt.ylabel("Bacteria Number")
    plt.legend()
    plt.show()


# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    # Run multiple simulations and plot the result
    sims = run_multiple_simulations(NUM_SIMULATION)
    plot_simulations(sims)
