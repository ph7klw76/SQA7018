# ---------------------------
# APPROXIMATE METHODS
# SUITABLE FOR LARGE CELLS
# SUITABLE FOR HIGH NUMBER OF BACTERIA
# ---------------------------

import random as rd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from memory_profiler import profile

# ---------------------------
# Simulation Parameters
# ---------------------------
GRID_SIZE = 1000
INITIAL_NUM_BACTERIA = 25
NUM_NUTRIENT = 100         # Potentially large
BITES_NUTRIENT = 50        
DEATH_PROBABILITY = 0.01
REPRODUCE_PROBABILITY = 0.5
MOVE_TOWARD_NUTRIENT_PROB = 0.8
MAX_STEPS = 500
NUM_SIMULATIONS = 3        # Fewer simulations here for brevity
INF_DISTANCE = 10**9

# ---------------------------
# Bacterium Class
# ---------------------------
class Bacterium:
    """
    Bacterium with position, step count, and movement/death/reproduction logic.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.steps = 0

    def random_move(self):
        """
        Moves the bacterium +/-1 in x and/or y (including diagonals).
        Wraps around grid boundaries.
        """
        self.x = (self.x + rd.choice([-1, 0, 1])) % GRID_SIZE
        self.y = (self.y + rd.choice([-1, 0, 1])) % GRID_SIZE

    def move_towards_nutrient(self, distance_map, next_step):
        """
        Moves ONE STEP closer to the nearest nutrient, 
        using precomputed BFS data (distance_map, next_step).
        If no nutrient is reachable (distance == INF_DISTANCE),
        the bacterium moves randomly.
        """
        if distance_map[self.x, self.y] >= INF_DISTANCE:
            # No valid source
            self.random_move()
            return

        dx, dy = next_step[self.x, self.y]
        self.x = (self.x + dx) % GRID_SIZE
        self.y = (self.y + dy) % GRID_SIZE

    def move(self, distance_map, next_step):
        """
        Decide whether to move toward nearest nutrient or move randomly,
        then increment life steps.
        """
        if rd.random() < MOVE_TOWARD_NUTRIENT_PROB:
            self.move_towards_nutrient(distance_map, next_step)
        else:
            self.random_move()
        self.steps += 1

    def check_death(self):
        """
        Returns True if bacterium dies (only eligible after MAX_STEPS).
        """
        if self.steps > MAX_STEPS:
            return (rd.random() < DEATH_PROBABILITY)
        return False

    def reproduce(self, new_bacteria):
        """
        With probability REPRODUCE_PROBABILITY, spawns a new bacterium
        in a neighboring cell (including diagonals).
        """
        if rd.random() < REPRODUCE_PROBABILITY:
            nx = (self.x + rd.choice([-1, 0, 1])) % GRID_SIZE
            ny = (self.y + rd.choice([-1, 0, 1])) % GRID_SIZE
            new_bacteria.append(Bacterium(nx, ny))

# ---------------------------
# Multi-Source BFS Initialization
# ---------------------------
def multi_source_bfs(grid):
    """
    Computes a BFS from all nutrient cells simultaneously, setting:
      - distance_map[x, y]: Manhattan distance to the nearest nutrient.
      - next_step[x, y]: direction (dx, dy) to move from (x, y) one step 
        closer to that nearest nutrient.
      - nearest_source[x, y]: the (sx, sy) of the nutrient cell that 
        is the BFS "owner" of (x, y).

    Complexity: O(GRID_SIZE^2)

    Returns:
        distance_map, next_step, nearest_source
    """
    distance_map = np.full((GRID_SIZE, GRID_SIZE), INF_DISTANCE, dtype=np.int32)
    next_step = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.int32)
    nearest_source = -np.ones((GRID_SIZE, GRID_SIZE, 2), dtype=np.int32)

    queue = deque()

    # Enqueue all nutrient cells as BFS sources
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] > 0:
                distance_map[x, y] = 0
                nearest_source[x, y] = (x, y)
                queue.append((x, y))
                # No movement needed if you're on a nutrient cell
                next_step[x, y] = (0, 0)

    while queue:
        cx, cy = queue.popleft()
        cd = distance_map[cx, cy]
        sx, sy = nearest_source[cx, cy]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = (cx + dx) % GRID_SIZE
            ny = (cy + dy) % GRID_SIZE
            if distance_map[nx, ny] > cd + 1:
                distance_map[nx, ny] = cd + 1
                nearest_source[nx, ny] = (sx, sy)
                queue.append((nx, ny))
                # Next step for (nx, ny) is how to move one step 
                # closer to the source (sx, sy). Because BFS is from 
                # the source outward, if we stepped from (nx, ny) to (cx, cy),
                # the direction from (nx, ny) -> (cx, cy) is simply (dx, dy).
                # But we want the direction to reduce distance, so we store (dx, dy).
                next_step[nx, ny, 0] = dx * -1
                next_step[nx, ny, 1] = dy * -1

    return distance_map, next_step, nearest_source


def partial_bfs_remove_source(
    sx, sy, grid, distance_map, next_step, nearest_source
):
    """
    When a nutrient cell (sx, sy) is fully consumed (i.e., grid[sx, sy] -> 0),
    we remove it as a BFS source. We must "invalidate" all cells that 
    used (sx, sy) as their nearest source, setting their distance to INF, 
    then attempt to find if they can adopt a different nutrient source 
    through a local BFS wave.

    Approach:
      1. Collect all cells for which nearest_source[x, y] == (sx, sy).
      2. Set distance_map[x, y] = INF, next_step[x, y] = (0, 0).
      3. Enqueue all these "invalidated" cells.
      4. BFS outward from them, checking neighbors' distance. If a neighbor 
         has a valid source (some other (ux, uy) != (sx, sy)), we adopt 
         neighbor's distance + 1, nearest_source, and next_step accordingly.
      5. Continue until no more improvements.

    This partial BFS is less expensive than a full BFS from scratch, 
    but it's more complex. It ensures cells that lost their source 
    can potentially latch on to other nearby sources.
    """
    # 1. Find all cells referencing (sx, sy)
    to_invalidate = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            src_x, src_y = nearest_source[x, y]
            if (src_x == sx) and (src_y == sy):
                to_invalidate.append((x, y))

    if not to_invalidate:
        return  # No cells used this source, nothing to do

    queue = deque()
    # 2. Invalidate them
    for (x, y) in to_invalidate:
        distance_map[x, y] = INF_DISTANCE
        next_step[x, y] = (0, 0)
        nearest_source[x, y] = (-1, -1)
        queue.append((x, y))

    # 3. BFS wave from these invalidated cells
    while queue:
        cx, cy = queue.popleft()

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = (cx + dx) % GRID_SIZE
            ny = (cy + dy) % GRID_SIZE

            # If neighbor is valid, i.e., it has a real source and distance < INF
            # Then (cx, cy) can adopt that neighbor's source with +1 distance 
            nd = distance_map[nx, ny]
            if nd < INF_DISTANCE:
                # new possible distance is nd + 1
                new_dist = nd + 1
                if distance_map[cx, cy] > new_dist:
                    distance_map[cx, cy] = new_dist
                    nearest_source[cx, cy] = nearest_source[nx, ny]
                    # next_step for (cx, cy) is the direction from (cx, cy) to (nx, ny)
                    # but reversed, so that moving from (cx, cy) -> (nx, ny) lowers distance.
                    # BFS was originally from source outward, so we do:
                    # (cx, cy)->(nx, ny) = (dx, dy). But we want one step in the direction 
                    # that leads from (cx, cy) to the source. That direction is -dx, -dy
                    next_step[cx, cy, 0] = -dx
                    next_step[cx, cy, 1] = -dy
                    queue.append((cx, cy))


def initialize_grid():
    """
    Create the grid with random nutrient cells.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    for _ in range(NUM_NUTRIENT):
        x = rd.randint(0, GRID_SIZE - 1)
        y = rd.randint(0, GRID_SIZE - 1)
        grid[x, y] = BITES_NUTRIENT
    return grid


def run_simulation():
    """
    Runs a single simulation using multi-source BFS for initial distances,
    then partial BFS updates whenever a nutrient cell is fully consumed.
    """
    grid = initialize_grid()

    # Build BFS distance data
    distance_map, next_step, nearest_source = multi_source_bfs(grid)

    # Initialize bacteria
    bacteria_list = [
        Bacterium(rd.randint(0, GRID_SIZE - 1), rd.randint(0, GRID_SIZE - 1))
        for _ in range(INITIAL_NUM_BACTERIA)
    ]

    population_history = []

    while bacteria_list:
        population_history.append(len(bacteria_list))
        new_bacteria_list = []

        for bacterium in bacteria_list:
            # Move
            bacterium.move(distance_map, next_step)

            # Check death
            if bacterium.check_death():
                continue

            # If there's nutrient at current location, consume 1 bite
            if grid[bacterium.x, bacterium.y] > 0:
                grid[bacterium.x, bacterium.y] -= 1

                # If it's now fully consumed, do partial BFS to remove that source
                if grid[bacterium.x, bacterium.y] <= 0:
                    sx, sy = bacterium.x, bacterium.y
                    # We "remove" it as a BFS source
                    partial_bfs_remove_source(
                        sx, sy, grid, distance_map, next_step, nearest_source
                    )

                # Attempt reproduction
                bacterium.reproduce(new_bacteria_list)

            new_bacteria_list.append(bacterium)

        bacteria_list = new_bacteria_list

    return population_history


@profile
def run_multiple_simulations(num_sims):
    """
    Runs multiple simulations, measuring memory usage with memory_profiler.
    """
    results = []
    for _ in range(num_sims):
        pop_history = run_simulation()
        results.append(pop_history)
    return results


def plot_results(simulations):
    """
    Plots each simulation's population history.
    """
    plt.figure(figsize=(10, 6))
    for i, sim_data in enumerate(simulations, start=1):
        plt.plot(sim_data, label=f"Simulation {i}")
    plt.title("Bacteria Population Over Time (Partial BFS Updates)")
    plt.xlabel("Step")
    plt.ylabel("Number of Bacteria")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    all_sims = run_multiple_simulations(NUM_SIMULATIONS)
    plot_results(all_sims)
