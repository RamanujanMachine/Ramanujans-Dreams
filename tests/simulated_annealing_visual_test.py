import matplotlib.pyplot as plt
import numpy as np
import copy
from typing import Optional, List

from ramanujantools import Position
from dreamer.search.methods.sa import SimulatedAnnealingSearchMethod
from dreamer.utils.schemes.searchable import Searchable


class VisualSimulatedAnnealing(SimulatedAnnealingSearchMethod):
    """A wrapper that tracks the accepted trajectory history for visualization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self.best_history = []

    def search(self, starts: Optional[Position | List[Position]] = None):
        # We hook into the search loop by overriding it slightly to track states
        self._setup_flatland()

        start_point = getattr(self.space, 'get_interior_point', lambda: None)()
        samples = list(self.space.sample_trajectories(lambda dim: 10))
        cur_traj_orig = samples[0] if samples else Position({s: 1 for s in self.symbols})
        cur_traj_flat = self._to_flatland(cur_traj_orig)

        T = self.t0
        iter_left = self.iterations

        # Track initial
        initial_eval = self._evaluate_trajectory(cur_traj_flat, start_point)
        cur_delta = initial_eval["delta"]
        best_res = cur_delta

        self.history.append(copy.deepcopy(initial_eval["traj_orig"]))
        self.best_history.append(cur_delta)

        # Standard Loop with tracking
        while iter_left > 0 and T > self.tmin:
            neighs_flat = self._get_neighbors_flatland(cur_traj_flat, start_point)
            if not neighs_flat:
                break

            # For testing, we just use a standard list comprehension instead of parallel mapping
            # to keep the visualizer standalone and simple
            neighs_results = [self._evaluate_trajectory(n, start_point) for n in neighs_flat]
            neighs_results.sort(key=lambda d: d["delta"], reverse=True)

            best_neighbor = neighs_results[0]
            new_delta = best_neighbor["delta"]
            accepted = False

            import random
            import math
            if new_delta >= cur_delta:
                cur_traj_flat = best_neighbor["traj_flat"]
                cur_delta = new_delta
                accepted = True
                iter_left -= 1
            else:
                diff = new_delta - cur_delta
                prob = math.exp(diff / T)
                if random.random() < prob:
                    cur_traj_flat = best_neighbor["traj_flat"]
                    cur_delta = new_delta
                    accepted = True
                    iter_left -= 1

            if not accepted:
                cur_traj_flat = cur_traj_flat * 2

            if cur_delta > best_res:
                best_res = cur_delta

            # Record state
            self.history.append(self._to_original(cur_traj_flat))
            self.best_history.append(best_res)

            T = self._get_temp(self.iterations - iter_left)

        return self.data_manager


def plot_sa_walk_2d(visualizer: VisualSimulatedAnnealing):
    """Generates a 2D plot showing the trajectory exploration path."""
    if len(visualizer.symbols) != 2:
        print("Visualization is currently only supported for 2D spaces.")
        return

    sym_x, sym_y = visualizer.symbols[0], visualizer.symbols[1]

    # Extract coordinates
    x_coords = [pos[sym_x] for pos in visualizer.history]
    y_coords = [pos[sym_y] for pos in visualizer.history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Trajectory Walk ---
    ax1.plot(x_coords, y_coords, marker='o', linestyle='-', color='b', alpha=0.6, markersize=4)

    # Mark Start and End
    ax1.plot(x_coords[0], y_coords[0], marker='s', color='g', markersize=8, label='Start')
    ax1.plot(x_coords[-1], y_coords[-1], marker='X', color='r', markersize=8, label='End')

    # Add an arrow to show direction for the first few steps
    for i in range(min(5, len(x_coords) - 1)):
        ax1.annotate('', xy=(x_coords[i + 1], y_coords[i + 1]), xytext=(x_coords[i], y_coords[i]),
                     arrowprops=dict(arrowstyle="->", color='black', alpha=0.5))

    ax1.set_title("Simulated Annealing Trajectory Walk")
    ax1.set_xlabel(str(sym_x))
    ax1.set_ylabel(str(sym_y))
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # --- Plot 2: Delta Improvement Over Time ---
    ax2.plot(visualizer.best_history, color='orange', linewidth=2)
    ax2.set_title("Best Delta Found Over Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Delta")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# Example Usage:
# if __name__ == "__main__":
#     my_space = ... # Load a real Shard or CMF here
#     my_constant = ...
#     vis = VisualSimulatedAnnealing(my_space, my_constant, iterations=100)
#     vis.search()
#     plot_sa_walk_2d(vis)