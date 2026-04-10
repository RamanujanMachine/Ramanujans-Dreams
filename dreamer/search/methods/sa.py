import math
import random
import numpy as np
from typing import List, Optional, cast

from dreamer.utils.schemes.searcher_scheme import SearchMethod
from dreamer.utils.storage.storage_objects import DataManager, SearchVector
from dreamer.utils.schemes.searchable import Searchable
from dreamer.extraction.samplers import ShardSamplingOrchestator
from dreamer.extraction.samplers.conditioner import HyperSpaceConditioner
from dreamer.utils.logger import Logger
from dreamer.utils.multi_processing import create_pool
from ramanujantools import Position


class SimulatedAnnealingSearchMethod(SearchMethod):
    def __init__(self,
                 space: Searchable,
                 constant,
                 iterations: int = 100,
                 max_res: int = 10,
                 cores: int = 1,
                 t0: float = 1.0,
                 tmin: float = 1e-3,
                 schedule_type: str = "linear",
                 data_manager: DataManager = None,
                 share_data: bool = True,
                 use_LIReC: bool = True):
        super().__init__(space, constant, use_LIReC, data_manager, share_data)

        if self.data_manager is None:
            self.data_manager = DataManager(use_LIReC=self.use_LIReC)

        self.iterations = iterations
        self.max_res = max_res
        self.cores = cores
        self.t0 = t0
        self.tmin = tmin
        self.schedule_type = schedule_type

        self.symbols = None
        self.Z = None
        self.dim_flat = None

    def _setup_flatland(self):
        self.symbols = list(self.space.symbols)
        dim_orig = len(self.symbols)

        a_matrix = getattr(self.space, 'A', None)
        if getattr(self.space, 'is_whole_space', True) or a_matrix is None:
            self.Z = np.eye(dim_orig, dtype=np.int64)
            Logger("SA Setup: Unconstrained space, using Identity basis.", Logger.Levels.debug).log()
        else:
            conditioner = HyperSpaceConditioner(a_matrix)
            self.Z, _, _ = conditioner.process()
            Logger(f"SA Setup: Constrained space conditioned. Flatland dim: {self.Z.shape[1]}",
                   Logger.Levels.debug).log()

        self.dim_flat = self.Z.shape[1]

    def _to_flatland(self, traj_orig: Position) -> np.ndarray:
        v_orig = np.array([traj_orig[sym] for sym in self.symbols], dtype=float)
        v_flat, _, _, _ = np.linalg.lstsq(self.Z, v_orig, rcond=None)
        return np.round(v_flat).astype(int)

    def _to_original(self, v_flat: np.ndarray) -> Position:
        v_orig = self.Z @ v_flat
        return Position({sym: int(round(val)) for sym, val in zip(self.symbols, v_orig)})

    def _get_temp(self, k: int) -> float:
        if self.schedule_type == "log":
            return self.t0 / math.log(k + 1 + 1e-5)
        return self.t0 / (k + 1)

    def _get_neighbors_flatland(self, cur_traj_flat: np.ndarray, start_point: Position, num_samples: int = 10) -> List[
        np.ndarray]:
        neighbors_flat = []
        max_attempts = num_samples * 20

        for _ in range(max_attempts):
            if len(neighbors_flat) >= num_samples:
                break

            # --- THE HIGH-RESOLUTION LATTICE WALK ---
            # 1. Scale up to increase angular resolution (larger scale = smaller angular shift)
            # We use a random scale between 1 and 10 to explore both big and small angle changes
            scale = random.randint(1, 10)
            new_v_flat = cur_traj_flat * scale

            # 2. Perturb the scaled vector
            # Pick a random dimension in the flatland basis
            ax = random.randint(0, self.dim_flat - 1)

            # Take a small discrete step
            step = random.choice([-1, 1]) * random.randint(1, 2)
            new_v_flat[ax] += step

            # 3. Simplify the Ray (Vector Minimization)
            # A ray of [20, 20] is mathematically the same direction as [1, 1].
            # We divide by the Greatest Common Divisor to keep the integers manageable.
            ray_gcd = np.gcd.reduce(np.abs(new_v_flat))
            if ray_gcd > 1:
                new_v_flat = new_v_flat // ray_gcd

            # Prevent collapsing into the zero vector or testing the exact same ray
            if np.all(new_v_flat == 0) or np.array_equal(new_v_flat, cur_traj_flat):
                continue

            # Map the flatland vector back to the original symbol space
            new_traj_orig = self._to_original(new_v_flat)

            # 4. Strict Cone Check: Does the ray point in a valid direction? (A * v <= 0)
            if getattr(self.space, 'is_valid_trajectory', lambda t: True)(new_traj_orig):

                # 5. Start Point Check: Does taking a step from the start point stay inside?
                test_pos = Position({sym: start_point[sym] + new_traj_orig[sym] for sym in self.symbols})
                if getattr(self.space, 'in_space', lambda p: True)(test_pos):

                    # Ensure uniqueness in this batch
                    if not any(np.array_equal(new_v_flat, n) for n in neighbors_flat):
                        neighbors_flat.append(new_v_flat)

        if len(neighbors_flat) < num_samples:
            Logger(
                f"SA Warning: Found {len(neighbors_flat)}/{num_samples} valid neighbors. "
                "Cone is extremely narrow, but sampling is proceeding.",
                Logger.Levels.debug
            ).log()

        return neighbors_flat

    def _evaluate_trajectory(self, traj_flat: np.ndarray, start: Position) -> dict:
        traj_orig = self._to_original(traj_flat)
        sv = SearchVector(start, traj_orig)

        if sv in self.data_manager:
            sd = self.data_manager[sv]
        else:
            sd = self.space.compute_trajectory_data(
                traj=traj_orig, start=start, use_LIReC=self.use_LIReC
            )

        delta_val = -1.0 if sd.delta is None or isinstance(sd.delta, str) else float(sd.delta)
        return {"traj_flat": traj_flat, "traj_orig": traj_orig, "delta": delta_val, "sd": sd, "sv": sv}

    def search(self, starts: Optional[Position | List[Position]] = None) -> DataManager:
        Logger(f"SA: Starting search with {self.iterations} iterations.", Logger.Levels.info).log()
        self._setup_flatland()

        if starts is None:
            start_point = getattr(self.space, 'get_interior_point', lambda: None)()
        elif isinstance(starts, list):
            start_point = starts[0]
        else:
            start_point = starts

        if start_point is None:
            raise ValueError("SA requires a valid start point")
        start_point = cast(Position, start_point)

        # Draw a valid initial trajectory directly from the shard sampler.
        sampler = ShardSamplingOrchestator(self.space)
        samples = list(sampler.sample_trajectories(lambda dim: 10))
        cur_traj_orig = samples[0] if samples else Position({s: 1 for s in self.symbols})
        cur_traj_flat = self._to_flatland(cur_traj_orig)

        T = self.t0
        iter_left = self.iterations
        traj_mul = 1

        initial_eval = self._evaluate_trajectory(cur_traj_flat, start_point)
        cur_delta = initial_eval["delta"]
        self.data_manager[initial_eval["sv"]] = initial_eval["sd"]
        best_res = cur_delta

        with create_pool() as pool:
            while iter_left > 0 and T > self.tmin:
                neighs_flat = self._get_neighbors_flatland(cur_traj_flat, start_point)

                if not neighs_flat:
                    Logger("SA: Premature exit. No valid neighbors could be found.", Logger.Levels.warning).log()
                    break

                starts_iterable = [start_point] * len(neighs_flat)
                neighs_results = list(pool.map(self._evaluate_trajectory, neighs_flat, starts_iterable))

                for res in neighs_results:
                    self.data_manager[res["sv"]] = res["sd"]

                neighs_results.sort(key=lambda d: d["delta"], reverse=True)
                best_neighbor = neighs_results[0]
                new_delta = best_neighbor["delta"]
                accepted = False

                if new_delta >= cur_delta:
                    cur_traj_flat = best_neighbor["traj_flat"]
                    cur_delta = new_delta
                    accepted = True
                else:
                    diff = new_delta - cur_delta
                    prob = math.exp(diff / T)
                    if random.random() < prob:
                        cur_traj_flat = best_neighbor["traj_flat"]
                        cur_delta = new_delta
                        accepted = True

                if not accepted:
                    cur_traj_flat = cur_traj_flat * 2
                    if traj_mul > self.max_res:
                        Logger("SA: Max scaling reached. Resetting to new random ray.", Logger.Levels.warning).log()
                        traj_mul = 1
                        # Reset to a new random valid trajectory to escape the local minimum trap.
                        fallback_samples = list(sampler.sample_trajectories(lambda dim: 5))
                        if fallback_samples:
                            cur_traj_flat = self._to_flatland(random.choice(fallback_samples))
                    else:
                        traj_mul += 1
                else:
                    traj_mul = 1

                if cur_delta > best_res:
                    best_res = cur_delta

                T = self._get_temp(self.iterations - iter_left)

        Logger(f"SA: Search complete. Best Delta Found: {best_res:.4f}", Logger.Levels.info).log()
        return self.data_manager
