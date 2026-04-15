from dreamer.utils.rand import np
import scipy.special as spc
from dreamer.extraction.samplers.conditioner import HyperSpaceConditioner
from dreamer.extraction.samplers.raycaster import RayCastingSamplingMethod
from dreamer.utils.logger import Logger
from dreamer.configs.search import search_config
from .sampler import Sampler
from typing import Callable, cast
import math



class RaycastPipelineSampler(Sampler):
    def __init__(self, A_prime):
        self.A_prime = A_prime
        self.d_orig: int = int(A_prime.shape[1])

        Logger("Initializing Sampler: Conditioning...", Logger.Levels.debug).log()
        conditioner = HyperSpaceConditioner(self.A_prime, max_beta=10, defect_tolerance=5.0)
        self.Z_reduced, self.B_reduced, _ = conditioner.process()
        self.d_flat = int(self.Z_reduced.shape[1])
        self.fraction = float(self._estimate_cone_fraction(self.B_reduced, self.d_flat))
        Logger(
            f"Shard Estimated Volume: {self.fraction * 100:.6f}%",
            Logger.Levels.debug
        ).log()

        super().__init__(self.d_flat)

    @staticmethod
    def _estimate_cone_fraction(B: np.ndarray, d_flat: int, samples: int = 100_000) -> float:
        """
        Gaussian Measure Dart Throw.
        :param B: Bounds matrix
        :param d_flat: dimension of the flatland space
        :param samples: number of samples - darts to throw.
        """
        if len(B) == 0:
            return 1.0

        # Pure Gaussian distribution (Spherically symmetric)
        darts = np.random.randn(samples, d_flat)
        darts /= np.linalg.norm(darts, axis=1, keepdims=True)

        valid_count = 0
        for i in range(samples):
            if np.max(B @ darts[i]) <= 1e-9:
                valid_count += 1

        # If the cone is a severe needle, give it a baseline epsilon so our math doesn't divide by zero
        fraction = max(valid_count / samples, 1e-7)
        return fraction

    @staticmethod
    def _calculate_R_max(target_quota: int, fraction: float, d_flat: int) -> float:
        """
        Calculates the theoretical radius needed to hold the quota.
        :param target_quota: target quota of points to sample
        :param fraction: fraction of points to sample
        :param d_flat: dimension of the flatland space
        :return R_max: theoretical radius of the hypersphere containing the target quota samples.
        """
        numerator = target_quota * spc.gamma((d_flat / 2.0) + 1)
        denominator = fraction * (np.pi ** (d_flat / 2.0))
        R_max_d = numerator / denominator
        R_max = R_max_d ** (1.0 / d_flat)
        return R_max

    @staticmethod
    def _verify_uniformity(rays, fraction: float, d_flat: int) -> None:
        """
        Check the generated rays angular uniformity.
        :param rays: The generated rays
        :param fraction: The fraction of space the cone takes
        :param d_flat: The dimension of the sample space
        """
        if len(rays) < 2:
            return

        # 1. Calculate the dynamic theoretical threshold
        surface_dim = max(1.0, float(d_flat - 1))
        safe_fraction = max(1e-12, fraction)
        theoretical_gap = 180.0 * ((safe_fraction / len(rays)) ** (1.0 / surface_dim))

        # Threshold is 50% of the mathematical ideal
        threshold_degrees = theoretical_gap * 0.5

        sample_size = min(2000, len(rays))
        sample = rays[np.random.choice(len(rays), sample_size, replace=False)]

        # Normalize and compute cosine similarity matrix
        norms = np.linalg.norm(sample, axis=1, keepdims=True)
        normalized = sample / np.clip(norms, 1e-9, None)
        cos_sim = np.asarray(np.clip(normalized @ normalized.T, -1.0, 1.0), dtype=np.float64)

        # Ignore self-similarity (diagonal)
        diag = np.arange(cos_sim.shape[0])
        cos_sim[diag, diag] = -1.0
        max_sim = np.max(cos_sim, axis=1)

        # Convert to degrees
        min_angles = np.arccos(max_sim) * (180.0 / np.pi)
        median_gap = np.median(min_angles)
        mean_gap = np.mean(min_angles)

        success = True
        if median_gap < threshold_degrees:
            Logger(
                f"⚠ WARNING: Severe angular clustering detected. Median NN gap: {median_gap:.2f}°",
                Logger.Levels.debug
            ).log()
            success = False
        else:
            Logger(
                f"Uniformity Check Passed: Healthy angular separation. Median NN gap: {median_gap:.2f}°",
                Logger.Levels.debug
            ).log()

        if mean_gap < threshold_degrees:
            Logger(
                f"⚠ WARNING: Severe angular clustering detected. Mean NN gap: {mean_gap:.2f}°",
                Logger.Levels.debug
            ).log()
            success = False
        else:
            Logger(
                f"Uniformity Check Passed: Healthy angular separation. Mean NN gap: {mean_gap:.2f}°",
                Logger.Levels.debug
            ).log()

        if not success:
            Logger(
                "Could not preform uniform sampling as expected... (if this repeats many times please report)",
                Logger.Levels.warning
            ).log()

    def harvest(
        self,
        target_func: Callable[[int], int] | int,
        guidance_method: str = 'mcmc',
        exact: bool = False,
    ) -> np.ndarray:
        """
        Harvest samples
        :param target_func: Target function to compute total expected quota
        :param guidance_method: Ray sampling guidance method - MCMC or MHS
        :param exact: If true and target_func is callable, enforce exactly target_func(d_flat) rays.
        :return: The samples
        """
        Z_reduced = self.Z_reduced
        B_reduced = self.B_reduced
        d_flat = self.d_flat
        fraction = self.fraction

        requested_rays: int
        if isinstance(target_func, int):
            requested_rays = target_func
            target_rays = requested_rays
        else:
            compute_quota = cast(Callable[[int], int], target_func)
            requested_rays = int(compute_quota(d_flat))
            if exact:
                target_rays = requested_rays
            else:
                amount_safety = 1.05
                target_rays = max(int(requested_rays * fraction * amount_safety), 5)

        if target_rays <= 0:
            return np.empty((0, self.d_orig), dtype=np.int64)

        R_max = self._calculate_R_max(target_rays, fraction, d_flat)
        Logger(
            f"[Pipeline] Mathematical R_max needed for {target_rays} rays: {R_max:.2f}",
            Logger.Levels.debug
        ).log()
        Logger("[Pipeline] Initializing Stage 2: Universal Raycaster...", Logger.Levels.debug).log()
        sampler = RayCastingSamplingMethod(Z_reduced, B_reduced, self.d_orig, guidance_method)

        # In exact mode we keep the internal shoot quota strict.
        guide_rays_to_shoot = int(target_rays * 3)
        current_R_max = R_max * 1.05
        final_rays = np.empty((0, self.d_orig), dtype=np.int64)

        if d_flat >= 4:
            # Massive outer shell. strictly enforce Fair Slice (1 point per ray)
            dynamic_max_per_ray = 1
        else:
            # Microscopic outer shell. Must penetrate deep to fill quota.
            dynamic_max_per_ray = max(1, int(1.5 * (target_rays ** (1.0 / d_flat))))
            Logger(
                f"[Pipeline] Low-D Space Detected. Allowing depth penetration: max_per_ray={dynamic_max_per_ray}",
                Logger.Levels.debug
            ).log()

        def finalize_rays(raw_rays, target_rays):
            lengths = np.linalg.norm(raw_rays, axis=1)
            sorted_indices = np.argsort(lengths)
            best_rays = raw_rays[sorted_indices][:target_rays]
            np.random.shuffle(best_rays)
            final_rays = best_rays
            return final_rays

        max_radius = math.sqrt(pow(search_config.MAX_TRAJECTORY_COORD, 2) * d_flat) + 1
        raw_rays = np.array([])

        raddai = []
        expansions = []

        while len(final_rays) < target_rays:
            if current_R_max >= max_radius:
                final_rays = finalize_rays(raw_rays, len(raw_rays))
                Logger(
                    f"[Pipeline] Could not achieve quota, found {len(final_rays)}/{target_rays}",
                    Logger.Levels.debug
                ).log()
                break

            raw_rays = sampler.harvest(
                target_rays=guide_rays_to_shoot,
                R_max=current_R_max,
                max_per_ray=dynamic_max_per_ray
            )

            if len(raw_rays) >= target_rays:
                Logger(f"[Pipeline] Quota exceeded ({len(raw_rays)})!", Logger.Levels.debug).log()
                final_rays = finalize_rays(raw_rays, target_rays)
                break
            else:
                if len(raw_rays) == 0:
                    momentum_multiplier = 2.0
                else:
                    # Dimensional scaling law: V_new / V_old = R_multiplier ^ d_flat
                    ratio_needed = target_rays / len(raw_rays)
                    momentum_multiplier = ratio_needed ** (1.0 / d_flat)

                # Cap the multiplier between 1.10 (minimum safety step) and 3.0 (max jump)
                multiplier = np.clip(momentum_multiplier, 1.10, 3.0)
                expansions.append(multiplier)
                raddai.append(current_R_max)
                current_R_max *= multiplier

        multipliers = [f'{radius:.2f} by {multiplier:.3f}' for multiplier, radius in zip(expansions, raddai)]
        multipliers = ', '.join(multipliers)
        if multipliers:
            Logger(f"\tMomentum Expansion up to {current_R_max:.2f}: {multipliers}", Logger.Levels.debug).log()
        else:
            Logger(f"\tSearch radius used: {current_R_max:.2f}", Logger.Levels.debug)

        self._verify_uniformity(final_rays, fraction, d_flat)
        return final_rays
