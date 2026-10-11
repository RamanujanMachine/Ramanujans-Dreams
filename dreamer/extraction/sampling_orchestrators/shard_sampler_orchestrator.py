from __future__ import annotations

from typing import Callable, List, Set
import sympy as sp

from dreamer.extraction.shard import Shard
from dreamer.utils.caching import cached_property
from dreamer.utils.rand import np, derive_seed
from ramanujantools import Position

from dreamer.configs.search import search_config
from dreamer.extraction.samplers.raycast_sampler import RaycastPipelineSampler
from dreamer.extraction.samplers.discrete_raycaster import DiscreteMCMCSampler
from dreamer.extraction.samplers.parallel_tempering_raycaster import ParallelTemperingSampler
from dreamer.extraction.sampling_orchestrators.sampling_orchestrator import SamplingOrchestrator
from dreamer.extraction.samplers.sphere_sampler import PrimitiveSphereSampler


def _build_trajectory_sampler(a_matrix: np.ndarray, method: str | None = None, *, seed: int = -1):
    """Construct the trajectory sampler for the requested (or configured) method.

    The ``discrete`` / ``pt`` lattice walkers harvest primitive integer directions whose
    original-space norm stays within ``search_config.MAX_TRAJECTORY_LENGTH`` (the same
    usable-length bound the raycast pipeline filters to), so the choice is transparent to
    callers — all three return an ``(n, d_orig)`` integer array from ``harvest``.

    :param a_matrix: ``(rows, d_orig)`` constraint matrix of the shard.
    :param method: explicit engine name (``raycast`` / ``discrete`` / ``pt``); when
        ``None`` the stage-default ``search_config.SAMPLING_METHOD`` is used.  The
        analysis stage passes ``analysis_config.SAMPLING_METHOD`` here so it can differ.
    :param seed: per-(shard, method) RNG seed derived from ``search_config.GLOBAL_SEED``
        (see :func:`dreamer.utils.rand.derive_seed`); ``< 0`` disables seeding
        (nondeterministic), used when the master seed is ``None``.
    :return: a constructed :class:`Sampler` for the chosen method.
    :raises ValueError: if ``method`` is not one of ``raycast`` / ``discrete`` / ``pt``.
    """
    method = method if method is not None else search_config.SAMPLING_METHOD
    useful_norm = float(search_config.MAX_TRAJECTORY_LENGTH)
    if method == "raycast":
        return RaycastPipelineSampler(a_matrix, seed=seed)
    if method == "discrete":
        return DiscreteMCMCSampler(a_matrix, max_useful_norm=useful_norm, rng_seed=seed)
    if method == "pt":
        return ParallelTemperingSampler(a_matrix, max_useful_norm=useful_norm, rng_seed=seed)
    raise ValueError(
        f"Unknown SAMPLING_METHOD '{method}'. Expected 'raycast', 'discrete', or 'pt'."
    )


class ShardSamplingOrchestrator(SamplingOrchestrator):
    """Trajectory sampler for shards using the extraction sampling pipeline.

    The concrete trajectory-sampling engine is selected by
    ``search_config.SAMPLING_METHOD`` (``raycast`` / ``discrete`` / ``pt``), or by an
    explicit ``sampling_method`` override (used by the analysis stage to pass
    ``analysis_config.SAMPLING_METHOD``).
    """
    def __init__(self, searchable: Shard, *, sampling_method: str | None = None):
        """
        :param searchable: the :class:`Shard` to sample trajectories for.
        :param sampling_method: optional engine override (``raycast`` / ``discrete`` /
            ``pt``); when ``None`` the search-stage default
            ``search_config.SAMPLING_METHOD`` is used.
        """
        super().__init__(searchable)
        if not isinstance(self.searchable, Shard):
            raise ValueError(f"{self.__class__.__name__} can only be used with {Shard.__name__} objects.")

        # Per-(shard, method) reproducible seed: same GLOBAL_SEED + shard + method
        # always yields the same trajectory sample, while distinct shards/methods get
        # independent streams.  ``derive_seed`` returns a nondeterministic seed when
        # ``search_config.GLOBAL_SEED is None``.
        method = sampling_method if sampling_method is not None else search_config.SAMPLING_METHOD
        from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids
        _, shard_id, _ = derive_cmf_and_shard_ids(self.searchable)
        seed = derive_seed(shard_id, method)

        # Direction constraints (e.g. {'x0': 12, 'y1': 28}) are folded into the cone as
        # extra homogeneous rows so every sampler honours them with no kernel change; the
        # strict v_i != 0 / sign part is applied as a post-harvest mask (``self._fixed``).
        from dreamer.extraction.samplers.constraints import (
            augment_cone,
            get_trajectory_constraints,
        )
        constraints = get_trajectory_constraints()
        a_matrix, self._fixed = augment_cone(
            self.searchable.A, self.searchable.symbols, constraints
        )

        if a_matrix is None:
            self.sampler = PrimitiveSphereSampler(len(self.searchable.symbols), seed=seed)
        else:
            self.sampler = _build_trajectory_sampler(
                np.asarray(a_matrix, dtype=np.float64), sampling_method, seed=seed
            )

    def sample_trajectories(self, compute_n_samples: Callable[[int], int] | int, *, exact: bool = False) -> List[Position]:
        # Local imports avoid a circular dependency (logger/trajectory_attributes
        # pull in extraction modules at import time).
        import time
        from dreamer.utils.logger import Logger
        from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids

        _, shard_id, _ = derive_cmf_and_shard_ids(self.searchable)
        sampler_name = type(self.sampler).__name__
        Logger(
            f"Starting trajectory sampling in shard {shard_id} via {sampler_name}",
            Logger.Levels.debug,
        ).log()
        t0 = time.perf_counter()

        if isinstance(self.sampler, PrimitiveSphereSampler):
            samples = self.sampler.harvest(compute_n_samples)
        else:
            samples = self.sampler.harvest(compute_n_samples, exact=exact)

        # Enforce the strict fixed-coordinate sign/non-zero rule the closed cone admits on
        # its facet (no-op when no direction constraints are configured).
        if self._fixed and len(samples) > 0:
            from dreamer.extraction.samplers.constraints import fixed_sign_mask

            samples = np.asarray(samples)
            samples = samples[fixed_sign_mask(samples, self._fixed)]

        # Deduplicate while preserving the sampler's harvest order (which is
        # deterministic for a fixed seed).  Returning a *list* — not a set — is
        # essential for reproducibility: ``Position``'s hash is process-dependent
        # (it hashes through symbol names, which Python salts per process via
        # PYTHONHASHSEED), so set-iteration order differs every run.  Downstream
        # seed-vector selection (e.g. the search reservoirs) sorts these and picks
        # the first match, so a nondeterministic order would silently pick a
        # different seed each run.  The dedup key is the integer coordinate tuple
        # (content-based, order-independent), matching the old set's semantics.
        seen: Set[tuple] = set()
        result: List[Position] = []
        for p in samples:
            key = tuple(int(v) for v in p)
            if key in seen:
                continue
            seen.add(key)
            result.append(
                Position({sym: sp.sympify(int(v)) for v, sym in zip(p, self.searchable.symbols)})
            )
        Logger(
            f"Finished sampling {len(result)} trajectories in shard {shard_id} "
            f"in {time.perf_counter() - t0:.1f}s",
            Logger.Levels.debug,
        ).log()
        return result

    @cached_property
    def search_space_dim(self):
        return self.sampler.d
