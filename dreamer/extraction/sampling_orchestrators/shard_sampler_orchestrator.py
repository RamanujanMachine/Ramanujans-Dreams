from __future__ import annotations

from typing import Callable, Set
import sympy as sp

from dreamer.extraction.shard import Shard
from dreamer.utils.caching import cached_property
from dreamer.utils.rand import np
from ramanujantools import Position

from dreamer.configs.search import search_config
from dreamer.extraction.samplers.raycast_sampler import RaycastPipelineSampler
from dreamer.extraction.samplers.discrete_raycaster import DiscreteMCMCSampler
from dreamer.extraction.samplers.parallel_tempering_raycaster import ParallelTemperingSampler
from dreamer.extraction.sampling_orchestrators.sampling_orchestrator import SamplingOrchestrator
from dreamer.extraction.samplers.sphere_sampler import PrimitiveSphereSampler


def _build_trajectory_sampler(a_matrix: np.ndarray, method: str | None = None):
    """Construct the trajectory sampler for the requested (or configured) method.

    The ``discrete`` / ``pt`` lattice walkers harvest primitive integer directions whose
    original-space norm stays within ``search_config.MAX_TRAJECTORY_LENGTH`` (the same
    usable-length bound the raycast pipeline filters to), so the choice is transparent to
    callers — all three return an ``(n, d_orig)`` integer array from ``harvest``.

    :param a_matrix: ``(rows, d_orig)`` constraint matrix of the shard.
    :param method: explicit engine name (``raycast`` / ``discrete`` / ``pt``); when
        ``None`` the stage-default ``search_config.SAMPLING_METHOD`` is used.  The
        analysis stage passes ``analysis_config.SAMPLING_METHOD`` here so it can differ.
    :return: a constructed :class:`Sampler` for the chosen method.
    :raises ValueError: if ``method`` is not one of ``raycast`` / ``discrete`` / ``pt``.
    """
    method = method if method is not None else search_config.SAMPLING_METHOD
    useful_norm = float(search_config.MAX_TRAJECTORY_LENGTH)
    if method == "raycast":
        return RaycastPipelineSampler(a_matrix)
    if method == "discrete":
        return DiscreteMCMCSampler(a_matrix, max_useful_norm=useful_norm)
    if method == "pt":
        return ParallelTemperingSampler(a_matrix, max_useful_norm=useful_norm)
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

        a_matrix = self.searchable.A
        if a_matrix is None:
            self.sampler = PrimitiveSphereSampler(len(self.searchable.symbols))
        else:
            self.sampler = _build_trajectory_sampler(np.asarray(a_matrix, dtype=np.float64), sampling_method)

    def sample_trajectories(self, compute_n_samples: Callable[[int], int] | int, *, exact: bool = False) -> Set[Position]:
        if isinstance(self.sampler, PrimitiveSphereSampler):
            samples = self.sampler.harvest(compute_n_samples)
        else:
            samples = self.sampler.harvest(compute_n_samples, exact=exact)

        return {
            Position({sym: sp.sympify(int(v)) for v, sym in zip(p, self.searchable.symbols)})
            for p in samples
        }

    @cached_property
    def search_space_dim(self):
        return self.sampler.d
