from __future__ import annotations

from typing import Callable, Set

import sympy as sp

from dreamer.extraction.shard import Shard
from dreamer.utils.rand import np
from ramanujantools import Position

from dreamer.extraction.samplers.raycast_sampler import RaycastPipelineSampler
from dreamer.extraction.sampling_orchestrators.sampling_orchestrator import SamplingOrchestrator
from dreamer.extraction.samplers.sphere_sampler import PrimitiveSphereSampler


class ShardSamplingOrchestrator(SamplingOrchestrator):
    """Trajectory sampler for shards using the extraction sampling pipeline."""
    def __init__(self, searchable: "Shard"):
        super().__init__(searchable)
        if not isinstance(self.searchable, Shard):
            raise ValueError(f"{self.__class__.__name__} can only be used with {Shard.__name__} objects.")

    def sample_trajectories(self, compute_n_samples: Callable[[int], int], *, exact: bool = False) -> Set[Position]:
        a_matrix = getattr(self.searchable, "A", None)
        symbols = self.searchable.symbols
        if a_matrix is None:
            sampler = PrimitiveSphereSampler(len(symbols))
        else:
            sampler = RaycastPipelineSampler(np.asarray(a_matrix, dtype=np.float64))
        samples = sampler.harvest(compute_n_samples, exact=exact)

        return {
            Position({sym: sp.sympify(int(v)) for v, sym in zip(p, symbols)})
            for p in samples
        }
