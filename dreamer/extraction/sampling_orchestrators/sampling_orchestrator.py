from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Set, TYPE_CHECKING

from ramanujantools import Position

if TYPE_CHECKING:
    from dreamer.utils.schemes.searchable import Searchable


class SamplingOrchestrator(ABC):
    """Abstract trajectory sampler bound to a searchable space."""

    def __init__(self, searchable: "Searchable"):
        self.searchable = searchable

    @abstractmethod
    def sample_trajectories(self, compute_n_samples: Callable[[int], int]) -> Set[Position]:
        """Sample valid trajectories for the owning searchable."""
        raise NotImplementedError()
