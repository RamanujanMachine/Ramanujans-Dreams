from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Sampler(ABC):
    """Abstract trajectory sampler bound to a searchable space."""

    @abstractmethod
    def harvest(self, compute_n_samples: Callable[[int], int]) -> np.ndarray:
        """
        Sample valid points in the defined space.
        :param compute_n_samples: Number of points to sample as a function of the dimensionality.
        :return: The sampled points
        """
        raise NotImplementedError()
