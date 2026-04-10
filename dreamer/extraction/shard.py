from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.constants.constant import Constant
from dreamer.configs import config
from .sampler.e2e import EndToEndSamplingEngine
from ramanujantools.cmf import CMF
from ramanujantools import Position
from typing import List, Set, Optional, Callable, Tuple, Union
import sympy as sp
import numpy as np


class Shard(Searchable):
    def __init__(self,
                 cmf: CMF,
                 constant: Constant,
                 hyperplanes: List[Hyperplane],
                 encoding: List[int],
                 shift: Position,
                 interior_point: Optional[Position] = None,
                 use_inv_t: Optional[bool] = None
                 ):
        """
        :param cmf: The CMF this shard is a part of
        :param constant: The constant to search for in the shard
        :param hyperplanes: The hyperplanes defining the shard
        :param encoding: The indicator vector that indicates whether the shard is below or above the hyperplanes
        :param shift: The shift in start points required
        :param interior_point: A point within the shard
        :param use_inv_t: Whether to use inverse transpose when preforming walk or not
        """
        use_inv_t_value: bool = bool(config.search.DEFAULT_USES_INV_T if use_inv_t is None else use_inv_t)

        super().__init__(cmf, constant, shift, use_inv_t_value)
        self.symbols = list(cmf.matrices.keys())
        if not hyperplanes:
            self.A, self.b = None, None
        else:
            # Work in shifted coordinates, then translate tested points back by `shift`.
            shifted_hyperplanes = [hp.apply_shift(shift) for hp in hyperplanes]
            self.A, self.b, self.symbols = self.generate_matrices(shifted_hyperplanes, encoding)
        self.start_coord = interior_point
        self.is_whole_space = self.A is None or self.b is None

    def in_space(self, point: Position) -> bool:
        """
        Checks if a point is inside the shard.
        :param point: A point to check if it is inside the shard
        :return: True if A @ point < b else False
        """
        if self.is_whole_space:
            return True

        # Convert absolute coordinates to the shifted frame; keep symbolic precision (e.g., Rational).
        point_vec = np.array([sp.sympify(point[sym] - self.shift[sym]) for sym in self.symbols], dtype=object)
        return np.all(self.A @ point_vec < self.b)

    def is_unconstrained(self) -> bool:
        return self.is_whole_space

    def is_valid_trajectory(self, trajectory: Position) -> bool:
        """
        Checks if a trajectory ray remains inside the shard as it scales to infinity.
        Mathematically, the vector v must satisfy A @ v <= 0.
        """
        if self.is_whole_space:
            return True

        # Ensure we match the symbol ordering of the Shard's A matrix
        v = np.array([trajectory[sym] for sym in self.symbols], dtype=np.float64)

        # Check A @ v <= 0 (allowing a tiny float tolerance)
        return np.all(self.A @ v <= 1e-9)

    def get_interior_point(self) -> Position:
        """
        :return: A point inside the shard
        """
        if not self.start_coord:
            return Position({s: sp.Integer(0) for s in self.symbols})
        return Position({sym: self.start_coord[sym] for sym in self.symbols})

    def sample_trajectories(self, compute_n_samples: Callable[[int], int]) -> Set[Position]:
        sampler = EndToEndSamplingEngine(self.A)
        samples = sampler.harvest(compute_n_samples)

        return {
            Position({sym: sp.sympify(int(v)) for v, sym in zip(p, self.symbols)})
            for p in samples
        }

    @staticmethod
    def generate_matrices(
            hyperplanes: List[Hyperplane],
            above_below_indicator: Union[List[int], Tuple[int, ...]]
    ) -> Tuple[np.ndarray, np.ndarray, List[sp.Symbol]]:
        """
        Generate the matrix A and vector b corresponding to the given hyperplanes which represent a shard
        with a specific encoding.
        :param hyperplanes: The list of hyperplanes that represent the shard.
        :param above_below_indicator: The indicator vector that indicates whether the shard is below or above
        the hyperplanes.
        :return: (A, b) where A is a matrix with rows as the linear term coefficients of the hyperplanes
        and b is the free terms vector.
        """
        if any(ind != 1 and ind != -1 for ind in above_below_indicator):
            raise ValueError("Indicators vector must be 1 (above) or -1 (below)")
        if len(hyperplanes) == 0:
            raise ValueError('Cannot generate shard matrices without hyperplanes')

        symbols = list(hyperplanes[0].symbols)
        vectors = []
        free_terms = []

        for hp, ind in zip(hyperplanes, above_below_indicator):
            if ind == 1:
                v, free = hp.as_above_vector
            else:
                v, free = hp.as_below_vector
            free_terms.append(free)
            vectors.append(v)
        return np.vstack(tuple(vectors)), np.array(free_terms), symbols

    def __str__(self):
        return f'A:\n{self.A}\nb: {self.b}'
