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
                 A: np.ndarray | None,
                 b: np.ndarray | None,
                 shift: Position,
                 symbols: List[sp.Symbol],
                 interior_point: Optional[Position] = None,
                 use_inv_t: Optional[bool] = None
                 ):
        """
        :param cmf: The CMF this shard is a part of
        :param constant: The constant to search for in the shard
        :param A: Matrix A defining the linear terms in the inequalities - Ax < b
            (if None, then the shard will be the whole space)
        :param b: Vector b defining the free terms in the inequalities - Ax < b
            (if None, then the shard will be the whole space)
        :param shift: The shift in start points required
        :param symbols: Symbols used by the CMF which this shard is part of
        :param interior_point: A point within the shard
        :param use_inv_t: Whether to use inverse transpose when preforming walk or not
        """
        if use_inv_t is None:
            use_inv_t = config.search.DEFAULT_USES_INV_T

        super().__init__(cmf, constant, shift, use_inv_t)
        self.A = A
        self.b = b
        self.symbols = symbols
        self.shift = np.array([shift[sym] for sym in self.symbols])
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
        point = np.array(point.sorted().values())
        return np.all(self.A @ point < self.b)

    def get_interior_point(self) -> Position:
        """
        :return: A point inside the shard
        """
        if not self.start_coord:
            return Position({s: sp.Integer(0) for s in self.symbols})
        return Position({sym: v for v, sym in zip(self.start_coord.values(), self.symbols)})

    def sample_trajectories(
            self, compute_n_samples: Callable[[int], int], *, strict: Optional[bool] = False
    ) -> Set[Position]:
        # from dreamer.utils.logger import Logger
        # Logger(f'A:\n{self.A}\nb:{self.b}', Logger.Levels.info).log()

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
            raise ValueError(f"Indicators vector must be 1 (above) or -1 (below)")

        symbols = hyperplanes[0].symbols
        symbols = list(symbols)
        vectors = []
        free_terms = []

        for expr, ind in zip(hyperplanes, above_below_indicator):
            if isinstance(expr, Hyperplane):
                hp = expr
            else:
                hp = Hyperplane(expr, symbols)
            if ind == 1:
                v, free = hp.as_above_vector
            else:
                v, free = hp.as_below_vector
            free_terms.append(free)
            vectors.append(v)
        return np.vstack(tuple(vectors)), np.array(free_terms), symbols

    def __repr__(self):
        return f'A={self.A}\nb={self.b}'


if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    b = np.array([1, 1])
    x, y = sp.symbols('x y')
    shard = Shard(a, b, Position({x: 0.5, y: 0.5}), [x, y])
    print(shard.b_shifted)
