from __future__ import annotations

import base64
import pickle

from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.jsonable import JSONable
from dreamer.utils.constants.constant import Constant
from dreamer.configs import config
from ramanujantools.cmf import CMF
from ramanujantools import Position
from typing import Dict, List, Optional, Tuple, Union
import sympy as sp
import numpy as np

from dreamer.utils.types import CMFData


class Shard(Searchable, JSONable):
    def __init__(self,
                 cmf: CMF,
                 constants: Union[Constant, List[Constant]],
                 hyperplanes: List[Hyperplane],
                 encoding: List[int],
                 shift: Position,
                 interior_point: Optional[Position] = None,
                 use_inv_t: Optional[bool] = None,
                 cmf_name: str = 'UnknownCMF',
                 hyperplanes_already_shifted: bool = False,
                 selected_trajectory: Optional[Position] = None
                 ):
        """
        :param cmf: The CMF this shard is a part of
        :param constants: A constant or list of constants to search for in the shard
        :param hyperplanes: The hyperplanes defining the shard
        :param encoding: The indicator vector that indicates whether the shard is below or above the hyperplanes
        :param shift: The shift in start points required
        :param interior_point: A point within the shard
        :param use_inv_t: Whether to use inverse transpose when preforming walk or not
        :param cmf_name: The name of the CMF
        :param selected_trajectory: An optional user-supplied trajectory direction
            (real-space ``Position`` over the CMF symbols) associated with this shard.
            When present it is the analysis/search seed for the shard: the analysis
            stage evaluates it alongside the sampled trajectories and the search methods
            use it as the initial optimiser seed (instead of a reservoir-sampled seed).
            Kept in-memory only (not serialised to the shard DTO/cache).
        :param hyperplanes_already_shifted: When True, ``hyperplanes`` are
            already in shifted coordinates, so the (expensive, sympy)
            per-hyperplane ``apply_shift`` is skipped.  The shift is the
            same for every shard of a CMF, so the caller can shift once
            and reuse the result across all shards instead of re-shifting
            in each ``Shard.__init__``.
        """
        use_inv_t_value: bool = bool(config.search.DEFAULT_USES_INV_T if use_inv_t is None else use_inv_t)

        super().__init__(cmf, constants, shift, use_inv_t_value, cmf_name)
        self.symbols = list(cmf.matrices.keys())
        self.A, self.b = None, None
        # Sign vector relative to the parent CMF's hyperplane list:
        # +1 = shard is above hp_i, -1 = below.  Paired one-to-one with
        # ``cmf.hyperplanes`` so the DTO can carry a compact combinatorial
        # label for the shard (DB-friendly; readers don't need ``A, b``).
        self.encoding: Tuple[int, ...] = tuple(int(v) for v in encoding) if encoding else ()

        if hyperplanes:
            # Work in shifted coordinates, then translate tested points back by `shift`.
            if hyperplanes_already_shifted:
                shifted_hyperplanes = hyperplanes
            else:
                shifted_hyperplanes = [hp.apply_shift(shift) for hp in hyperplanes]
            self.A, self.b, self.symbols = self.generate_matrices(shifted_hyperplanes, encoding)
        self.start_coord = interior_point
        self.selected_trajectory = selected_trajectory
        self.is_whole_space = self.A is None or self.b is None

    @classmethod
    def from_cmf_data(cls, cmf_data: CMFData, constants: Union[Constant, List[Constant]],
                      hyperplanes: List[Hyperplane], encoding: List[int],
                      interior_point: Optional[Position] = None, *args,
                      hyperplanes_already_shifted: bool = False,
                      selected_trajectory: Optional[Position] = None,
                      **kwargs) -> 'Shard':
        return cls(
            cmf_data.cmf, constants, hyperplanes, encoding, cmf_data.shift,
            interior_point, cmf_data.use_inv_t, cmf_data.cmf_name,
            hyperplanes_already_shifted=hyperplanes_already_shifted,
            selected_trajectory=selected_trajectory,
        )

    @classmethod
    def from_start_and_trajectory(cls, cmf_data: CMFData,
                                  constants: Union[Constant, List[Constant]],
                                  start: Position, trajectory: Position, *,
                                  hyperplanes: Optional[List[Hyperplane]] = None) -> 'Shard':
        """Build the shard that contains ``start`` after one step along ``trajectory``.

        The shard identity is its encoding (sign vector) ``sign(hp_i(point))`` over the
        CMF's canonically-ordered hyperplanes.  A ``start`` that lies *on* a hyperplane has
        a zero sign there, so its shard is ambiguous.  Stepping once along the full
        trajectory vector (``start + trajectory``) moves off the border into the interior;
        the encoding is computed from that stepped point, while the originally provided
        ``start`` is kept as the shard's interior/start point.

        :param cmf_data: The CMFData (CMF object + shift) to build the shard in.
        :param constants: A constant or list of constants to search for in the shard.
        :param start: The exact start point chosen by the user (absolute coordinates); may
            lie on a shard border.  Kept verbatim as the shard's start point.
        :param trajectory: The trajectory direction; one full step ``start + trajectory``
            must reach a strictly-interior point of a shard.
        :param hyperplanes: Optional pre-computed canonical hyperplanes; defaults to
            ``extract_cmf_hyperplanes(cmf_data)``.
        :raises ValueError: If ``start + trajectory`` lies on any hyperplane, i.e. one step
            does not reach a legal interior point of a shard.
        :return: The reconstructed :class:`Shard` carrying the derived encoding.
        """
        # Local import to avoid a circular import (extractor imports Shard).
        from dreamer.extraction.extractor import extract_cmf_hyperplanes

        hps = hyperplanes if hyperplanes is not None else extract_cmf_hyperplanes(cmf_data)
        symbols = list(cmf_data.cmf.matrices.keys())

        # One step along the trajectory, in absolute (unshifted) coordinates.  Evaluating
        # the unshifted hyperplane expressions here matches the sign convention used by
        # generate_matrices / in_space after the internal apply_shift.
        stepped = {s: sp.sympify(start[s]) + sp.sympify(trajectory[s]) for s in symbols}

        encoding, on_boundary = cls.encoding_at(hps, stepped)
        if on_boundary:
            raise ValueError(
                f"start + trajectory = {stepped} lies on hyperplane(s) {on_boundary}; "
                "one step does not reach a legal interior point of a shard — "
                "choose a different trajectory or start."
            )

        return cls.from_cmf_data(
            cmf_data, constants, hps, encoding, start,
            hyperplanes_already_shifted=False,
            selected_trajectory=Position({s: sp.sympify(trajectory[s]) for s in symbols}),
        )

    @classmethod
    def from_matrices(cls, cmf: CMF,
                 constants: Union[Constant, List[Constant]],
                 A: np.ndarray, b: np.ndarray,
                 shift: Position,
                 interior_point: Optional[Position] = None,
                 use_inv_t: Optional[bool] = None,
                 cmf_name: str = 'UnknownCMF'):
        shard = cls(cmf, constants, [], [], shift, interior_point, use_inv_t, cmf_name)
        shard.A = A
        shard.b = b
        shard.is_whole_space = False
        return shard

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
        Mathematically, the vector v must be a non-zero recession direction, i.e.
        ``v != 0`` and ``A @ v <= 0`` (the *closed* recession cone): a direction with
        ``A_i v == 0`` runs parallel to facet ``i`` and, from a strictly-interior start,
        stays inside the open shard forever.  The zero vector is rejected explicitly —
        it does not move and is not a trajectory (under the non-strict cone ``A @ 0 <= 0``
        would otherwise pass).
        """
        # Ensure we match the symbol ordering of the Shard's A matrix
        v = np.array([trajectory[sym] for sym in self.symbols], dtype=np.float64)

        # The zero vector is never a valid trajectory (even in whole space).
        if not np.any(v):
            return False

        if self.is_whole_space:
            return True

        # Check A @ v <= 0 (closed recession cone, allowing a tiny float tolerance)
        return np.all(self.A @ v <= 1e-9)

    def get_interior_point(self) -> Position:
        """
        :return: A point inside the shard
        """
        if not self.start_coord:
            return Position({s: sp.Integer(0) for s in self.symbols})
        return Position({sym: self.start_coord[sym] for sym in self.symbols})

    @staticmethod
    def encoding_at(
            hyperplanes: List[Hyperplane],
            point: Dict[sp.Symbol, sp.Expr],
    ) -> Tuple[List[int], List[sp.Expr]]:
        """
        Compute the sign vector (shard encoding) of ``point`` against ``hyperplanes``.

        The hyperplane expressions are in absolute (unshifted) coordinates, so ``point``
        must be given in the same absolute coordinates.  This matches the sign convention
        used by :meth:`generate_matrices` / :meth:`in_space` after the internal
        ``apply_shift`` (shifted-hp at ``point - shift`` == unshifted-hp at ``point``).
        :param hyperplanes: Canonically-ordered hyperplanes of the CMF.
        :param point: Mapping ``{symbol: value}`` in absolute coordinates.
        :return: ``(encoding, on_boundary)`` where ``encoding[i]`` is ``+1``/``-1`` (and
            ``0`` where the point lies exactly on hyperplane ``i``), and ``on_boundary``
            lists the hyperplane expressions the point lies on (empty when strictly interior).
        """
        encoding: List[int] = []
        on_boundary: List[sp.Expr] = []
        for hp in hyperplanes:
            val = sp.sympify(hp.expr.subs({s: point[s] for s in hp.symbols}))
            if val == 0:
                on_boundary.append(hp.expr)
                encoding.append(0)
            else:
                encoding.append(1 if val > 0 else -1)
        return encoding, on_boundary

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

    def to_json(self) -> dict:
        """
        Serialize shard state into JSON using a base64-encoded pickle payload.
        :return: JSON-compatible dictionary that can be restored with ``from_json_obj``.
        """
        payload = base64.b64encode(pickle.dumps(self)).decode("ascii")
        return {
            "__class__": "Shard",
            "cmf_name": self.cmf_name,
            "consts": [c.name for c in self.consts],
            "payload_b64": payload,
        }

    def to_json_obj(self) -> dict:
        """Backward-compatible alias used by existing exporter paths."""
        return self.to_json()

    @classmethod
    def from_json_obj(cls, obj: dict) -> "Shard":
        """
        Restore a shard from ``to_json`` payload.
        :param obj: JSON dictionary generated by ``to_json``.
        :raises ValueError: If the payload is malformed or not a Shard instance.
        :return: Restored shard instance.
        """
        payload = obj.get("payload_b64")
        if not isinstance(payload, str):
            raise ValueError("Shard JSON payload is missing 'payload_b64'")
        restored = pickle.loads(base64.b64decode(payload.encode("ascii")))
        if not isinstance(restored, cls):
            raise ValueError(f"Expected payload to decode into {cls.__name__}, got {type(restored)}")
        return restored

