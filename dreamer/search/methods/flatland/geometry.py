"""
Flatland geometry helper shared by search methods (Small Angle, Genetic, SA).

Isolates all of the "flatland" linear algebra so search methods stay focused on
their algorithm logic.  Flatland is the lower-dimensional integer lattice
produced by :class:`HyperSpaceConditioner` from a shard's constraint matrix
``A``: equality directions are collapsed and the remaining basis is LLL/BKZ-
reduced so that small integer steps correspond to small geometric angles.

A flatland coordinate vector ``z`` (length ``d_flat``) maps to a real-space
trajectory direction via ``v = Z_reduced @ z`` (length ``d_orig``).
Perturbation and length-doubling happen on ``z``; attribute computation
happens on ``v``.
"""

from typing import Iterator, List

import numpy as np
import sympy as sp
from ramanujantools import Position

from dreamer.extraction.samplers.conditioner import HyperSpaceConditioner
from dreamer.extraction.utils.fast_gcd import reduce_to_primitive
from dreamer.extraction.shard import Shard


class FlatlandGeometry:
    """Convert / check / perturb trajectory directions in flatland space."""

    def __init__(self, shard: Shard):
        """
        :param shard: The shard whose constraint geometry defines flatland space.
        """
        self.shard = shard
        self.symbols: List[sp.Symbol] = list(shard.symbols)
        d_orig = len(self.symbols)

        # Fold any extraction-stage direction constraints (e.g. {'x0': 12, 'y1': 28})
        # into the cone so local search stays on the constrained sub-cone, exactly like
        # the samplers.  ``_fixed`` drives the strict fixed-coordinate sign/non-zero test.
        from dreamer.extraction.samplers.constraints import (
            augment_cone,
            get_trajectory_constraints,
        )
        A_aug, self._fixed = augment_cone(
            shard.A, self.symbols, get_trajectory_constraints()
        )

        if A_aug is None:
            self.Z_reduced = np.eye(d_orig, dtype=np.int64)
            self.B_reduced = np.empty((0, d_orig))
            # No constraints → every direction is inside the cone.
            self._M = None
        else:
            conditioner = HyperSpaceConditioner(np.asarray(A_aug, dtype=np.float64))
            self.Z_reduced, self.B_reduced, _ = conditioner.process()
            # Cone-membership matrix in flatland: a flatland direction ``z`` is
            # inside iff ``A @ (Z_reduced @ z) <= 0`` ⇔ ``M @ z <= 0`` where
            # ``M = A @ Z_reduced``.  Precomputed once so membership is a pure
            # NumPy matmul (no per-call sympy ``Position``).
            A = np.asarray(A_aug, dtype=np.float64)
            self._M = A @ self.Z_reduced.astype(np.float64)

        # Rows of Z_reduced for each strictly-fixed coordinate (sign != 0), used to
        # reject the v_i == 0 / wrong-sign facet the closed cone would otherwise admit.
        self._fixed_pos = {i: s for i, s in self._fixed.items() if s != 0}

        self.d_flat = self.Z_reduced.shape[1]

    #: Tolerance for the ``M @ z <= 0`` cone test (matches ``is_valid_trajectory``).
    _CONE_TOL = 1e-9

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_real(self, z: np.ndarray) -> Position:
        """
        Map a flatland direction ``z`` to a real-space trajectory ``Position``.
        :param z: Integer flatland coordinate vector (length ``d_flat``).
        :return: Position over the shard's symbols (sympy Integers).
        """
        v = self.Z_reduced @ np.asarray(z, dtype=np.int64)
        return Position({sym: sp.Integer(int(val)) for sym, val in zip(self.symbols, v)})

    def to_real_primitive(self, z: np.ndarray) -> Position:
        """
        Map a flatland direction ``z`` to its **GCD-reduced (primitive) ray** in
        real space.

        δ is a property of the trajectory's *ray angle*, so attribute computation
        should always walk the primitive direction: scaled / doubled copies of a
        direction (e.g. ``z`` and ``2z``) collapse to the same real ray and hence
        the same cached walk.

        :param z: Integer flatland coordinate vector (length ``d_flat``).
        :return: Position over the shard's symbols for the primitive real ray.
        """
        v = self.Z_reduced @ np.asarray(z, dtype=np.int64)
        v = reduce_to_primitive(v)
        return Position({sym: sp.Integer(int(val)) for sym, val in zip(self.symbols, v)})

    def to_flatland(self, v: Position) -> np.ndarray:
        """
        Recover the flatland coordinates of a real-space direction.

        ``v`` lies in the integer lattice spanned by the columns of
        ``Z_reduced``, so ``Z_reduced @ z = v`` has an exact integer solution,
        recovered via the least-squares pseudo-inverse and rounded.

        :param v: Real-space trajectory direction (Position over shard symbols).
        :return: Integer flatland coordinate vector (length ``d_flat``).
        """
        v_vec = np.array([float(v[sym]) for sym in self.symbols], dtype=np.float64)
        Z = self.Z_reduced.astype(np.float64)
        z = np.linalg.solve(Z.T @ Z, Z.T @ v_vec)
        return np.rint(z).astype(np.int64)

    # ------------------------------------------------------------------
    # Membership + perturbation
    # ------------------------------------------------------------------

    def traj_norm(self, z: np.ndarray, norm: str = "linf") -> float:
        """
        Compute the trajectory length of flatland vector ``z`` in real shard space.

        The real-space direction is always GCD-reduced (primitive) first so that scaled
        copies ``z`` and ``2z`` return the same norm.

        :param z: Integer flatland coordinate vector.
        :param norm: Which norm to use:
            ``'linf'`` — max absolute coordinate (default; tightest bound on
                ``trajectory_matrix()`` cost, which scales with Σ|coords|);
            ``'l1'``  — sum of absolute coordinates (equals the exact symbolic-
                multiplication count inside ``trajectory_matrix()``);
            ``'l2'``  — Euclidean norm (used by ``depth_from_len``).
        :return: Non-negative float.
        """
        v = self.Z_reduced @ np.asarray(z, dtype=np.int64)
        v = reduce_to_primitive(v).astype(np.float64)
        if norm == "linf":
            return float(np.max(np.abs(v)))
        if norm == "l1":
            return float(np.sum(np.abs(v)))
        # default / "l2"
        return float(np.linalg.norm(v))

    def traj_norm_many(self, Z: np.ndarray, norm: str = "l2") -> np.ndarray:
        """
        Batch version of :meth:`traj_norm` — real shard-space length of many
        flatland directions at once.

        Each row of ``Z`` is mapped to real space (``Z_reduced @ z``) and
        GCD-reduced to its primitive ray (so scaled copies share a length, and
        the length matches the trajectory actually walked, which is the
        primitive ray), then measured with the requested ``norm``.

        :param Z: Integer array of shape ``(k, d_flat)`` — one direction per row.
        :param norm: ``'linf'`` | ``'l1'`` | ``'l2'`` (see :meth:`traj_norm`).
        :return: Float array of length ``k`` with the shard-space lengths.
        """
        Z = np.asarray(Z, dtype=np.int64)
        V = (self.Z_reduced.astype(np.int64) @ Z.T).T  # (k, d_orig)
        g = np.gcd.reduce(np.abs(V), axis=1)
        g[g == 0] = 1
        V = (V // g[:, None]).astype(np.float64)
        if norm == "linf":
            return np.max(np.abs(V), axis=1)
        if norm == "l1":
            return np.sum(np.abs(V), axis=1)
        return np.linalg.norm(V, axis=1)

    def is_inside(self, z: np.ndarray) -> bool:
        """
        :param z: Integer flatland direction.
        :return: True iff the real direction is a non-zero recession direction of the
            shard cone (``z != 0`` and ``M @ z <= 0``).  The zero vector is rejected: it
            is not a trajectory, and under the non-strict cone ``M @ 0 <= 0`` would
            otherwise pass.
        """
        z = np.asarray(z, dtype=np.float64)
        if not np.any(z):
            return False
        if self._M is None:
            return True
        if not bool(np.all(self._M @ z <= self._CONE_TOL)):
            return False
        if self._fixed_pos:
            v = self.Z_reduced.astype(np.float64) @ z
            for i, s in self._fixed_pos.items():
                if v[i] * s <= self._CONE_TOL:
                    return False
        return True

    def is_inside_many(self, Z: np.ndarray) -> np.ndarray:
        """
        Batch cone-membership test for many flatland directions at once.

        :param Z: Integer array of shape ``(k, d_flat)`` — one direction per row.
        :return: Boolean array of length ``k``; ``True`` where the row is a non-zero
            recession direction of the shard cone (``z != 0`` and ``M @ z <= 0``).  Zero
            rows are rejected (not trajectories; ``M @ 0 <= 0`` would otherwise pass).
        """
        Z = np.asarray(Z, dtype=np.float64)
        nonzero = np.any(Z != 0, axis=1)
        if self._M is None:
            return nonzero
        inside = nonzero & np.all((self._M @ Z.T) <= self._CONE_TOL, axis=0)
        if self._fixed_pos:
            V = (self.Z_reduced.astype(np.float64) @ Z.T)  # (d_orig, k)
            for i, s in self._fixed_pos.items():
                inside &= (V[i] * s) > self._CONE_TOL
        return inside

    def perturbations(self, z: np.ndarray, *, reduce: bool = True) -> Iterator[np.ndarray]:
        """
        Yield the ``2 * d_flat`` neighbours of ``z`` (each coordinate ±1).

        :param z: Integer flatland direction to perturb.
        :param reduce: If ``True`` (default, SmallAngle behaviour), each
            candidate is GCD-reduced to a primitive (GCD == 1) vector and zero
            vectors are skipped.  If ``False`` (GA/SA behaviour, faithful to
            the reference algorithms), candidates are returned as-is (raw ±1
            step) — only the all-zero vector is skipped.
        """
        z = np.asarray(z, dtype=np.int64)
        for i in range(self.d_flat):
            for sign in (1, -1):
                cand = z.copy()
                cand[i] += sign
                if not np.any(cand):
                    continue
                yield reduce_to_primitive(cand) if reduce else cand
