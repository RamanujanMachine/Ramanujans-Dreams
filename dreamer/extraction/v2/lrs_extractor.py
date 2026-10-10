"""
Exact shard extractor.

Pipeline per call:

1. Enumerate every non-empty cell of the arrangement (sign vectors) via
   :func:`dreamer.extraction.v2.cells.iter_cells` (memoryless reverse
   search), streamed so classification can interleave.
2. Classify each cell as bounded / unbounded.  Two backends:
   * ``"lp"`` (default) — one in-process recession-cone LP per cell
     (`cells.make_unbounded_checker`), ~0.5 ms, no subprocess.
   * ``"lrs"`` — spawn the lrslib ``lrs`` binary, compute the cell's
     V-representation and look for a ray.  Kept as an authoritative
     cross-check; ~30x slower per cell due to subprocess + full vertex
     enumeration.
3. For each unbounded cell, witness it with an integer point via
   :func:`dreamer.extraction.v2.milp.find_integer_point`.
"""

from __future__ import annotations

import subprocess
import time
from typing import List, Optional, Tuple

import numpy as np

from dreamer.extraction.hyperplanes import Hyperplane
from .base import BaseExtractor, ShardMapping
from .cells import (
    ExtractionTimeout,
    iter_cells,
    iter_cells_canonical,
    iter_subtree,
    make_unbounded_checker,
    reverse_search_seeds,
)
from .symmetry import SymmetryStrategy
from .lrs_io import format_hrep, lrs_available, parse_vrep_unbounded, run_lrs
from .milp import find_integer_point


def _subtree_extract_worker(args) -> Tuple[ShardMapping, bool]:
    """
    Enumerate + classify + locate one subtree in a child process.

    Each worker builds its own LP solvers (CBC models are not picklable).
    Returns ``(shards, timed_out)`` where ``shards`` maps the subtree's
    unbounded-cell encodings to integer points; ``timed_out`` is True if
    the wall-clock ``deadline`` was passed mid-subtree (``shards`` then
    holds whatever was completed first — salvage).
    """
    A, c, base, root, milp_bound, max_cells, deadline = args
    A = np.asarray(A, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    is_unbounded = make_unbounded_checker(A)
    out: ShardMapping = {}
    try:
        for sig in iter_subtree(
            A, c, base, root, max_cells=max_cells, deadline=deadline,
        ):
            if deadline is not None and time.time() > deadline:
                return out, True
            sig_arr = np.asarray(sig, dtype=np.int64)
            if not is_unbounded(sig_arr):
                continue
            point = find_integer_point(A, c, sig_arr, bound=milp_bound)
            if point is not None:
                out[sig] = point
    except ExtractionTimeout:
        # Enumeration tripped the deadline — return the partial we built.
        return out, True
    return out, False


class LrslibExtractor(BaseExtractor):
    """
    Exact strategy: enumerate cells, classify by unboundedness, witness
    each unbounded cell with an MILP integer point.

    :param unbounded_check: ``"lp"`` (default, in-process recession-cone
        LP) or ``"lrs"`` (the lrslib binary, authoritative cross-check).
    :param binary: Name or path of the ``lrs`` executable (only used when
        ``unbounded_check="lrs"``).
    :param per_call_timeout: Timeout (seconds) for each ``lrs`` call.
    :param max_cells: Optional safety ceiling on the cell enumeration
        (``None`` = unbounded; the deadline is the intended stop).
    :param milp_bound: Symmetric box ``|x_d| <= milp_bound`` passed to
        the MILP solver -- larger lets it reach integer points in cells
        whose interior is far from the origin.
    :param seed: RNG seed used when picking the starting cell.
    :param num_workers: Process count for parallel reverse-search cell
        enumeration.  ``1`` (default) runs serially.  Ignored when a
        ``symmetry`` is active (the canonical BFS runs serially).
    :param symmetry: Optional :class:`SymmetryStrategy`.  When set, the
        extractor enumerates one representative per symmetry orbit via the
        canonical-teleportation BFS instead of full reverse search.
    :raises FileNotFoundError: If ``unbounded_check="lrs"`` and the
        ``lrs`` binary is not on ``PATH``.
    :raises ValueError: For an unknown ``unbounded_check``.
    """

    name = "exact"

    def __init__(
        self,
        *,
        unbounded_check: str = "lp",
        binary: str = "lrs",
        per_call_timeout: float = 60.0,
        max_cells: Optional[int] = None,
        milp_bound: int = 10**6,
        seed: Optional[int] = 0,
        num_workers: int = 1,
        symmetry: Optional[SymmetryStrategy] = None,
    ):
        if unbounded_check not in ("lp", "lrs"):
            raise ValueError(
                f"unbounded_check must be 'lp' or 'lrs', got {unbounded_check!r}"
            )
        # The lrs binary is only required for the (opt-in) lrs backend;
        # the default LP backend has no external dependency.
        if unbounded_check == "lrs" and not lrs_available(binary):
            raise FileNotFoundError(
                f"lrs binary not found on PATH (looked for {binary!r}). "
                "Install lrslib (e.g. `apt-get install lrslib` on WSL/Ubuntu) "
                "or use unbounded_check='lp'."
            )
        self.unbounded_check = unbounded_check
        self.binary = binary
        self.per_call_timeout = per_call_timeout
        self.max_cells = max_cells
        self.milp_bound = milp_bound
        self.seed = seed
        self.num_workers = num_workers
        self.symmetry = symmetry

    def extract(
        self,
        hyperplanes: List[Hyperplane],
        *,
        deadline: Optional[float] = None,
    ) -> ShardMapping:
        if not hyperplanes:
            return {}

        A, c = self.hyperplanes_to_matrix(hyperplanes)
        # Symmetry reduction routes to the canonical-teleportation BFS,
        # which is inherently serial (it is not subtree-decomposable like
        # reverse search) — see _extract_serial.
        if self.symmetry is None and (
            self.num_workers
            and self.num_workers > 1
            and self.unbounded_check != "lrs"
        ):
            # Parallel needs the in-process LP check (lrs runs serially so
            # its per-cell subprocess is honoured); the LP and lrs backends
            # agree, so this only affects speed, not the result.
            return self._extract_parallel(A, c, deadline)
        return self._extract_serial(A, c, deadline)

    def _extract_serial(
        self, A: np.ndarray, c: np.ndarray, deadline: Optional[float]
    ) -> ShardMapping:
        """
        Stream cells and classify + locate each on the fly.  At any instant
        ``out`` holds fully-formed unbounded shards; a deadline hit re-raises
        an ``ExtractionTimeout`` carrying them so the manager can union with
        the heuristic instead of discarding.

        With a ``symmetry`` active, cells are streamed by the canonical
        BFS (one representative per orbit); otherwise by reverse search.
        """
        is_unbounded = self._build_unbounded_checker(A, c)
        if self.symmetry is not None:
            cell_stream = iter_cells_canonical(
                A, c, self.symmetry,
                max_cells=self.max_cells, seed=self.seed, deadline=deadline,
            )
        else:
            cell_stream = iter_cells(
                A, c, max_cells=self.max_cells, seed=self.seed, deadline=deadline,
            )
        out: ShardMapping = {}
        try:
            for sig in cell_stream:
                if deadline is not None and time.time() > deadline:
                    raise ExtractionTimeout(
                        f"cell classification passed its deadline after "
                        f"{len(out)} unbounded cells",
                        partial=out,
                    )
                sig_arr = np.asarray(sig, dtype=np.int64)
                if not is_unbounded(sig_arr):
                    continue
                point = find_integer_point(A, c, sig_arr, bound=self.milp_bound)
                if point is None:
                    continue
                out[sig] = point
        except ExtractionTimeout as exc:
            exc.partial = {**(exc.partial or {}), **out}
            raise
        return out

    def _extract_parallel(
        self, A: np.ndarray, c: np.ndarray, deadline: Optional[float]
    ) -> ShardMapping:
        """
        Salvage-aware parallel exact: dispatch each disjoint root subtree
        to a worker that enumerates + classifies + locates it.  Workers
        return partial results on a deadline hit, so the merged map is
        always valid; if any worker timed out we re-raise an
        ``ExtractionTimeout`` carrying the merge for the manager to union
        with the heuristic.
        """
        base, children = reverse_search_seeds(A, c, seed=self.seed)
        is_unbounded = make_unbounded_checker(A)

        out: ShardMapping = {}
        # The base cell belongs to no subtree — classify it here.
        if is_unbounded(base):
            pt = find_integer_point(A, c, base, bound=self.milp_bound)
            if pt is not None:
                out[tuple(base.tolist())] = pt

        if len(children) < 2:
            # Nothing to parallelise — fall back to the serial sweep
            # (which also gives proper streaming salvage).
            return self._extract_serial(A, c, deadline)

        tasks = [
            (A, c, base, child, self.milp_bound, self.max_cells, deadline)
            for child in children
        ]
        timed_out = False
        from dreamer.utils.resilient_pool import ResilientPool
        with ResilientPool(min(self.num_workers, len(tasks)), name="exact extraction") as pool:
            try:
                for sub_out, sub_timed_out in pool.imap_unordered(
                    _subtree_extract_worker, tasks
                ):
                    out.update(sub_out)
                    timed_out = timed_out or sub_timed_out
                    if self.max_cells is not None and len(out) > self.max_cells:
                        raise RuntimeError(
                            f"Cell enumeration exceeded max_cells={self.max_cells}"
                        )
            except BaseException:
                pool.terminate()
                raise

        if timed_out:
            raise ExtractionTimeout(
                f"parallel exact passed its deadline ({len(out)} shards salvaged)",
                partial=out,
            )
        return out

    def _build_unbounded_checker(self, A: np.ndarray, c: np.ndarray):
        """Return the per-cell ``unbounded(sign_vector) -> bool`` predicate."""
        if self.unbounded_check == "lrs":
            return lambda sig: self._is_unbounded_lrs(A, c, sig)
        return make_unbounded_checker(A)

    def _is_unbounded_lrs(
        self, A: np.ndarray, c: np.ndarray, sign_vector: np.ndarray
    ) -> bool:
        """
        Ask ``lrs`` whether the cell is unbounded (opt-in backend).

        :raises RuntimeError: If ``lrs`` times out or exits non-zero.
        """
        hrep = format_hrep(A, c, sign_vector, name="cell")
        try:
            vrep = run_lrs(hrep, binary=self.binary, timeout=self.per_call_timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"lrs timed out after {self.per_call_timeout}s on a cell "
                f"({A.shape[0]} hyperplanes in dim {A.shape[1]})"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"lrs failed (exit {exc.returncode}): {exc.stderr!r}"
            ) from exc
        return parse_vrep_unbounded(vrep)
