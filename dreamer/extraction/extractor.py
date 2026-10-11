from dreamer.configs import (
    sys_config,
    extraction_config
)
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.utils.schemes.extraction_scheme import ExtractionScheme, ExtractionModScheme
from dreamer.utils.logger import Logger
from dreamer.utils.constants.constant import Constant
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.atlas_writer import (
    read_shard_records,
    update_cmf_hyperplanes,
    write_shard_records,
)
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.configs import config
from dreamer.utils.types import CMFData
from .utils import initial_points as init_points
from .v2 import ExtractionManager, LrslibExtractor, RayShootingExtractor
from .v2 import symmetry_for_cmf

import os.path
import sympy as sp
import numpy as np
import math
from collections import defaultdict
from functools import partial
from ramanujantools.cmf import pFq as rt_pFq
from ramanujantools import Position
from typing import Dict, List, Optional, Set, Tuple, Union


# TODO: remove this copy
# def extract_cmf_hyperplanes(cmf_data: CMFData) -> List[Hyperplane]:
#     """Compute and return the canonically-ordered hyperplanes of *cmf_data*.

#     Exposed at module level so callers (e.g. shard reconstruction from a
#     cached ``ShardDTO``) don't need a full ``ShardExtractor`` instance.

#     The ordering is deterministic (sorted by ``str(hp.expr)``) and matches
#     what ``ShardExtractor._extract_cmf_hps`` produces, so
#     ``shard_encoding[i]`` unambiguously labels ``hyperplanes[i]``.
#     """
#     cmf = cmf_data.cmf
#     shift = cmf_data.shift
#     symbols = list(cmf.matrices.keys())
#     hps: Set[Hyperplane] = set()

#     for s in symbols:
#         if isinstance(cmf, rt_pFq):
#             det = rt_pFq.determinant(cmf.p, cmf.q, cmf.z, s)
#         else:
#             det = cmf.matrices[s].det()
#         zeros = sp.solve(det)
#         zeros = [Hyperplane(lhs - rhs, symbols) for sol in zeros for lhs, rhs in sol.items()]
#         hps.update(zeros)

#         poles: Set[Hyperplane] = set()
#         for v in cmf.matrices[s].iter_values():
#             if (den := v.as_numer_denom()[1]) == 1:
#                 continue
#             solutions = {
#                 (sym, sol)
#                 for sym in den.free_symbols
#                 for sol in sp.solve(sp.simplify(den), sym)
#             }
#             for lhs, rhs in solutions:
#                 poles.add(Hyperplane(lhs - rhs, symbols))
#         hps.update(poles)

#     filtered = [hp for hp in hps if hp.apply_shift(shift).is_in_integer_shift()]
#     filtered.sort(key=lambda hp: str(hp.expr))
#     return filtered


def extract_cmf_hyperplanes(cmf_data: CMFData) -> List[Hyperplane]:
    """Compute and return the canonically-ordered hyperplanes of *cmf_data*.

    This is the same computation as ``ShardExtractor._extract_cmf_hps`` but
    exposed as a module-level function so callers (e.g. shard reconstruction
    from a cached ShardDTO) don't need to construct a full ``ShardExtractor``.

    The result is sorted by ``str(hp.expr)`` for determinism: the same CMF
    always produces the same hyperplane ordering, so ``shard_encoding[i]``
    unambiguously labels ``hyperplanes[i]`` across runs.
    """
    cmf = cmf_data.cmf
    shift = cmf_data.shift
    symbols = list(cmf.matrices.keys())
    hps: Set[Hyperplane] = set()

    for s in symbols:
        if isinstance(cmf, rt_pFq):
            det = rt_pFq.determinant(cmf.p, cmf.q, cmf.z, s)
        else:
            det = cmf.matrices[s].det()
        zeros = sp.solve(det)
        zeros = [Hyperplane(lhs - rhs, symbols) for sol in zeros for lhs, rhs in sol.items()]
        hps.update(zeros)

        poles: Set[Hyperplane] = set()
        for v in cmf.matrices[s].iter_values():
            if (den := v.as_numer_denom()[1]) == 1:
                continue
            solutions = {
                (sym, sol)
                for sym in den.free_symbols
                for sol in sp.solve(sp.simplify(den), sym)
            }
            for lhs, rhs in solutions:
                poles.add(Hyperplane(lhs - rhs, symbols))
        hps.update(poles)

    filtered = [hp for hp in hps if hp.apply_shift(shift).is_in_integer_shift()]
    filtered.sort(key=lambda hp: str(hp.expr))
    return filtered


class ShardExtractorMod(ExtractionModScheme):
    """
    Module for shard extraction
    """

    def __init__(self, cmf_data: Dict[Constant, List[CMFData]]):
        """
        Creates a shard extraction module
        :param cmf_data: A mapping from constants to a list of CMFs
        """
        super().__init__(
            cmf_data,
            name=self.__class__.__name__,
            desc='Shard extractor module',
            version='0.0.1'
        )

    def execute(self) -> Dict[Constant, List[Searchable]]:
        """
        Extract shards from CMFs.

        CMFs shared by multiple constants (same ``CMFData`` object appearing
        under several constant keys) are extracted *once* with all of their
        constants bundled into a single multi-constant ``Shard``.  The same
        ``Shard`` object is then placed under every one of its constants in
        the returned dict so that downstream stages can still iterate by
        constant when needed.
        :return: A mapping from constants to a list of shards
        """
        # ----------------------------------------------------------------
        # Group by CMFData identity: build a mapping
        #   cmf_data_id → (CMFData, [constants_that_share_it])
        # ----------------------------------------------------------------
        cmf_id_to_entry: Dict[int, tuple] = {}  # id(CMFData) → (CMFData, list[Constant])
        for const, cmf_data_list in self.cmf_data.items():
            for cmd_data in cmf_data_list:
                key = id(cmd_data)
                if key not in cmf_id_to_entry:
                    cmf_id_to_entry[key] = (cmd_data, [])
                cmf_id_to_entry[key][1].append(const)

        all_shards: Dict[Constant, List[Searchable]] = defaultdict(list)

        call_number = 0
        for cmd_data, consts in SmartTQDM(
                cmf_id_to_entry.values(),
                desc='Extracting shards',
                **sys_config.TQDM_CONFIG,
        ):
            call_number += 1
            extractor = ShardExtractor(consts, cmd_data)
            shards = extractor.extract(call_number=call_number)

            # Distribute the shared Shard objects to every constituent constant.
            for const in consts:
                all_shards[const] += shards

            # DB-ready ShardDTO records (flat file per CMF at root).  ``found_constants``
            # is written empty: a constant is recorded as found in a shard only after
            # the analysis stage identifies a converging trajectory there
            # (update_shard_found_constants).
            if sys_config.EXPORT_CMFS:
                write_shard_records(
                    sys_config.EXPORT_CMFS,
                    cmd_data.cmf_name,
                    shards,
                    found_constants=[],
                )
                update_cmf_hyperplanes(
                    sys_config.EXPORT_CMFS,
                    cmd_data.cmf_name,
                    extractor.hyperplanes,
                )

        return all_shards


class ShardExtractor(ExtractionScheme):
    """
    Shard extractor is a representation of a shard finding method.
    """

    def __init__(self, constants: Union[Constant, List[Constant]], cmf_data: CMFData):
        """
        Extracts the shards of a CMF
        :param constants: Constant or list of constants searched in this CMF
        :param cmf_data: CMF to extract shards from, more data for extraction and later usage
        """
        consts_list = constants if isinstance(constants, list) else [constants]
        # ExtractionScheme base stores a single const; pass the first one for compatibility.
        super().__init__(consts_list[0], cmf_data)
        self._constants: List[Constant] = consts_list
        # Populated by extract(); read by ShardExtractorMod.execute() so it
        # can backfill the CmfDTO row with the hyperplanes used to derive
        # the shards.
        self.hyperplanes: Set[Hyperplane] = set()
        # self.pool = create_pool() if extraction_config.PARALLELIZE else None

    @property
    def symbols(self) -> List[sp.Symbol]:
        """
        :return: The CMF's symbols
        """
        return list(self.cmf_data.cmf.matrices.keys())

    def _extract_cmf_hps(self) -> List[Hyperplane]:
        """Delegate to the module-level ``extract_cmf_hyperplanes`` function."""
        return extract_cmf_hyperplanes(self.cmf_data)

    def extract(self, call_number=None) -> List[Shard]:
        """
        Extracts the shards from the CMF.

        The discovery method is chosen by
        ``extraction_config.STRATEGY``:

        * ``"auto" | "exact" | "heuristic"`` -- delegate to the v2
          :class:`~dreamer.extraction.v2.ExtractionManager` (lrs + MILP
          with a ray-shooting fallback).  ``"auto"`` is the default and
          enables the wall-clock timeout protection.
        * ``"legacy"`` -- the original brute-force lattice scan in
          :mod:`dreamer.extraction.utils.initial_points` (kept verbatim
          for parity and benchmarking).

        Either path may be supplemented or fully driven by
        ``cmf_data.selected_points`` exactly as before.
        :return: The list of shards matching the CMF
        """
        # compute hyperplanes and prepare sample point
        hps = self._extract_cmf_hps()
        self.hyperplanes = hps

        if not hps:
            return [Shard.from_cmf_data(self.cmf_data, self._constants, [], [])]

        symbols = list(hps)[0].symbols
        shard_encodings: Dict[Tuple[int, ...], Position] = dict()
        # A user-supplied trajectory is kept (in-memory) on the shard so the analysis
        # and search stages can use it as their seed.  Paired 1:1 with the encoding
        # (last write wins, matching ``shard_encodings``); empty unless trajectories given.
        shard_trajectories: Dict[Tuple[int, ...], Optional[Position]] = dict()
        selected = [] if self.cmf_data.selected_points is None else self.cmf_data.selected_points

        if self.cmf_data.only_selected:
            if self.cmf_data.selected_points is None:
                raise ValueError('No start points were provided for extraction.')
        else:
            cached = None
            if config.extraction.LOAD_SHARD_CACHE:
                cached = self._load_cached_encodings(hps, symbols)
            if cached is not None:
                shard_encodings.update(cached)
                Logger(
                    f'Loaded {len(cached)} cached shards from shards.jsonl; '
                    'skipping extraction',
                    level=Logger.Levels.info,
                ).log()
            else:
                strategy = config.extraction.STRATEGY
                if strategy == 'legacy':
                    shard_encodings.update(self._discover_via_legacy(hps, symbols))
                elif strategy in ('auto', 'exact', 'heuristic'):
                    shard_encodings.update(
                        self._discover_via_v2(hps, symbols, strategy)
                    )
                else:
                    raise ValueError(
                        f"Unknown extraction strategy {strategy!r}; expected "
                        "'auto', 'exact', 'heuristic' or 'legacy'"
                    )

        if self.cmf_data.selected_points:
            # Trajectories pair 1:1 with the selected start points.  A None entry (or no
            # trajectories at all) means "use the start point as-is".  A provided trajectory
            # disambiguates a border start point by deriving the encoding one step along it.
            trajectories = self.cmf_data.selected_trajectories
            if trajectories is None:
                trajectories = [None] * len(selected)
            elif len(trajectories) != len(selected):
                raise ValueError(
                    f'selected_trajectories length ({len(trajectories)}) must match '
                    f'selected_start_points length ({len(selected)})'
                )

            shift_vals = list(self.cmf_data.shift.values())

            # validate shards using the sampled points
            for p, traj in SmartTQDM(
                    list(zip(selected, trajectories)),
                    desc='Computing shard encodings', **sys_config.TQDM_CONFIG):
                # Absolute (unshifted) coordinates of the user's start point.
                abs_point = tuple(coord + shift for coord, shift in zip(p, shift_vals))
                point_dict = {sym: coord for sym, coord in zip(symbols, abs_point)}

                if traj is None:
                    # No trajectory: encode at the start point itself; a point lying on a
                    # shard border is ambiguous, so skip it (legacy behaviour).
                    enc, on_boundary = Shard.encoding_at(hps, point_dict)
                    if on_boundary:
                        continue
                else:
                    # One step along the trajectory (direction is shift-invariant).  The
                    # stepped point must be a strict interior point of a shard.
                    stepped_dict = {
                        sym: coord + step
                        for sym, coord, step in zip(symbols, abs_point, traj)
                    }
                    enc, on_boundary = Shard.encoding_at(hps, stepped_dict)
                    if on_boundary:
                        raise ValueError(
                            f'start point {p} + trajectory {traj} lies on hyperplane(s) '
                            f'{on_boundary}; one step does not reach a legal interior point '
                            'of a shard — choose a different trajectory or start point.'
                        )

                # Keep the user's start point verbatim as the shard's interior/start point.
                shard_encodings[tuple(enc)] = Position(point_dict)
                # Retain the trajectory (a shift-invariant direction) so the analysis
                # and search stages can use it as their seed.
                if traj is not None:
                    shard_trajectories[tuple(enc)] = Position(
                        {sym: sp.sympify(step) for sym, step in zip(symbols, traj)}
                    )

        Logger(
            f'In CMF no. {call_number}: found {len(hps)} hyperplanes and {len(shard_encodings)} shards ',
            level=Logger.Levels.info
        ).log()

        # Create shard objects.  The shift is identical for every shard,
        # so shift the hyperplanes ONCE here and reuse the result — this
        # avoids re-running the (sympy) per-hyperplane apply_shift inside
        # every Shard.__init__, which otherwise dominates this loop.
        shifted_hps = [hp.apply_shift(self.cmf_data.shift) for hp in hps]

        # Optional direction-constraint shard filter: when the extraction config pins the
        # trajectory ratio (e.g. {'x0': 12, 'y1': 28}), keep only shards whose recession
        # cone actually admits such a direction, so downstream stages never search a shard
        # that cannot contain a constrained trajectory.
        from dreamer.extraction.samplers.constraints import (
            constrained_cone_feasible,
            get_trajectory_constraints,
        )
        constraints = get_trajectory_constraints()

        shards = []
        dropped = 0
        for enc in SmartTQDM(shard_encodings.keys(), desc='Creating shard objects', **sys_config.TQDM_CONFIG):
            shard = Shard.from_cmf_data(
                self.cmf_data, self._constants, shifted_hps, enc, shard_encodings[enc],
                hyperplanes_already_shifted=True,
                selected_trajectory=shard_trajectories.get(enc),
            )
            if constraints and not shard.is_whole_space and not constrained_cone_feasible(
                shard.A, shard.symbols, constraints
            ):
                dropped += 1
                continue
            shards.append(shard)

        if constraints:
            Logger(
                f'Trajectory constraints {constraints}: kept {len(shards)} shard(s), '
                f'dropped {dropped} that admit no such direction.',
                level=Logger.Levels.info,
            ).log()
        return shards

    def _load_cached_encodings(
        self, hps: List[Hyperplane], symbols: List[sp.Symbol]
    ) -> Optional[Dict[Tuple[int, ...], Position]]:
        """
        Load previously-computed shards from the ``<cmf>__shards.jsonl``
        cache so extraction can be skipped.

        Returns a mapping ``{sign-encoding: interior-point}`` rebuilt
        from the cached :class:`ShardDTO` records, or :data:`None` when
        there is no usable cache (no ``EXPORT_CMFS`` configured, missing
        / empty file, or a stale cache whose encodings no longer match
        the current hyperplane count).

        Hyperplanes are recomputed by the caller and passed in:
        ``_extract_cmf_hps`` returns them in a canonical, deterministic
        order, so ``encoding[i]`` still labels ``hps[i]`` exactly as it
        did when the cache was written.
        """
        if not sys_config.EXPORT_CMFS:
            return None

        dtos = read_shard_records(
            sys_config.EXPORT_CMFS, self.cmf_data.cmf_name
        )
        if not dtos:
            return None

        n = len(hps)
        out: Dict[Tuple[int, ...], Position] = {}
        for dto in dtos:
            enc = tuple(int(v) for v in dto.shard_encoding)
            if len(enc) != n:
                # Cache was written for a different hyperplane set — the
                # CMF or its hyperplanes changed.  Treat as stale and
                # force a fresh extraction rather than mis-aligning signs.
                Logger(
                    f'Ignoring stale shard cache (encoding length {len(enc)} '
                    f'!= {n} hyperplanes) for "{self.cmf_data.cmf_name}"',
                    level=Logger.Levels.warning,
                ).log()
                return None
            point = None
            if dto.interior_point is not None:
                # ``sympify`` (not ``int``) so a stored rational coordinate like
                # "7/2" (from a rational shift) is restored to its exact value;
                # plain ints round-trip unchanged.
                point = Position(
                    {sym: sp.sympify(v) for sym, v in zip(symbols, dto.interior_point)}
                )
            out[enc] = point
        return out

    def _discover_via_legacy(
        self, hps: List[Hyperplane], symbols: List[sp.Symbol]
    ) -> Dict[Tuple[int, ...], Position]:
        """
        Original brute-force lattice scan in
        :mod:`dreamer.extraction.utils.initial_points`.

        Preserved verbatim from the pre-v2 implementation so the
        ``legacy`` strategy remains a byte-for-byte fallback.
        """
        hps_list = list(hps)
        shifted_hps = [hp.apply_shift(self.cmf_data.shift) for hp in hps_list]
        A = np.array([hp.vectors[0] for hp in shifted_hps], dtype=np.int64)
        b = np.array([hp.vectors[1] for hp in shifted_hps], dtype=np.int64)
        S = config.extraction.INIT_POINT_MAX_COORD * 2 + 1
        cpus = cpus if (cpus := os.cpu_count()) else 1
        prefix_dims = max(min(int(round(math.log(cpus, S))), cpus - 1), 1)

        symmetries_func = None
        if issubclass(self.cmf_data.cmf.__class__, rt_pFq) and config.extraction.IGNORE_DUPLICATE_SEARCHABLES:
            symmetries_func = partial(init_points.filter_symmetrical_cones,
                                      p=self.cmf_data.cmf.p,
                                      q=self.cmf_data.cmf.q,
                                      shift=list(self.cmf_data.shift.values()))
        final_results = init_points.compute_mapping(
            self.cmf_data.cmf.dim(), S, A, b, prefix_dims, symmetries_func
        )
        unique_sigs = list(final_results.keys())
        decoded_vectors = init_points.decode_signatures(unique_sigs, len(hps))
        out: Dict[Tuple[int, ...], Position] = {}
        for i, sig in enumerate(unique_sigs):
            sign_vector = decoded_vectors[i]
            if 0 in sign_vector:
                continue
            actual_point = final_results[sig]
            out[tuple(sign_vector)] = Position(
                {sym: int(v) + self.cmf_data.shift[sym] for sym, v in zip(symbols, actual_point)}
            )
        return out

    def _discover_via_v2(
        self,
        hps: List[Hyperplane],
        symbols: List[sp.Symbol],
        strategy: str,
    ) -> Dict[Tuple[int, ...], Position]:
        """
        Route through the v2 :class:`ExtractionManager`.

        The v2 module works on the *shifted* hyperplanes (so that the
        integer point it returns lives in the shifted lattice) and
        labels each shard by a ``+/-1`` sign tuple ordered identically
        to the input list -- matching how :class:`Shard.encoding` is
        interpreted downstream.  Integer witnesses are translated back
        to absolute coordinates by adding the shift.

        :param strategy: One of ``"auto" | "exact" | "heuristic"``.
        """
        # Build the CMF family's symmetry strategy (canonical teleportation)
        # when symmetry reduction is requested.  The strategy operates in the
        # *shifted* lattice coordinates the v2 module uses (column order =
        # cmf.matrices.keys()), so it carries the per-coordinate shift.
        symmetry = None
        if config.extraction.IGNORE_DUPLICATE_SEARCHABLES:
            symmetry = symmetry_for_cmf(
                self.cmf_data.cmf, list(self.cmf_data.shift.values())
            )

        hps_list = list(hps)
        shifted_hps = [hp.apply_shift(self.cmf_data.shift) for hp in hps_list]

        manager = ExtractionManager(
            strategy=strategy,
            timeout_seconds=config.extraction.EXACT_TIMEOUT_SECONDS,
            exact_unbounded_check=config.extraction.EXACT_UNBOUNDED_CHECK,
            exact_num_workers=config.extraction.EXACT_NUM_WORKERS,
            heuristic_refine=config.extraction.HEURISTIC_REFINE_WITNESSES,
            heuristic_refine_threshold=config.extraction.HEURISTIC_REFINE_L1_THRESHOLD,
            heuristic_refine_workers=config.extraction.HEURISTIC_REFINE_WORKERS,
            heuristic_num_rays=config.extraction.HEURISTIC_NUM_RAYS,
            heuristic_max_seconds=config.extraction.HEURISTIC_TIMEOUT_SECONDS,
            heuristic_missing_mass=config.extraction.HEURISTIC_MISSING_MASS,
            heuristic_face_aligned=config.extraction.HEURISTIC_FACE_ALIGNED,
            heuristic_face_subsets=config.extraction.HEURISTIC_FACE_SUBSETS,
            heuristic_face_offsets=config.extraction.HEURISTIC_FACE_OFFSETS,
            symmetry=symmetry,
        )
        mapping = manager.extract(shifted_hps)

        out: Dict[Tuple[int, ...], Position] = {}
        for sig, point in mapping.items():
            out[tuple(sig)] = Position(
                {sym: int(v) + self.cmf_data.shift[sym] for sym, v in zip(symbols, point)}
            )
        return out
