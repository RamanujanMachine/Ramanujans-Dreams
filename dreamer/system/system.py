from collections import defaultdict
from ramanujantools.cmf import CMF
from functools import partial
from typing import List, Dict, Optional, Type, Union, Set, Any, Iterable
import networkx as nx
from itertools import combinations
import os
import json as _json
import math

from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.analysis_scheme import AnalyzerModScheme
from dreamer.utils.schemes.db_scheme import DBModScheme
from dreamer.loading.funcs.formatter import Formatter
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.schemes.post_process_scheme import PostProcessModScheme
from dreamer.utils.schemes.extraction_scheme import ExtractionModScheme
from dreamer.utils.storage import Importer, Formats
from dreamer.utils.storage.atlas_writer import (
    load_shards_from_export,
    update_found_constants,
    write_cmf_records,
    read_shard_records,
    reconstruct_shard_from_dto
)
from dreamer.utils.storage.summary import write_summary
from dreamer.utils.storage.optimization_objectives import (
    objective_display_label,
    record_raw_value,
    score_record,
)
from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids
from dreamer.utils.types import CMFData
from dreamer.utils.logger import Logger
from dreamer.utils.constants.constant import Constant
from dreamer.configs.system import sys_config

constant_type = Union[Constant, str]


class System:
    """
    System class wraps together all given modules and connects them.
    """

    def __init__(self,
                 *,
                 function_sources: List[DBModScheme | str | Formatter],
                 extractor: Optional[Type[ExtractionModScheme]] = None,
                 analyzers: Optional[List[Type[AnalyzerModScheme] | partial[AnalyzerModScheme] | str | Searchable]] = None,
                 searcher: Type[SearcherModScheme] | partial[SearcherModScheme],
                 post_processor: Optional[Type[PostProcessModScheme] | partial[PostProcessModScheme]] = None):
        """
        Constructing a system runnable instance for a given combination of modules.
        :param function_sources: A list of DBModScheme instances used as sources.
        :param extractor: An optional ExtractionModScheme type used to extract shards from the CMFs.
        If extractor not provided, analysis will try to read from the default searchables directory.
        :param analyzers: Optional list of AnalyzerModScheme types used for prioritization + preparation before search.
            If omitted, priorities are loaded from sys_config.EXPORT_ANALYSIS_PRIORITIES.
        :param searcher: A SearcherModScheme type used to deepen the search done by the analyzers
        :param post_processor: Optional ``PostProcessModScheme`` type that runs after Search to
            compute expensive Tier-3 attributes.  When omitted, no post-process stage runs.
        """
        if not isinstance(function_sources, list):
            raise ValueError('Inspiration Functions must be contained in a list')

        self.func_srcs = function_sources
        self.extractor = extractor
        self.analyzers = analyzers or []
        self.searcher = searcher
        self.post_processor = post_processor
        self._analysis_constants: List[Constant] = []
        self._analysis_relevant_cmf_names: Dict[str, Optional[Set[str]]] = {}

        if not self.func_srcs and self.extractor:
            raise ValueError('Could not preform extraction if no sourced to extract from where provided')

    def run(self, constants: Optional[List[str | Constant] | str | Constant] = None):
        """
        Run the system given the constants to search for.
        :param constants: if None, search for constants defined in the configuration file in 'configs.database.py'.
        """
        Logger.start_run()

        constants: List[Constant] = self.__validate_constants(constants)

        # ======================================================
        # LOAD STAGE - loading constants (and optional storage)
        # ======================================================
        cmf_data, cmf_formatters = self.__loading_stage(constants)
        relevant_cmf_names = self.__derive_relevant_cmf_names(constants, cmf_data)

        if path := sys_config.EXPORT_CMFS:
            os.makedirs(path, exist_ok=True)

            seen_cmf_ids: set = set()
            all_unique_cmf_data = []

            for const, l in cmf_data.items():
                safe_key = "".join(c for c in const.name if c.isalnum() or c in ('-', '_'))
                const_path = os.path.join(path, safe_key)
                os.makedirs(const_path, exist_ok=True)

                for data in l:
                    # Write formatter JSON so shards can be reconstructed without pickle.
                    formatter = cmf_formatters.get(data.cmf_name)
                    if formatter is not None:
                        fmt_path = os.path.join(const_path, f"{data.cmf_name}.json")
                        with open(fmt_path, "w") as _fh:
                            _fh.write(_json.dumps(formatter.to_json_obj()))

                    if id(data) not in seen_cmf_ids:
                        seen_cmf_ids.add(id(data))
                        all_unique_cmf_data.append(data)

                Logger(
                    f'CMFs for {const.name} exported to {const_path}', Logger.Levels.info
                ).log()

            # DB-ready DTO records: flat cmfs.jsonl / cmf_families.jsonl at root.
            # found_constants written empty here — updated after analysis
            # via update_found_constants() to reflect only actually-found constants.
            write_cmf_records(path, all_unique_cmf_data)

        # print constants and CMFs
        for constant, funcs in cmf_data.items():
            functions = '\n'
            for i, func in enumerate(funcs):
                if func.cmf.__class__ == CMF:
                    # pretty_mats = '\n\n>>> '.join(
                    #     f'{sp.pretty(sym, use_unicode=True)}:\n{sp.pretty(mat, use_unicode=True)}'
                    #     for sym, mat in func.cmf.matrices.items()
                    # )
                    functions += f'{i+1}. CMF: {func.cmf_name} with offset {tuple(func.shift.values())}\n'
                else:
                    functions += f'{i+1}. CMF: {repr(func.cmf)} with offset {tuple(func.shift.values())}\n'
            Logger(
                f'Searching for {constant.name} using inspiration functions: {functions}', Logger.Levels.info
            ).log()

        # ====================================================
        # EXTRACTION STAGE - computing shards and saving them
        # ====================================================
        shard_dict = dict()
        if self.extractor:
            shard_dict = self.extractor(cmf_data).execute()
        elif sys_config.EXPORT_CMFS and os.path.isdir(sys_config.EXPORT_CMFS):
            # Reconstruct shards from JSONL + formatter JSON (no pickle needed).
            shard_dict = load_shards_from_export(
                sys_config.EXPORT_CMFS, constants, relevant_cmf_names
            )
            if not shard_dict:
                Logger(
                    "No pre-extracted shards found in EXPORT_CMFS; skipping extraction.",
                    Logger.Levels.warning,
                ).log()

        # =======================================================
        # ANALYSIS STAGE - analyzes shards and prioritize search
        # =======================================================
        self._analysis_constants = constants
        self._analysis_relevant_cmf_names = relevant_cmf_names
        priorities = self.__analysis_stage(shard_dict)
        Logger.timer_summary()

        # Update found_constants in cmfs.jsonl for every CMF that was
        # actually identified in the analysis stage.
        if (path := sys_config.EXPORT_CMFS) and priorities:
            seen_updates: set = set()
            for const, shards in priorities.items():
                for shard in shards:
                    cmf_name = getattr(shard, 'cmf_name', None)
                    key = (cmf_name, const.name)
                    if cmf_name and key not in seen_updates:
                        seen_updates.add(key)
                        update_found_constants(path, cmf_name, [const.name])

        # Store priorities to be used in the search stage and future runs.
        # Written as shard-ID JSON (no pickle) so priorities are stable across
        # code changes and reconstructable via load_shards_from_export.
        filtered_priorities = dict()
        bad_run = False
        if path := sys_config.EXPORT_ANALYSIS_PRIORITIES:
            os.makedirs(path, exist_ok=True)

            for const, l in priorities.items():
                if not l:
                    Logger(
                        f'No shards remained after analysis. Run for constant "{const.name}" is stopped.',
                        Logger.Levels.warning
                    ).log()
                    continue

                const_path = os.path.join(path, const.name)
                os.makedirs(const_path, exist_ok=True)
                grouped = self.__group_searchables_by_cmf_name(l)

                for cmf_name, spaces in grouped.items():
                    shard_ids = []
                    for s in spaces:
                        try:
                            _, sid, _ = derive_cmf_and_shard_ids(s)
                            shard_ids.append(sid)
                        except Exception:
                            pass
                    record = {"cmf_name": cmf_name, "const_name": const.name, "shard_ids": shard_ids}
                    out_path = os.path.join(const_path, self.__safe_fs_name(cmf_name) + ".json")
                    with open(out_path, "w") as _fh:
                        _fh.write(_json.dumps(record))

                    Logger(
                        f'--- Found {len(shard_ids)} shards containing {const.name} in {cmf_name} ---', Logger.Levels.info
                    ).log()

                Logger(
                    f'Priorities for {const.name} exported to {const_path}', Logger.Levels.info
                ).log()
                filtered_priorities[const] = l

            if not filtered_priorities:
                bad_run = True

        if bad_run or not priorities:
            Logger('No relevant shards found, run stopped', Logger.Levels.warning).log()
            return

        # =======================================================
        # SEARCH STAGE - preform deep search
        # =======================================================
        if len(filtered_priorities) == 0:
            filtered_priorities = priorities
        self.__search_stage(filtered_priorities)

        # =======================================================
        # POST-PROCESS STAGE - expensive Tier-3 attributes
        # =======================================================
        # Only runs when both a post-processor is wired and TIER3_ATTRIBUTES
        # is non-empty.  The module itself short-circuits on the config check,
        # but we also skip the call when no post-processor was supplied.
        if self.post_processor is not None:
            self.post_processor(filtered_priorities).execute()

        # =======================================================
        # SUMMARY - write a human-readable summary.md of the run
        # =======================================================
        # Always attempt to write summary.md so the run leaves a self-describing
        # artifact behind, even if earlier stages produced little data.
        # ``this_run_shards`` scopes the summary to the shards *this* pipeline
        # invocation actually touched: extraction sampling is stochastic, so
        # earlier runs may have left orphan JSONLs in the search-results dir
        # with a different shard_encoding hash — without the filter those
        # would silently inflate the shard count above what the extraction
        # stage reports.  Failures here are non-fatal: the data on disk is
        # still intact even if rendering breaks.
        try:
            # Source the "this-run" shard set from ``shard_dict`` (extraction
            # output), not ``priorities`` — the analyzer writes per-shard
            # JSONLs for *every* shard it sampled, including ones that
            # didn't pass the identification threshold and so don't make it
            # into priorities.  We want all of them in the summary.
            this_run_shards: Dict[str, Set[str]] = defaultdict(set)
            for const, shards in shard_dict.items():
                for shard in shards:
                    try:
                        _, shard_id, _ = derive_cmf_and_shard_ids(shard)
                    except Exception as _exc:
                        Logger(
                            f"Failed to derive shard id for summary filter: {_exc}",
                            Logger.Levels.warning,
                        ).log()
                        continue
                    this_run_shards[const.name].add(shard_id)
            # Always pass the dict (even if empty) so the summary is scoped
            # to this run's shards only and never shows orphan JSONLs from
            # earlier runs.  An empty dict means "extraction found nothing."
            summary_path = write_summary(
                search_results_root=sys_config.EXPORT_SEARCH_RESULTS,
                export_cmfs_root=sys_config.EXPORT_CMFS,
                this_run_shards=dict(this_run_shards),
            )
            if summary_path:
                Logger(
                    f"Run summary written to {summary_path}",
                    Logger.Levels.info,
                ).log()
        except Exception as e:
            Logger(
                f"Failed to write run summary: {e}",
                Logger.Levels.warning,
            ).log()

    def __loading_stage(self, constants: List[Constant]):
        """
        Preforms the loading of the inspiration functions from various sources.

        Formatters may now declare multiple constants (``fmt.consts``).  The
        same ``CMFData`` object is added under every constant in that list
        that is also present in the run's *constants* list.  Constants
        declared by a formatter but absent from the run are logged as a
        warning and dropped.

        :param constants: A list of all constants relevant to this run
        :return: Tuple of (cmf_data mapping, formatter mapping).
            ``cmf_data`` maps each constant to the list of its CMFData objects.
            ``cmf_formatters`` maps ``cmf_name → Formatter`` for every
            Formatter-sourced CMF so the export stage can write the formatter
            JSON (the canonical no-pickle shard-reconstruction artefact).
        """
        if not self.func_srcs:
            return dict(), dict()

        Logger('Loading CMFs ...', Logger.Levels.info).log()
        modules = []
        cmf_data: Dict[Constant, set] = defaultdict(set)
        cmf_formatters: Dict[str, Formatter] = {}  # cmf_name → Formatter
        constants_set = set(constants)

        for db in self.func_srcs:
            if isinstance(db, DBModScheme):
                modules.append(db)
            elif isinstance(db, str):
                shift_cmf = Importer.imprt(db)
                cmf_data[Constant.get_constant(db.split('/')[-2])].add(shift_cmf[0])
            elif isinstance(db, Formatter):
                cmf = db.to_cmf()
                cmf_formatters[cmf.cmf_name] = db  # record for JSON export
                fmt_consts = [Constant.get_constant(name) for name in db.consts]
                ignored = [c.name for c in fmt_consts if c not in constants_set]
                active = [c for c in fmt_consts if c in constants_set]
                if ignored:
                    Logger(
                        f'Constants ignored for CMF {db.cmf_name}: {ignored}',
                        level=Logger.Levels.warning,
                    ).log()
                for c in active:
                    cmf_data[c].add(cmf)
            else:
                raise ValueError(f'Unknown format: {db} (accepts only str | DBModScheme | Formatter)')

        # If DB were used, aggregate extracted constants
        cmf_data_2 = dict()
        if modules:
            cmf_data_2 = DBModScheme.aggregate(modules, constants, True)
        for const in cmf_data_2.keys():
            cmf_data[const].update(cmf_data_2[const])

        # convert back to dict[Constant, list], filtering to run constants only
        as_list = dict()
        for k, v in cmf_data.items():
            if k not in constants_set:
                Logger(
                    f'constant {k} is not in the search list, its inspiration function(s) will be ignored',
                    level=Logger.Levels.warning
                ).log()
                continue
            as_list[k] = list(v)
        return as_list, cmf_formatters

    def __analysis_stage(
            self,
            cmf_data: Optional[Dict[Constant, List[Searchable]]] = None,
            constants: Optional[List[Constant]] = None,
            relevant_cmf_names: Optional[Dict[str, Optional[Set[str]]]] = None
    ) -> Dict[Constant, List[Searchable]]:
        """
        Preform the analysis stage work
        :param cmf_data: Mapping from constants to candidate searchables produced by extraction/loading.
        :param constants: Optional validated constants list. When omitted, uses constants cached by run().
        :param relevant_cmf_names: Optional per-constant CMF-name filters. None per constant means no CMF filter.
        :return: The results of the analysis as a mapping from constant to a list of prioritized searchables.
        """
        if constants is None:
            constants = self._analysis_constants
        if relevant_cmf_names is None:
            relevant_cmf_names = self._analysis_relevant_cmf_names

        if not self.analyzers:
            return self.__import_priorities(constants or [], relevant_cmf_names or dict())

        analyzers: List[Type[AnalyzerModScheme] | partial[AnalyzerModScheme]] = []
        results = defaultdict(set)

        # prepare analyzers
        for analyzer in self.analyzers:
            match analyzer:
                case t if isinstance(t, type) and issubclass(t, AnalyzerModScheme):
                    analyzers.append(analyzer)
                case t if isinstance(t, partial) and isinstance(t.func, type) and issubclass(t.func, AnalyzerModScheme):
                    analyzers.append(analyzer)
                case Searchable():
                    results[analyzer.const].add(analyzer)
                case str():
                    f_data = Importer.imprt(analyzer)
                    for obj in self.__iter_searchables(f_data):
                        results[obj.const].add(obj)
                case _:
                    raise TypeError(f'unknown analyzer type {analyzer}')

        # Load saved shards from the default directory if data not provided
        # if not cmf_data:
        #     cmf_data = {}
        #     for const_name in os.listdir(sys_config.PATH_TO_SEARCHABLES):
        #         import_stream = Importer.import_stream(f'{sys_config.PATH_TO_SEARCHABLES}\\{const_name}')
        #         const_shards = []
        #         for shards in import_stream:
        #             const_shards += shards
        #         if const_shards:
        #             cmf_data[const_shards[0].const] = const_shards

        analyzers_results = [analyzer(cmf_data or dict()).execute() for analyzer in analyzers]
        priorities = self.__compact_analysis_results(analyzers_results) if analyzers_results else {}

        # add unprioritized elements to the end
        for c, l in results.items():
            if c not in priorities:
                priorities[c] = list(l)
            else:
                existing = set(priorities[c])
                priorities[c].extend([space for space in l if space not in existing])
        return priorities

    def __import_searchables(
            self,
            root_path: str,
            constants: List[Constant],
            relevant_cmf_names: Dict[str, Optional[Set[str]]]
    ) -> Dict[Constant, List[Searchable]]:
        """
        Load saved searchables filtered by relevant constants and optional CMF names.
        :param root_path: Root directory containing searchable exports in constant subdirectories.
        :param constants: Constants requested for the current run.
        :param relevant_cmf_names: Per-constant set of allowed cmf_name values, or None to allow all CMFs.
        :return: Mapping from constant object to imported searchable list.
        """
        if not root_path or not os.path.isdir(root_path):
            return {}

        shard_dict: Dict[Constant, List[Searchable]] = {}
        for const in constants:
            const_dir = os.path.join(root_path, const.name)
            if not os.path.isdir(const_dir):
                continue

            allowed_cmf_names = relevant_cmf_names.get(const.name)
            const_shards: List[Searchable] = []

            if allowed_cmf_names is None:
                for imported in Importer.import_stream(const_dir):
                    const_shards.extend(self.__iter_searchables(imported))
            else:
                allowed_safe = {self.__safe_fs_name(name) for name in allowed_cmf_names}
                searchables_ext = f'.{Formats(sys_config.EXPORT_SEARCHABLES_FORMAT).value}'
                for entry in sorted(os.listdir(const_dir)):
                    entry_path = os.path.join(const_dir, entry)
                    entry_stem = os.path.splitext(entry)[0]
                    if entry not in allowed_cmf_names and entry not in allowed_safe \
                            and entry_stem not in allowed_cmf_names and entry_stem not in allowed_safe:
                        continue

                    if os.path.isdir(entry_path):
                        for imported in Importer.import_stream(entry_path):
                            const_shards.extend(self.__iter_searchables(imported))
                    elif os.path.isfile(entry_path) and entry_path.endswith(searchables_ext):
                        const_shards.extend(self.__iter_searchables(Importer.imprt(entry_path)))

            if const_shards:
                shard_dict[const] = const_shards
        return shard_dict

    def __import_priorities(
            self,
            constants: List[Constant],
            relevant_cmf_names: Dict[str, Optional[Set[str]]]
    ) -> Dict[Constant, List[Searchable]]:
        """Load priorities from the shard-ID JSON files written by the analysis stage.

        Expected layout: ``<priorities_root>/<const>/<safe_cmf_name>.json``
        Each JSON file contains ``{"cmf_name": ..., "const_name": ..., "shard_ids": [...]}``.
        Shards are reconstructed via :func:`~dreamer.utils.storage.atlas_writer.load_shards_from_export`
        (formatter JSON + ``<cmf>__shards.jsonl`` in ``EXPORT_CMFS``).

        :param constants: Constants requested for the current run.
        :param relevant_cmf_names: Per-constant set of allowed cmf_name values, or None for all.
        :return: Mapping from constant to ordered shard list.
        """
        prio_path = sys_config.EXPORT_ANALYSIS_PRIORITIES
        export_root = sys_config.EXPORT_CMFS

        if not prio_path or not os.path.isdir(prio_path):
            return {}

        priorities: Dict[Constant, List[Searchable]] = {}

        for const in constants:
            const_prio_path = os.path.join(prio_path, const.name)
            if not os.path.isdir(const_prio_path):
                continue

            allowed_cmf_names = relevant_cmf_names.get(const.name)
            allowed_safe = {self.__safe_fs_name(n) for n in allowed_cmf_names} if allowed_cmf_names else None
            spaces: List[Searchable] = []

            for f_name in sorted(os.listdir(const_prio_path)):
                if not f_name.endswith(".json"):
                    continue
                file_path = os.path.join(const_prio_path, f_name)
                try:
                    with open(file_path) as fh:
                        record = _json.load(fh)
                except Exception:
                    continue

                cmf_name = record.get("cmf_name", "")
                shard_ids_wanted = set(record.get("shard_ids", []))
                cmf_stem = os.path.splitext(f_name)[0]
                if allowed_safe is not None and cmf_stem not in (allowed_safe or set()):
                    continue
                if not shard_ids_wanted or not export_root:
                    continue

                # Load CMFData from formatter JSON stored in EXPORT_CMFS/<const>/<cmf>.json.
                safe_key = "".join(c for c in const.name if c.isalnum() or c in ("-", "_"))
                fmt_json_path = os.path.join(export_root, safe_key, f"{cmf_name}.json")
                try:
                    with open(fmt_json_path) as fh:
                        fmt_raw = _json.load(fh)
                    formatter = Formatter.from_json_obj(fmt_raw)
                    cmf_data = formatter.to_cmf()
                except Exception as exc:
                    Logger(
                        f"__import_priorities: could not load formatter for {cmf_name}: {exc}",
                        Logger.Levels.warning,
                    ).log()
                    continue

                dtos = read_shard_records(export_root, cmf_name)
                for dto in dtos:
                    if dto.shard_id not in shard_ids_wanted:
                        continue
                    shard = reconstruct_shard_from_dto(dto, cmf_data, [const])
                    if shard is not None:
                        spaces.append(shard)

            if spaces:
                priorities[const] = spaces
        return priorities

    @staticmethod
    def __derive_relevant_cmf_names(
            constants: List[Constant],
            cmf_data: Dict[Constant, List[CMFData]]
    ) -> Dict[str, Optional[Set[str]]]:
        """
        Build per-constant CMF name filters derived from loaded function sources.
        None means all CMFs under that constant are considered relevant.
        :param constants: Constants requested for the current run.
        :param cmf_data: Mapping from constant to loaded CMFData objects.
        :return: Mapping from constant-name string to allowed CMF-name set (or None for no filter).
        """
        filters: Dict[str, Optional[Set[str]]] = {const.name: None for const in constants}
        for const in constants:
            cmfs = cmf_data.get(const, [])
            cmf_names = {data.cmf_name for data in cmfs if data and getattr(data, 'cmf_name', None)}
            if cmf_names:
                filters[const.name] = cmf_names
        return filters

    @staticmethod
    def __iter_searchables(data: Any) -> Iterable[Searchable]:
        """
        Recursively flatten imported payloads into Searchable objects.
        :param data: Imported payload that may contain nested dict/list/set structures.
        :return: Iterator of Searchable instances found in the payload.
        """
        if isinstance(data, Searchable):
            yield data
            return

        if isinstance(data, dict):
            for value in data.values():
                yield from System.__iter_searchables(value)
            return

        if isinstance(data, list | tuple | set):
            for value in data:
                yield from System.__iter_searchables(value)

    @staticmethod
    def __group_searchables_by_cmf_name(searchables: List[Searchable]) -> Dict[str, List[Searchable]]:
        """
        Group searchables by their cmf_name to support per-CMF export.
        :param searchables: Prioritized searchables for a single constant.
        :return: Mapping from cmf_name to the list of matching searchables.
        """
        grouped = defaultdict(list)
        for space in searchables:
            grouped[getattr(space, 'cmf_name', 'UnknownCMF')].append(space)
        return grouped

    @staticmethod
    def __safe_fs_name(name: str) -> str:
        """
        Convert a logical CMF name into a filesystem-safe file stem.
        :param name: Original CMF name.
        :return: Sanitized name containing alphanumerics, '-' and '_' only.
        """
        sanitized = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(name)).strip('_')
        return sanitized or 'unknown'

    def __search_stage(self, priorities: Dict[Constant, List[Searchable]]):
        """
        Preform deep search using the provided search module.
        :param priorities: mapping from each constant to its prioritized shards
        """
        # Pass the full priorities dict so the searcher can deduplicate shards
        # and compute only the identified constants per shard.
        self.searcher(priorities, sys_config.USE_LIReC).execute()

        # Print best delta for each constant — scoped to *this run's* searched
        # shards so a previous run on a different CMF (same constant) sharing the
        # flat results dir cannot bleed its best delta into this report.
        objective_name = sys_config.OPTIMIZATION_OBJECTIVE
        label = objective_display_label(objective_name)
        for const in priorities.keys():
            run_shard_ids: Set[str] = {
                derive_cmf_and_shard_ids(s)[1] for s in priorities.get(const, [])
            }
            best_record, best_value = self.__best_trajectory_record(
                const, run_shard_ids, objective_name
            )
            if best_record is None:
                Logger(
                    f'No trajectory results found for "{const.name}"',
                    Logger.Levels.warning,
                ).log()
                continue

            # δ is always meaningful (the irrationality measure); when optimising a
            # different objective, report both the objective value and δ.
            extra = ""
            if objective_name != "delta":
                # Records are flat per-(trajectory, constant): δ is the scalar
                # ``delta`` column on this constant's row (not a nested dict).
                delta_val = best_record.get("delta")
                if delta_val is not None:
                    extra = f"\n* δ:             {float(delta_val):.6f}"
            Logger(
                f'Best {label} for "{const.name}" is {best_value:.6f}\n'
                f'* Trajectory id: {best_record["trajectory_id"]}\n'
                f'* Start:         {tuple(best_record["start_point"])}\n'
                f'* Direction:     {tuple(best_record["direction"])}'
                f'{extra}',
                Logger.Levels.info,
            ).log()

        # delete temp directory
        if sys_config.EXPORT_SEARCH_RESULTS.split('.')[-1] == sys_config.DEFAULT_DIR_SUFFIX:
            if os.path.isdir(sys_config.EXPORT_SEARCH_RESULTS):
                try:
                    os.rmdir(sys_config.EXPORT_SEARCH_RESULTS)
                except OSError:
                    pass

    @staticmethod
    def __best_trajectory_record(const: Constant, shard_ids: Set[str], objective_name: str):
        """Return the best record for *const* under *objective_name* among **this
        run's** searched shards, plus that objective's raw (display) value.

        "Best" is the maximum **signed** objective score (via ``score_record``),
        so it is correct for both "larger is better" and "smaller is better"
        objectives; the returned value is the *raw* (unsigned) objective value for
        display.  Only the JSONL files ``EXPORT_SEARCH_RESULTS/<shard_id>.jsonl``
        for ``shard_id in shard_ids`` are scanned.  ``shard_ids`` must be the set of
        shard ids searched for *const* in the current run (derived from the
        ``priorities`` dict) — the results directory is flat and **persists across
        runs**, so scanning the whole directory would report a stale run's best.

        Returns ``(None, None)`` when the dir is missing, no listed shard file
        exists, or no record carries a finite objective score for this constant.
        """
        dir_path = sys_config.EXPORT_SEARCH_RESULTS
        if not os.path.isdir(dir_path):
            return None, None

        jsonl_ext = '.' + Formats.JSONL.value
        best_score: float = -float('inf')
        best_record = None

        for shard_id in sorted(shard_ids):
            fpath = os.path.join(dir_path, f"{shard_id}{jsonl_ext}")
            if not os.path.isfile(fpath):
                # Shard produced no JSONL (e.g. no trajectories searched).
                continue
            records = Importer.imprt(fpath)
            for record in records:
                # Rows are per-(trajectory, constant); only this constant's rows.
                if record.get("constant") != const.name:
                    continue
                scored = score_record(record, objective_name)
                if scored is None:
                    continue
                score, _ = scored
                if not math.isfinite(score):
                    continue
                if score > best_score:
                    best_score = score
                    best_record = record
        if best_record is None:
            return None, None
        return best_record, record_raw_value(best_record, objective_name)

    @staticmethod
    def __compact_analysis_results(dicts: List[Dict[Constant, List[Searchable]]]) -> Dict[Constant, List[Searchable]]:
        """
        Aggregates the priority lists from several analyzers into one
        :param dicts: A list of mappings from constant name to a list of its relevant subspaces
        :return: The aggregated priority dictionaries
        """
        all_keys = set().union(*dicts)
        result = {}

        for key in all_keys:
            lists = [d[key] for d in dicts if key in d]
            prefs = defaultdict(int)
            searchables = set().union(*lists)

            # Count preferences
            for lst in lists:
                for i, a in enumerate(lst[:-1]):
                    for j, b in enumerate(lst[i + 1:]):
                        prefs[(a, b)] += 1  # (j - i) * 1. / len(lst)

            G = nx.DiGraph()
            G.add_nodes_from(searchables)
            for a, b in combinations(searchables, 2):
                if prefs[(a, b)] > prefs[(b, a)]:
                    G.add_edge(a, b)
                elif prefs[(a, b)] < prefs[(b, a)]:
                    G.add_edge(b, a)
                else:
                    if hash(a) > hash(b):
                        G.add_edge(a, b)
                    else:
                        G.add_edge(b, a)

            try:
                consensus = list(nx.topological_sort(G))
            except nx.NetworkXUnfeasible:
                raise Exception('This was not supposed to happen')
            result[key] = consensus
        return result

    @staticmethod
    def __validate_constants(constants: Optional[List[str | Constant] | str | Constant] = None) -> List[Constant]:
        """
        Validates constants are in the correct format and usable
        :param constants: One or more Constant object or a constant name
        :return: A list of Constant objects
        """
        if not constants:
            Logger(
                'No constants provided, searching for all constants in configurations', Logger.Levels.warning
            ).log()
            constants = sys_config.CONSTANTS

        # prepare constants for loading
        if isinstance(constants, str | Constant):
            constants = [constants]
        as_obj = []
        for c in constants:
            if isinstance(c, str):
                if not Constant.is_registered(c):
                    raise ValueError(f'Constant "{c}" is not a registered constant.')
                as_obj.append(Constant.get_constant(c))
            else:
                as_obj.append(c)
        return as_obj
