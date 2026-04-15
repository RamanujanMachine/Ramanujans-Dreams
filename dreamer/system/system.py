from collections import defaultdict
from ramanujantools.cmf import CMF
from functools import partial
from typing import List, Dict, Optional, Type, Union, Set, Any, Iterable
import sympy as sp
import networkx as nx
from itertools import combinations
import os

from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.analysis_scheme import AnalyzerModScheme
from dreamer.utils.schemes.db_scheme import DBModScheme
from dreamer.loading.funcs.formatter import Formatter
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.schemes.extraction_scheme import ExtractionModScheme
from dreamer.utils.storage import Exporter, Importer, Formats
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
                 searcher: Type[SearcherModScheme] | partial[SearcherModScheme]):
        """
        Constructing a system runnable instance for a given combination of modules.
        :param function_sources: A list of DBModScheme instances used as sources.
        :param extractor: An optional ExtractionModScheme type used to extract shards from the CMFs.
        If extractor not provided, analysis will try to read from the default searchables directory.
        :param analyzers: Optional list of AnalyzerModScheme types used for prioritization + preparation before search.
            If omitted, priorities are loaded from sys_config.EXPORT_ANALYSIS_PRIORITIES.
        :param searcher: A SearcherModScheme type used to deepen the search done by the analyzers
        """
        if not isinstance(function_sources, list):
            raise ValueError('Inspiration Functions must be contained in a list')

        self.func_srcs = function_sources
        self.extractor = extractor
        self.analyzers = analyzers or []
        self.searcher = searcher
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
        cmf_data = self.__loading_stage(constants)
        relevant_cmf_names = self.__derive_relevant_cmf_names(constants, cmf_data)

        if path := sys_config.EXPORT_CMFS:
            os.makedirs(path, exist_ok=True)

            for const, l in cmf_data.items():
                safe_key = "".join(c for c in const.name if c.isalnum() or c in ('-', '_'))
                const_path = os.path.join(path, safe_key)

                for data in l:
                    Exporter.export(
                        root=const_path, f_name=data.cmf_name, exists_ok=True, clean_exists=True,
                        data=[data],
                        fmt=Formats.PICKLE
                    )
                Logger(
                    f'CMFs for {const.name} exported to {const_path}', Logger.Levels.info
                ).log()

        # print constants and CMFs
        for constant, funcs in cmf_data.items():
            functions = '\n'
            for i, func in enumerate(funcs):
                if func.cmf.__class__ == CMF:
                    pretty_mats = '\n\n>>> '.join(
                        f'{sp.pretty(sym, use_unicode=True)}:\n{sp.pretty(mat, use_unicode=True)}'
                        for sym, mat in func.cmf.matrices.items()
                    )
                    functions += f'{i+1}. CMF: \n>>>{pretty_mats}\n with offset {tuple(func.shift.values())}\n'
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
        elif sys_config.PATH_TO_SEARCHABLES:
            shard_dict = self.__import_searchables(sys_config.PATH_TO_SEARCHABLES, constants, relevant_cmf_names)

        # =======================================================
        # ANALYSIS STAGE - analyzes shards and prioritize search
        # =======================================================
        self._analysis_constants = constants
        self._analysis_relevant_cmf_names = relevant_cmf_names
        priorities = self.__analysis_stage(shard_dict)
        Logger.timer_summary()

        # Store priorities to be used in the search stage and future runs
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
                    Exporter.export(
                        root=const_path,
                        f_name=self.__safe_fs_name(cmf_name),
                        exists_ok=True,
                        clean_exists=False,
                        fmt=Formats.PICKLE,
                        data=spaces,
                    )
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

    def __loading_stage(self, constants: List[Constant]) -> Dict[Constant, List[CMFData]]:
        """
        Preforms the loading of the inspiration functions from various sources
        :param constants: A list of all constants relevant to this run
        :return: A mapping from a constant to the list of its CMFs (matching the inspiration functions)
        """
        if not self.func_srcs:
            return dict()

        Logger('Loading CMFs ...', Logger.Levels.info).log()
        modules = []
        cmf_data = defaultdict(set)

        for db in self.func_srcs:
            if isinstance(db, DBModScheme):
                modules.append(db)
            elif isinstance(db, str):
                shift_cmf = Importer.imprt(db)
                cmf_data[Constant.get_constant(db.split('/')[-2])].add(shift_cmf[0])
            elif isinstance(db, Formatter):
                cmf_data[Constant.get_constant(db.const)].add(db.to_cmf())
            else:
                raise ValueError(f'Unknown format: {db} (accepts only str | DBModScheme | Formatter)')

        # If DB were used, aggregate extracted constants
        cmf_data_2 = dict()
        if modules:
            cmf_data_2 = DBModScheme.aggregate(modules, constants, True)
        for const in cmf_data_2.keys():
            cmf_data[const].update(cmf_data_2[const])

        # convert back to dict[str, list]
        as_list = dict()
        for k, v in cmf_data.items():
            if k not in constants:
                Logger(
                    f'constant {k} is not in the search list, its inspiration function(s) will be ignored',
                    level=Logger.Levels.warning
                ).log()
                continue
            as_list[k] = list(v)
        return as_list

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
                for entry in sorted(os.listdir(const_dir)):
                    entry_path = os.path.join(const_dir, entry)
                    entry_stem = os.path.splitext(entry)[0]
                    if entry not in allowed_cmf_names and entry not in allowed_safe \
                            and entry_stem not in allowed_cmf_names and entry_stem not in allowed_safe:
                        continue

                    if os.path.isdir(entry_path):
                        for imported in Importer.import_stream(entry_path):
                            const_shards.extend(self.__iter_searchables(imported))
                    elif os.path.isfile(entry_path) and entry_path.endswith(f'.{Formats.PICKLE.value}'):
                        const_shards.extend(self.__iter_searchables(Importer.imprt(entry_path)))

            if const_shards:
                shard_dict[const] = const_shards
        return shard_dict

    def __import_priorities(
            self,
            constants: List[Constant],
            relevant_cmf_names: Dict[str, Optional[Set[str]]]
    ) -> Dict[Constant, List[Searchable]]:
        """
        Load priorities from export path in arbitrary order when analyzers are not provided.
        Expected layout: <priorities_root>/<constant>/<cmf>.pkl
        :param constants: Constants requested for the current run.
        :param relevant_cmf_names: Per-constant set of allowed cmf_name values, or None to allow all CMFs.
        :return: Mapping from constant object to imported priority list.
        """
        path = sys_config.EXPORT_ANALYSIS_PRIORITIES
        if not path or not os.path.isdir(path):
            return {}

        priorities: Dict[Constant, List[Searchable]] = {}
        for const in constants:
            const_path = os.path.join(path, const.name)
            if not os.path.isdir(const_path):
                continue

            allowed_cmf_names = relevant_cmf_names.get(const.name)
            allowed_safe = {self.__safe_fs_name(name) for name in allowed_cmf_names} if allowed_cmf_names else set()
            spaces: List[Searchable] = []

            for f_name in sorted(os.listdir(const_path)):
                file_path = os.path.join(const_path, f_name)
                if not os.path.isfile(file_path) or not f_name.endswith(f'.{Formats.PICKLE.value}'):
                    continue

                cmf_stem = os.path.splitext(f_name)[0]
                if allowed_cmf_names is not None and cmf_stem not in allowed_cmf_names and cmf_stem not in allowed_safe:
                    continue

                spaces.extend(self.__iter_searchables(Importer.imprt(file_path)))

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
        Preform deep search using the provided search module
        :param priorities: a list prioritized searchables for each constant
        """
        # Execute searchers
        for data in priorities.values():
            self.searcher(data, sys_config.USE_LIReC).execute()

        # Print best delta for each constant
        for const in priorities.keys():
            best_delta = -sp.oo
            best_sv = None
            dir_path = os.path.join(sys_config.EXPORT_SEARCH_RESULTS, const.name)

            # TODO: we first need to read inside the directories
            stream_gen = Importer.import_stream(dir_path)
            for dm in stream_gen:
                delta, sv = dm.best_delta
                if delta is None:
                    continue
                if best_delta < delta:
                    best_delta, best_sv = delta, sv

            if best_sv is None:
                # Should not happen
                Logger('No best delta found').log()
            else:
                Logger(
                    f'Best delta for "{const.name}" found by the searcher is {best_delta}\n'
                    f'* Trajectory: {best_sv.trajectory} \n* Start: {best_sv.start}',
                    Logger.Levels.info
                ).log()

        # delete temp directory
        if sys_config.EXPORT_SEARCH_RESULTS.split('.')[-1] == sys_config.DEFAULT_DIR_SUFFIX:
            os.rmdir(sys_config.EXPORT_SEARCH_RESULTS)

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
