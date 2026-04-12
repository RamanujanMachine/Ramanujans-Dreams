import os
from typing import List, Optional

from dreamer.configs.system import sys_config
from dreamer.search.methods.genetic import GeneticSearchMethod
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.extraction.shard import Shard
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.storage import Exporter, Formats
from dreamer.utils.ui.tqdm_config import SmartTQDM


class GeneticSearchMod(SearcherModScheme):
    """
    Searcher module that executes GA trajectory optimization per searchable shard.
    """

    def __init__(
        self,
        searchables: List[Shard],
        use_LIReC: Optional[bool] = True,
        generations: int = 25,
        pop_size: int = 40,
        max_coord_init: int = 10,
        elite_fraction: float = 0.2,
        mutation_prob: float = 0.3,
        mutation_step: int = 1,
        crossover_prob: float = 0.5,
        max_retries: int = 3,
        refine_prob: float = 0.5,
        refine_coord_prob: float = 0.5,
        parallel_eval: bool = True,
    ):
        """
        Initialize GA module-level hyperparameters used for each searchable.
        :param searchables: Shards/searchables to optimize.
        :param use_LIReC: Optional backend flag forwarded to trajectory evaluation.
        :param generations: Number of GA generations.
        :param pop_size: Population size per generation.
        :param max_coord_init: Reserved initialization bound forwarded to search method.
        :param elite_fraction: Fraction of elites retained each generation.
        :param mutation_prob: Probability to mutate each child.
        :param mutation_step: Max coordinate mutation step.
        :param crossover_prob: Probability to perform crossover.
        :param max_retries: Retry rounds for invalid evaluations.
        :param refine_prob: Probability to use refine mutation mode.
        :param refine_coord_prob: Per-coordinate refine perturbation probability.
        :param parallel_eval: Whether trajectory evaluation uses multiprocessing.
        :return: None.
        """
        use_lirec_flag = bool(use_LIReC)
        super().__init__(
            searchables,
            use_lirec_flag,
            name="GeneticSearch",
            description="Genetic algorithm optimization module for trajectory discovery",
            version="1.0.0",
        )
        self.generations = generations
        self.pop_size = pop_size
        self.max_coord_init = max_coord_init
        self.elite_fraction = elite_fraction
        self.mutation_prob = mutation_prob
        self.mutation_step = mutation_step
        self.crossover_prob = crossover_prob
        self.max_retries = max_retries
        self.refine_prob = refine_prob
        self.refine_coord_prob = refine_coord_prob
        self.parallel_eval = parallel_eval

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> None:
        """
        Run GA search for each searchable and export results as pickled DataManager chunks.
        :return: None.
        """
        if not self.searchables:
            return

        export_root = sys_config.EXPORT_SEARCH_RESULTS or ""
        os.makedirs(
            dir_path := os.path.join(export_root, self.searchables[0].const.name),
            exist_ok=True,
        )

        with Exporter.export_stream(dir_path, exists_ok=True, clean_exists=True, fmt=Formats.PICKLE) as write_chunk:
            for space in SmartTQDM(
                self.searchables,
                desc="Optimizing trajectories via GA: ",
                **sys_config.TQDM_CONFIG,
            ):
                searcher = GeneticSearchMethod(
                    space,
                    space.const,
                    generations=self.generations,
                    pop_size=self.pop_size,
                    max_coord_init=self.max_coord_init,
                    elite_fraction=self.elite_fraction,
                    mutation_prob=self.mutation_prob,
                    mutation_step=self.mutation_step,
                    crossover_prob=self.crossover_prob,
                    max_retries=self.max_retries,
                    refine_prob=self.refine_prob,
                    refine_coord_prob=self.refine_coord_prob,
                    parallel_eval=self.parallel_eval,
                    use_LIReC=self.use_LIReC,
                )
                res = searcher.search()
                space: Searchable
                write_chunk(res, space.cmf_name)
