from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.storage import Exporter, Formats
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.configs.system import sys_config
from dreamer.search.methods.sa import SimulatedAnnealingSearchMethod
from typing import List, Optional
from ramanujantools.cmf import CMF
import os


class SimulatedAnnealingSearchMod(SearcherModScheme):
    def __init__(
            self,
            searchables: List[Searchable],
            use_LIReC: Optional[bool] = True,
            iterations: int = 100,
            max_res: int = 10,
            t0: float = 1.0,
            tmin: float = 1e-3
    ):
        super().__init__(
            searchables, use_LIReC,
            name='SimulatedAnnealingSearch',
            description='Simulated Annealing optimization module for trajectory discovery',
            version='1.0.0'
        )
        self.iterations = iterations
        self.max_res = max_res
        self.t0 = t0
        self.tmin = tmin

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self):
        if not self.searchables:
            return

        # Prepare export directory using the constant's name
        os.makedirs(
            dir_path := os.path.join(sys_config.EXPORT_SEARCH_RESULTS, self.searchables[0].const.name),
            exist_ok=True
        )

        with Exporter.export_stream(dir_path, exists_ok=True, clean_exists=True, fmt=Formats.PICKLE) as write_chunk:
            for space in SmartTQDM(
                    self.searchables, desc='Optimizing trajectories via SA: ', **sys_config.TQDM_CONFIG
            ):
                searcher = SimulatedAnnealingSearchMethod(
                    space,
                    space.const,
                    iterations=self.iterations,
                    max_res=self.max_res,
                    t0=self.t0,
                    tmin=self.tmin,
                    use_LIReC=self.use_LIReC
                )

                res = searcher.search()

                # Export chunk per searchable
                space: Searchable
                if space.cmf.__class__ == CMF:
                    filename = f'generated_cmf_hashed_{hash(space.cmf)}'
                else:
                    filename = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in repr(space.cmf)).strip('_')
                write_chunk(res, filename)
