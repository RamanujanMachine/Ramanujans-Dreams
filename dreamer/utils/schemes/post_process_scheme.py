from abc import abstractmethod
from typing import Dict, List, Optional

from dreamer.utils.schemes.module import Module
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.constants.constant import Constant


class PostProcessModScheme(Module):
    """Scheme for the post-process stage.

    Post-process modules run after the Search stage finishes.  They read
    the JSONL files written by the searcher, reconstruct
    ``TrajectoryAttributesHandler`` for trajectories that still need
    expensive (Tier-3) attributes, compute those attributes (typically in
    parallel via :func:`worker_pool`), and append patch records so the
    merge-on-read reader picks them up transparently.
    """

    def __init__(
        self,
        priorities: Dict[Constant, List[Searchable]],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
    ):
        """
        :param priorities: Mapping from constant → searchables produced by
            the search stage.  Each searchable carries its ``cmf`` and
            ``cmf_name``, so the post-process can look up CMF symbolic
            matrices without re-loading from disk.
        """
        super().__init__(name, description, version)
        self.priorities = priorities

    @abstractmethod
    def execute(self) -> None:
        """Run the post-process stage end-to-end.  No return value."""
        raise NotImplementedError
