import sympy as sp
from typing import List, Tuple, Optional
from ramanujantools.cmf import CMF
from ramanujantools import Position
from dataclasses import dataclass


@dataclass(frozen=True)
class CMFData:
    cmf: CMF
    shift: Position
    selected_points: Optional[List[Tuple[int | sp.Rational, ...]]] = None
    only_selected: bool = False
    use_inv_t: bool = True
    cmf_name: str = 'UnknownCMF'
    # Optional trajectories paired 1:1 with ``selected_points``.  When a point's
    # trajectory is not None, the shard encoding is derived from one step along it
    # (``point + trajectory``) so a point lying on a shard border still resolves to
    # the correct shard.  ``None`` for a point ⇒ behave as if no trajectory was given.
    selected_trajectories: Optional[List[Tuple[int | sp.Rational, ...]]] = None

    def __hash__(self):
        return hash((self.cmf, self.shift))
