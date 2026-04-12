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

    def __hash__(self):
        return hash((self.cmf, self.shift))
