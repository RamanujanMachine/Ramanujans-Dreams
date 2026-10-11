import json
from ramanujantools.cmf import pFq as rt_pFq
from dreamer.utils.constants.constant import Constant
from dreamer.loading.funcs.formatter import Formatter
from dreamer.utils.types import CMFData
from typing import Optional, List, Tuple, Union
import sympy as sp
from ramanujantools import Position


class pFq(Formatter):
    def __init__(self,
                 const: Union[str, Constant, List[Union[str, Constant]]],
                 p: int, q: int, z: sp.Expr | int, shifts: Optional[list] = None,
                 selected_start_points: Optional[List[Tuple[Union[int, sp.Rational], ...]]] = None,
                 only_selected: bool = False,
                 use_inv_t: bool = None,
                 selected_trajectories: Optional[List[Optional[Tuple[Union[int, sp.Rational], ...]]]] = None
                 ):
        """
        Represents a pFq and its CMF + allows conversion to and from JSON.
        :var const: The constant related to this pFq function
        :var p: The p value of the pFq.
        :var q: The q value of the pFq.
        :var z: The z value of the pFq.
        :var shifts: The shifts in starting point in the CMF where a sp.Rational indicates a shift.
        While 0 indicates no shift (None if not doesn't matter).
        :var selected_start_points: Optional list of start points to extract shards from.
        :var only_selected: If True, only extract shards from the selected start points.
        :var selected_trajectories: Optional list of trajectories paired 1:1 with
            ``selected_start_points``.  A non-None trajectory lets a border start point
            resolve to the correct shard via one step; a None entry uses the start point
            as-is (must be a strict interior point).
        """
        self.p = p
        self.q = q
        self.z = z
        self.shifts = shifts
        if self.shifts is None:
            self.shifts = [0] * (self.p + self.q)

        super().__init__(const, self.shifts, selected_start_points, only_selected, use_inv_t,
                         [[self.__class__.__name__, self.p, self.q, self.z]],
                         selected_trajectories=selected_trajectories)

        if self.p <= 0 or self.q <= 0:
            raise ValueError("p and q should be positive integers")
        if not isinstance(self.shifts, list) and not isinstance(self.shifts, Position):
            raise ValueError("Shifts should be a list or Position")
        if self.p + self.q != len(self.shifts):
            raise ValueError("Shifts should be of length p + q")

    @classmethod
    def _from_json_obj(cls, data: dict | list) -> "pFq":
        """
        Converts a JSON string to a pFq.
        :param data: The JSON string representation of the pFq (only attributes).
        :return: A pFq object.
        """
        data = dict(data)
        # Normalise to a list under the 'const' key (pFq.__init__ param name).
        # Old format: {'const': "log2", ...}  → const=["log2"]
        # New format: {'consts': ["log2", "pi"], ...} → const=["log2", "pi"]
        if 'consts' in data:
            data['const'] = data.pop('consts')
        # 'const' stays as-is (may be str or list).
        data['z'] = sp.sympify(data['z']) if isinstance(data['z'], str) else data['z']
        data['shifts'] = cls._shift_from_json(data['shifts'])
        data['selected_start_points'] = cls._selected_start_points_from_json(data['selected_start_points'])
        data['selected_trajectories'] = cls._selected_trajectories_from_json(data.get('selected_trajectories'))
        return cls(**data)

    def _to_json_obj(self) -> dict:
        """
        Converts the pFq to a JSON string (i.e., convert sp.Expr to str)
        :return: A dictionary representation of the pFq matching the JSON format.
        """
        return {
            **super()._to_json_obj(),
            "p": self.p,
            "q": self.q,
            "z": str(self.z) if isinstance(self.z, sp.Expr) else self.z
        }

    def to_cmf(self) -> CMFData:
        """
        Converts the pFq to a CMF.
        :return: A tuple (CMF, shifts)
        """
        cmf = rt_pFq(self.p, self.q, self.z)
        shifts = Position({k: v for k, v in zip(cmf.matrices.keys(), self.shifts)})
        return CMFData(cmf, shifts, self.selected_start_points, self.only_selected, self.use_inv_t, self.cmf_name,
                       selected_trajectories=self.selected_trajectories)

    def __repr__(self):
        return json.dumps(self._to_json_obj())

    def __str__(self):
        return f'<{self.__class__.__name__}: {self.__repr__()}>'

    def __hash__(self):
        return hash((self.p, self.q, str(self.z), super().__hash__()))
