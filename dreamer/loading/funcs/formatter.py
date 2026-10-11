from abc import ABC, abstractmethod
from dreamer.loading.config import DATA_ANNOTATE, TYPE_ANNOTATE
from dreamer.utils.caching import cached_property
from dreamer.utils.constants.constant import Constant
from dreamer.utils.types import CMFData
from dreamer.configs import config
import json
from typing import Dict, Any, Type, Optional, List, Tuple, Union
import sympy as sp


class Formatter(ABC):
    """
    This class defines the bridge between a CMF in Ramanujan Tools and a JSON representation of it.
    This class is a registry of all formatters
    """
    registry: Dict[str, Type['Formatter']] = dict()

    def __init__(self, const: Union[str, Constant, List[Union[str, Constant]]], shifts: list,
                 selected_start_points: Optional[List[Tuple[Union[int, sp.Rational], ...]]] = None,
                 only_selected: bool = False,
                 use_inv_t: bool = None,
                 cmf_name_segments: Optional[List[List[Union[str, sp.Expr, int]]]] = None,
                 selected_trajectories: Optional[List[Optional[Tuple[Union[int, sp.Rational], ...]]]] = None):
        if use_inv_t is None:
            use_inv_t = config.search.DEFAULT_USES_INV_T

        # Trajectories pair 1:1 with selected_start_points.  A None entry (or an entirely
        # omitted list) means "use the start point as-is" (must be a strict interior point);
        # a provided trajectory lets a border start point resolve to the correct shard by
        # taking one step along it.  See ShardExtractor.extract / Shard.encoding_at.
        if selected_trajectories is not None:
            if selected_start_points is None:
                raise ValueError('selected_trajectories requires selected_start_points')
            if len(selected_trajectories) != len(selected_start_points):
                raise ValueError(
                    f'selected_trajectories length ({len(selected_trajectories)}) must match '
                    f'selected_start_points length ({len(selected_start_points)})'
                )

        # Normalise to a list of name strings; accept single constant or list.
        if isinstance(const, list):
            self.consts: List[str] = [
                c.name if isinstance(c, Constant) else c for c in const
            ]
        else:
            self.consts = [const.name if isinstance(const, Constant) else const]

        self.shifts = self._normalize_shifts(shifts)
        self.selected_start_points = selected_start_points
        self.selected_trajectories = selected_trajectories
        self.only_selected = only_selected
        self.use_inv_t = use_inv_t

        cmf_name_segments = cmf_name_segments if cmf_name_segments is not None else [[self.__class__.__name__]]
        cmf_name_segments: List[List[Union[str, sp.Expr, int]]]

        # The constant is intentionally *not* appended to the name: a CMF
        # may later be searched for additional constants, and we want the
        # cmf_id (= cmf_name) to remain stable across those re-runs.  The
        # parent constant directory (EXPORT_CMFS/<const>/...) still
        # disambiguates files by constant.
        cmf_name_segments += [shifts]

        name_segments_concat = []

        for segment in cmf_name_segments:
            segment_components = []

            for component in segment:
                match component:
                    case str():
                        segment_components.append(component)
                    case sp.Expr():
                        canonized_expr = sp.sympify(component)
                        expr_str = str(canonized_expr).replace(" ", "")
                        expr_str = expr_str.replace("**", "p")
                        expr_str = expr_str.replace("*", "x")
                        expr_str = expr_str.replace("/", ".")
                        segment_components.append(expr_str)
                    case int():
                        segment_components.append(str(component))
            segment_str = '_'.join(segment_components)
            name_segments_concat.append(segment_str)
        self.cmf_name = '__'.join(name_segments_concat)

    @staticmethod
    def _normalize_shifts(shifts):
        """Validate and coerce shifts to exact rational sympy numbers.

        A coordinate shift must be a rational (an integer or an ``sp.Rational``):
        a Python ``float`` (e.g. ``0.5``) is silently inexact and previously leaked
        floats into the start-point coordinates, while an irrational/symbolic shift
        is meaningless for a lattice walk.  Both now raise a clear, user-facing
        error instead of corrupting the search.

        Integers (Python ``int`` or ``sp.Integer``) and ``sp.Rational`` pass through
        as their sympy form — name-stable (the serialized ``cmf_name`` is unchanged)
        and consistent for downstream coordinate arithmetic.

        :param shifts: The raw ``shifts`` argument (list, ``Position``, or ``None``).
        :return: The normalised shifts (a list of sympy rationals when a list was
            given; otherwise the input is returned unchanged).
        :raises ValueError: If any list entry is not a rational number.
        """
        if not isinstance(shifts, list):
            return shifts
        normalized = []
        for s in shifts:
            val = sp.sympify(s)
            if val.is_Float or not val.is_rational:
                raise ValueError(
                    f"Invalid shift {s!r}: shifts must be rational numbers "
                    f"(an integer or sp.Rational). Pass e.g. sp.Rational(1, 2) "
                    f"instead of a float like 0.5 or an irrational value."
                )
            normalized.append(val)
        return normalized

    @property
    def const(self) -> str:
        """Backward-compatible accessor — returns the first (primary) constant name."""
        return self.consts[0]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Formatter.registry[cls.__name__] = cls

    @abstractmethod
    def __repr__(self):
        return json.dumps(self._to_json_obj())

    @abstractmethod
    def __str__(self):
        return f'<{self.__class__.__name__}: {self.__repr__()}>'

    @abstractmethod
    def __hash__(self):
        return hash((
            tuple(self.consts),
            tuple(self.shifts if self.shifts else []),
            frozenset(self.selected_start_points if self.selected_start_points else []),
            tuple(self.selected_trajectories) if self.selected_trajectories else (),
            self.only_selected,
            self.use_inv_t
        ))

    @abstractmethod
    def to_cmf(self) -> CMFData:
        """
        Converts the Formatter to a CMF.
        :return: The CMF with shift as ShiftCMF object
        """
        raise NotImplementedError

    def _to_json_obj(self) -> dict:
        # Prepare shifts
        shifts = self.shifts
        if shifts:
            shifts = [str(shift) if isinstance(shift, sp.Expr) else shift for shift in self.shifts]

        # Prepare start points
        points = self.selected_start_points
        if points:
            points = [[v if isinstance(v, int) else str(v) for v in p] for p in self.selected_start_points]

        # Prepare trajectories (paired 1:1 with start points; entries may be None)
        trajectories = self.selected_trajectories
        if trajectories:
            trajectories = [
                None if t is None else [v if isinstance(v, int) else str(v) for v in t]
                for t in self.selected_trajectories
            ]

        return {
            'consts': self.consts,
            'use_inv_t': self.use_inv_t,
            'shifts': shifts,
            'selected_start_points': points,
            'selected_trajectories': trajectories,
            'only_selected': self.only_selected
        }

    @classmethod
    @abstractmethod
    def _from_json_obj(cls, obj: dict | list) -> object:
        raise NotImplementedError

    @staticmethod
    def _shift_from_json(data):
        return [sp.sympify(shift) if isinstance(shift, str) else shift for shift in data]

    @staticmethod
    def _selected_start_points_from_json(data):
        points = []
        if not data:
            return points

        for point_list in data:
            points.append(tuple(sp.sympify(v) if isinstance(v, str) else v for v in point_list))
        return points

    @staticmethod
    def _selected_trajectories_from_json(data):
        """Deserialize selected_trajectories (entries may be ``None``); ``None``/empty → ``None``."""
        if not data:
            return None

        trajectories = []
        for traj in data:
            if traj is None:
                trajectories.append(None)
            else:
                trajectories.append(tuple(sp.sympify(v) if isinstance(v, str) else v for v in traj))
        return trajectories

    @classmethod
    def fetch_from_registry(cls, name: str) -> Type['Formatter']:
        """
        Checks if a formatter is registered and returns it
        :param name: The name of the formatter class
        :return: The formatter class
        """
        if name in cls.registry:
            return Formatter.registry[name]
        raise KeyError(f'{name} is not registered as a Formatter')

    def to_json_obj(self) -> Dict[str, Any]:
        """
        Converts the Formatter to a JSON object
        :return: The JSON like object (dictionary)
        """
        if not issubclass(self.__class__, Formatter):
            raise TypeError(f'Not a Formatter subclass: {type(self)}')
        return {TYPE_ANNOTATE: self.__class__.__name__, DATA_ANNOTATE: self._to_json_obj()}

    @classmethod
    def from_json_obj(cls, src: dict) -> 'Formatter':
        """
        Converts from a JSON object to the relevant Formatter
        :param src: The source JSON like object (dictionary)
        :return: The Formatter object
        """
        try:
            if src[TYPE_ANNOTATE] not in cls.registry:
                raise NotImplementedError(f'Not a Formatter subclass: {src[TYPE_ANNOTATE]}')
            return cls.registry[src[TYPE_ANNOTATE]]._from_json_obj(src[DATA_ANNOTATE])
        except AttributeError:
            raise NotImplementedError(f'constructor for {src[TYPE_ANNOTATE]} is not implemented.'
                                      f' Make sure that the name of the file is the same as the class')
        except TypeError:
            raise NotImplementedError(f'All formatters must inherit from {cls.__name__} and be in registry')
