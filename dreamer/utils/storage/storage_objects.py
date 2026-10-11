from dataclasses import dataclass, field
from ramanujantools import Matrix, Position
import pandas as pd
from collections import UserDict
from typing import Any, Dict, List, Optional, Tuple

from dreamer.utils.schemes.jsonable import JSONable


def _serialize_jsonable(value: Any) -> Any:
    """Serialize nested JSONable payloads for JSON export."""
    if isinstance(value, JSONable):
        return value.to_json()
    if isinstance(value, dict):
        return {str(k): _serialize_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_serialize_jsonable(v) for v in value]
    return value


def _deserialize_jsonable(value: Any) -> Any:
    """Deserialize nested JSON payloads into known runtime objects when possible."""
    if isinstance(value, dict):
        class_name = value.get("__class__")
        if class_name == "Shard":
            from dreamer.extraction.shard import Shard
            return Shard.from_json_obj(value)
        return {k: _deserialize_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialize_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class SearchVector:
    """
    A class representing a search vector in a specific space
    """
    start: Position
    trajectory: Position

    def __hash__(self):
        return hash((self.start, self.trajectory))

    def __eq__(self, other):
        return self.start == other.start and self.trajectory == other.trajectory


@dataclass
class SearchData:
    """
    A class representing search data alongside a specific search vector
    """
    sv: SearchVector
    limit: float = None
    delta: float | str = None
    eigen_values: Dict = field(default_factory=dict)
    gcd_slope: float | None = None
    initial_values: Matrix = None
    LIReC_identify: bool = False
    errors: Dict[str, Exception | None] = field(default_factory=dict)

    # def to_hdf5(self, filename):
    #     ...
    # def


class DataManager(UserDict[SearchVector, SearchData], JSONable):
    """
    DataManager represents a set of results found in a specific search in a CMF
    """

    def __init__(self, use_LIReC: bool, searchable_space: Optional[Any] = None):
        """
        :param use_LIReC: If true, LIReC will be used to identify constants within the searchable spaces.
        :param searchable_space: The searchable object whose trajectories are evaluated by this manager.
        """
        super().__init__()
        self.use_LIReC = use_LIReC
        self.searchable_space = searchable_space

    @property
    def identified_percentage(self) -> float:
        """
        Computes the percentage identified by the search vector, if no data collected mark as 1 (i.e. 100%)
        :return: The percentage identified by the search vector as a number in [0, 1]
        """
        df = self.as_df()
        if df is None:
            return 1
        if self.use_LIReC:
            frac = df['LIReC_identify'].sum() / len(df['LIReC_identify'])
        else:
            frac = 1 - df['initial_values'].isna().sum() / len(df['initial_values'])
        return frac

    @property
    def best_delta(self) -> Tuple[Optional[float], Optional[SearchVector]]:
        """
        The best delta found
        :return: A tuple of the delta value and the search vector it was found in.
        """
        df = self.as_df()
        if df.empty:
            return None, None

        deltas = df['delta'].dropna()
        if deltas.empty:
            return None, None

        row = df.loc[deltas.idxmax()]
        return row['delta'], row['sv']

    def get_data(self) -> List[SearchData]:
        """
        Gather all search data in the manager into a list
        :return: The data collected as a list
        """
        return list(self.values())

    def to_json(self) -> Dict:
        """
        Convert the DataManager to a JSON-serializable dictionary.
        """
        data_list = []
        for sv, sd in self.items():
            errs = {str(k): str(v) for k, v in sd.errors.items()} if sd.errors else {}
            item = {
                "sv": {
                    "start": {str(k): int(v) for k, v in sv.start.items()},
                    "trajectory": {str(k): int(v) for k, v in sv.trajectory.items()}
                },
                "limit": float(sd.limit) if sd.limit is not None else None,
                "delta": (float(sd.delta) if isinstance(sd.delta, (int, float)) else str(sd.delta)) if sd.delta is not None else None,
                "eigen_values": {str(k): float(v) if isinstance(v, (int, float)) else str(v) for k, v in sd.eigen_values.items()} if sd.eigen_values else {},
                "gcd_slope": float(sd.gcd_slope) if sd.gcd_slope is not None else None,
                "initial_values": [[str(cell) for cell in row] for row in sd.initial_values.tolist()] if sd.initial_values is not None else None,
                "LIReC_identify": bool(sd.LIReC_identify),
                "errors": errs
            }
            data_list.append(item)

        return {
            "__class__": "DataManager",
            "use_LIReC": bool(self.use_LIReC),
            "searchable_space": _serialize_jsonable(self.searchable_space),
            "data": data_list
        }

    def to_json_obj(self) -> Dict:
        """Backward-compatible alias used by existing exporter paths."""
        return self.to_json()

    @classmethod
    def from_json_obj(cls, obj: Dict) -> "DataManager":
        use_LIReC = obj.get("use_LIReC", False)
        searchable_space = _deserialize_jsonable(obj.get("searchable_space"))
        manager = cls(use_LIReC=use_LIReC, searchable_space=searchable_space)
        for item in obj.get("data", []):
            start = Position(item["sv"]["start"])
            trajectory = Position(item["sv"]["trajectory"])
            sv = SearchVector(start, trajectory)

            init_vals = None
            if item.get("initial_values") is not None:
                import sympy as sp
                from ramanujantools import Matrix
                init_vals = Matrix([[sp.sympify(c) for c in row] for row in item["initial_values"]])

            sd = SearchData(
                sv=sv,
                limit=item.get("limit"),
                delta=item.get("delta"),
                eigen_values=item.get("eigen_values", {}),
                gcd_slope=item.get("gcd_slope"),
                initial_values=init_vals,
                LIReC_identify=item.get("LIReC_identify", False),
                errors={k: Exception(v) for k, v in item.get("errors", {}).items()}
            )
            manager[sv] = sd
        return manager

    def as_df(self) -> pd.DataFrame:
        """
        Convert the data into a dataframe
        :return: The pandas dataframe.
        """
        rows = [
            {
                "sv": sv,
                "delta": data.delta,
                "limit": data.limit,
                "eigen_values": data.eigen_values,
                "gcd_slope": data.gcd_slope,
                "initial_values": data.initial_values,
                "LIReC_identify": data.LIReC_identify,
                "errors": data.errors,
            }
            for sv, data in self.items()
        ]
        return pd.DataFrame(rows)
