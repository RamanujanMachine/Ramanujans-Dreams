import numpy as np
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.utils import initial_points
import sympy as sp


x, y, z = sp.symbols('x y z')

hps = [
    Hyperplane(x - z + 1, symbols=[x, y, z]),
    Hyperplane(y - z, symbols=[x, y, z]),
    Hyperplane(y, symbols=[x, y, z]),
    Hyperplane(z, symbols=[x, y, z])
]


class _DummyIterator:
    """Length-aware iterator for deterministic progress-wrapped iteration in tests."""

    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class _DummyPool:
    """Sequential pool stand-in implementing the imap_unordered API used by compute_mapping."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap_unordered(self, func, tasks, chunksize=1):
        del chunksize
        return _DummyIterator([func(task) for task in tasks])


class TestShardMaps:
    def test_compute_mapping(self, monkeypatch):
        """Assumption: mapping cardinality is stable; failure mode: pool API migration breaks shard aggregation."""
        monkeypatch.setattr(initial_points, "create_pool", lambda: _DummyPool())

        D = 3
        S = 8
        A = np.array([hp.vectors[0] for hp in hps], dtype=np.int64)
        b = np.array([hp.vectors[1] for hp in hps], dtype=np.int64)
        mappings = initial_points.compute_mapping(D, S, A, b)
        assert len(mappings) == 12
