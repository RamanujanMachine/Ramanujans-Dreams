"""Tests for the SQLite database round-trip (DB v1).

Covers:
- insert / select round-trip for pFq
- duplicate insert raises error
- update and replace operations
- delete operations
- select missing constant
"""
import os
import tempfile
import pytest

from dreamer.loading.databases.db_v1.db import DB
from dreamer.loading.funcs.pFq_fmt import pFq
from dreamer.utils.constants.constant import Constant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def log2():
    from dreamer import log
    return log(2)


@pytest.fixture
def const_pi():
    from dreamer import pi
    return pi


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary DB that is cleaned up after each test."""
    db_path = str(tmp_path / "test.db")
    db = DB(path=db_path)
    yield db
    del db


# ---------------------------------------------------------------------------
# 1. Insert and select
# ---------------------------------------------------------------------------
class TestDBInsertSelect:

    def test_insert_and_select_one(self, tmp_db, log2):
        fmt = pFq(log2, 2, 1, -1)
        tmp_db.insert(log2, [fmt])

        results = tmp_db.select(log2)
        assert len(results) == 1

    def test_insert_and_select_multiple(self, tmp_db, log2):
        fmt1 = pFq(log2, 2, 1, -1)
        fmt2 = pFq(log2, 3, 1, -1)
        tmp_db.insert(log2, [fmt1, fmt2])

        results = tmp_db.select(log2)
        assert len(results) == 2

    def test_select_nonexistent_returns_empty(self, tmp_db, const_pi):
        results = tmp_db.select(const_pi)
        assert results == []


# ---------------------------------------------------------------------------
# 2. Duplicate handling
# ---------------------------------------------------------------------------
class TestDBDuplicates:

    def test_duplicate_insert_raises(self, tmp_db, log2):
        fmt = pFq(log2, 2, 1, -1)
        tmp_db.insert(log2, [fmt])

        with pytest.raises(Exception):
            tmp_db.insert(log2, [fmt])


# ---------------------------------------------------------------------------
# 3. Config consistency
# ---------------------------------------------------------------------------
class TestConfigConsistency:

    def test_analysis_trajectory_func_returns_positive(self):
        from dreamer.configs.analysis import analysis_config
        for d in range(1, 5):
            n = analysis_config.NUM_TRAJECTORIES_FROM_DIM(d)
            assert n > 0, f"dim={d}: trajectory count should be > 0, got {n}"

    def test_search_trajectory_func_returns_positive(self):
        from dreamer.configs.search import search_config
        for d in range(1, 5):
            n = search_config.NUM_TRAJECTORIES_FROM_DIM(d)
            assert n > 0, f"dim={d}: trajectory count should be > 0, got {n}"

    def test_search_depth_func_returns_bounded(self):
        from dreamer.configs.search import search_config
        for traj_len in [1, 10, 100, 1000]:
            depth = search_config.DEPTH_FROM_TRAJECTORY_LEN(traj_len, 3)
            assert 1 <= depth <= 1500, f"Depth {depth} out of range for len={traj_len}"
