"""Selected trajectories that lie entirely inside a CMF hyperplane.

``extraction.ON_HYPERPLANE_TRAJECTORY``:

* ``'skip'`` (default) — such a (start, trajectory) pair is rejected, as before, with a
  hint naming the option;
* ``'touching_shards'`` — every shard on whose boundary the trajectory lies, and which
  contains lattice points, is built with the user's start and trajectory as its seed.

A start point *on* a hyperplane whose trajectory leaves it is unambiguous and keeps
being resolved by one step along the trajectory.
"""
import pytest
import sympy as sp

from dreamer.configs import config
from dreamer.extraction import extractor as extractor_mod
from dreamer.extraction.extractor import ShardExtractor
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.loading import pFq
from dreamer.utils.constants.ready_made_consts import zeta

X0, X1, Y0 = sp.symbols("x0 x1 y0")
SYMS = [X0, X1, Y0]


class _FakeLogger:
    Levels = extractor_mod.Logger.Levels
    records = []

    def __init__(self, msg, level=None):
        type(self).records.append((msg, level))

    def log(self):
        return None


@pytest.fixture
def logs(monkeypatch):
    _FakeLogger.records = []
    monkeypatch.setattr(extractor_mod, "Logger", _FakeLogger)
    return _FakeLogger.records


@pytest.fixture
def mode(monkeypatch):
    def set_mode(value):
        monkeypatch.setattr(config.extraction, "ON_HYPERPLANE_TRAJECTORY", value)
    return set_mode


def _extract(monkeypatch, hyperplanes, start, traj):
    """Extract the selected shard(s) of a 3-variable CMF with a fixed hyperplane list."""
    cmf_data = pFq(zeta(3), 2, 1, -1, selected_start_points=[start], only_selected=True,
                   selected_trajectories=[traj]).to_cmf()
    hps = [Hyperplane(e, SYMS) for e in hyperplanes]
    monkeypatch.setattr(ShardExtractor, "_extract_cmf_hps", lambda self: hps)
    return ShardExtractor([zeta(3)], cmf_data).extract()


# Start (0, 2, 3) lies on x0 = 0 and on x1 + y0 - 5 = 0.  Trajectory (0, 1, 1) stays inside
# x0 = 0 (ambiguous) but leaves x1 + y0 = 5 (resolved by one step: the shard is above it).
PLANES = [X0, X1 + Y0 - 5]
START, TRAJ = (0, 2, 3), (0, 1, 1)


class TestDefaultSkip:
    def test_rejected_with_hint(self, monkeypatch, mode):
        mode("skip")
        with pytest.raises(ValueError, match="ON_HYPERPLANE_TRAJECTORY"):
            _extract(monkeypatch, PLANES, START, TRAJ)

    def test_start_on_plane_but_leaving_it_is_unambiguous(self, monkeypatch, mode):
        mode("skip")
        shards = _extract(monkeypatch, PLANES, START, (1, 1, 1))     # leaves x0 = 0
        assert len(shards) == 1
        assert shards[0].encoding == (1, 1)

    def test_one_step_landing_on_a_plane_still_errors(self, monkeypatch, mode):
        mode("touching_shards")
        # (0,2,3) + (1,0,0) = (1,2,3): on x0 - 1 = 0, crossed (not contained) -> ambiguous step.
        with pytest.raises(ValueError, match="lies on hyperplane"):
            _extract(monkeypatch, [X0 - 1], (0, 2, 3), (1, 0, 0))


class TestTouchingShards:
    def test_both_sides_searched_and_logged(self, monkeypatch, mode, logs):
        mode("touching_shards")
        shards = _extract(monkeypatch, PLANES, START, TRAJ)
        assert sorted(s.encoding for s in shards) == [(-1, 1), (1, 1)]
        for s in shards:
            assert tuple(s.get_interior_point()[x] for x in SYMS) == START
            assert tuple(s.selected_trajectory[x] for x in SYMS) == TRAJ
        msg = next(m for m, _ in logs if "touching shard" in m)
        assert "2 touching shard(s)" in msg and "belongs to no single shard" in msg
        assert "x0" in msg

    def test_side_without_lattice_points_dropped(self, monkeypatch, mode, logs):
        """0 < x0 < 1/2 (between x0 = 0 and 2·x0 - 1 = 0) has no lattice point."""
        mode("touching_shards")
        shards = _extract(monkeypatch, [X0, 2 * X0 - 1], START, TRAJ)
        assert [s.encoding for s in shards] == [(-1, -1)]
        msg = next(m for m, _ in logs if "touching shard" in m)
        assert "dropped side(s) without lattice points" in msg

    def test_trajectories_on_the_hyperplane_are_legal(self, monkeypatch, mode):
        """The seed and any other direction parallel to the plane are valid recession directions."""
        mode("touching_shards")
        for s in _extract(monkeypatch, PLANES, START, TRAJ):
            assert s.is_valid_trajectory(s.selected_trajectory)
            from ramanujantools import Position
            assert s.is_valid_trajectory(Position({X0: 0, X1: 3, Y0: 2}))  # x0 stays 0


@pytest.mark.timeout(600)
class TestMeijerGCaseB:
    """Sheevy's construction lies inside Σa − Σb = 4 of MG55 (z = 1)."""

    def _shards(self, mode_value, monkeypatch):
        from dreamer.loading import MeijerG
        from dreamer.utils.constants.ready_made_consts import joint
        monkeypatch.setattr(config.extraction, "ON_HYPERPLANE_TRAJECTORY", mode_value)
        target = joint(zeta(3), zeta(4))
        cmf_data = MeijerG(target, 5, 2, 5, 5, 1,
                           selected_start_points=[(0, 0, 1, 1, 2, 0, 0, 0, 0, 0)],
                           selected_trajectories=[(-16, -17, 11, 12, 10, -2, -1, 0, 1, 2)],
                           only_selected=True, use_inv_t=True).to_cmf()
        return ShardExtractor([target], cmf_data).extract()

    def test_skip_rejects(self, monkeypatch):
        with pytest.raises(ValueError, match="a0 \\+ a1 \\+ a2 \\+ a3 \\+ a4 - b0 - b1 - b2 - b3 - b4 - 4"):
            self._shards("skip", monkeypatch)

    def test_touching_shards_keeps_the_lattice_side(self, monkeypatch, logs):
        shards = self._shards("touching_shards", monkeypatch)
        # 4 < Σa − Σb < 5 holds no lattice point: only the Σa − Σb < 4 side remains.
        assert len(shards) == 1
        shard = shards[0]
        assert shard.is_valid_trajectory(shard.selected_trajectory)
        msg = next(m for m, _ in logs if "touching shard" in m)
        assert "dropped side(s) without lattice points" in msg
