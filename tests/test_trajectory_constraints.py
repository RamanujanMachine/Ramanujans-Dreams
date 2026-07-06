"""Tests for extraction-stage trajectory direction constraints.

Covers the pure helpers (rows / augmentation / strict mask / LP feasibility), the
orchestrator post-filter wiring, and the search-stage ``FlatlandGeometry`` confinement.
See ``context/algorithms/06_trajectory_constraints.md``.
"""

import numpy as np
import sympy as sp
import pytest

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.configs import config
from dreamer.extraction.shard import Shard
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.sampling_orchestrators.shard_sampler_orchestrator import (
    ShardSamplingOrchestrator,
)
from dreamer.extraction.samplers.raycast_sampler import RaycastPipelineSampler
from dreamer.extraction.samplers import constraints as C


pytestmark = pytest.mark.timeout(60)

SYMS5 = list(sp.symbols("x0 x1 x2 y0 y1"))
SYMS2 = list(sp.symbols("x0 x1"))


# ----------------------------------------------------------------------------
# constraint_rows
# ----------------------------------------------------------------------------

def _has_row(rows, target):
    target = np.asarray(target, dtype=np.float64)
    return any(np.allclose(r, target) for r in rows)


def test_constraint_rows_ratio_reduced_and_signs():
    rows, fixed = C.constraint_rows(SYMS5, {"x0": 12, "y1": 28})
    # Ratio 12:28 reduces to 3:7 → equality -7*v_x0 + 3*v_y1 = 0 (and its negation).
    assert _has_row(rows, [-7, 0, 0, 0, 3])
    assert _has_row(rows, [7, 0, 0, 0, -3])
    # Sign walls -sign(c)*e_i for both fixed coords.
    assert _has_row(rows, [-1, 0, 0, 0, 0])
    assert _has_row(rows, [0, 0, 0, 0, -1])
    assert fixed == {0: 1, 4: 1}


def test_constraint_rows_negative_value_flips_sign_wall():
    _, fixed = C.constraint_rows(SYMS5, {"x0": 12, "y1": -28})
    assert fixed == {0: 1, 4: -1}


def test_constraint_rows_single_coord_is_sign_only():
    rows, fixed = C.constraint_rows(SYMS5, {"x0": 5})
    # No ratio (needs >=2 coords); just one sign wall.
    assert rows.shape == (1, 5)
    assert _has_row(rows, [-1, 0, 0, 0, 0])
    assert fixed == {0: 1}


def test_constraint_rows_zero_value_is_equality():
    rows, fixed = C.constraint_rows(SYMS5, {"x2": 0})
    assert _has_row(rows, [0, 0, 1, 0, 0])
    assert _has_row(rows, [0, 0, -1, 0, 0])
    assert fixed == {2: 0}


def test_resolve_unknown_variable_raises():
    with pytest.raises(ValueError, match="Unknown trajectory-constraint variable"):
        C.resolve_constraint_indices(SYMS5, {"nope": 1})


# ----------------------------------------------------------------------------
# augment_cone + fixed_sign_mask
# ----------------------------------------------------------------------------

def test_augment_cone_noop_without_constraints():
    A = np.eye(2)
    A_aug, fixed = C.augment_cone(A, SYMS2, None)
    assert A_aug is A and fixed == {}


def test_augment_cone_appends_rows():
    A = np.array([[-1.0, 0.0]])
    A_aug, fixed = C.augment_cone(A, SYMS2, {"x0": 1, "x1": 1})
    assert A_aug.shape[0] > 1
    assert np.allclose(A_aug[0], [-1.0, 0.0])  # original row preserved on top
    assert fixed == {0: 1, 1: 1}


def test_augment_cone_whole_space_becomes_constraint_rows():
    A_aug, fixed = C.augment_cone(None, SYMS2, {"x0": 1, "x1": 1})
    assert A_aug is not None and A_aug.shape[1] == 2
    assert fixed == {0: 1, 1: 1}


def test_fixed_sign_mask_drops_zero_and_wrong_sign():
    samples = np.array([[1, 1], [-1, -1], [1, 0], [2, 2]])
    mask = C.fixed_sign_mask(samples, {0: 1, 1: 1})
    assert list(mask) == [True, False, False, True]


def test_fixed_sign_mask_empty_is_all_true():
    samples = np.array([[1, -3], [0, 0]])
    assert C.fixed_sign_mask(samples, {}).all()


# ----------------------------------------------------------------------------
# constrained_cone_feasible
# ----------------------------------------------------------------------------

def test_feasible_when_cone_admits_ratio():
    # First-quadrant recession cone: -v0<=0, -v1<=0  → v0,v1 >= 0.
    A = np.array([[-1.0, 0.0], [0.0, -1.0]])
    assert C.constrained_cone_feasible(A, SYMS2, {"x0": 1, "x1": 1})


def test_infeasible_when_sign_conflicts_with_cone():
    # Cone forces v0 <= 0, but x0:1 wants v0 > 0.
    A = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert not C.constrained_cone_feasible(A, SYMS2, {"x0": 1, "x1": 1})


def test_infeasible_when_ratio_incompatible_sign():
    # First quadrant: v1 must be >= 0, but x1:-1 wants v1 < 0.
    A = np.array([[-1.0, 0.0], [0.0, -1.0]])
    assert not C.constrained_cone_feasible(A, SYMS2, {"x0": 1, "x1": -1})


# ----------------------------------------------------------------------------
# Orchestrator post-filter wiring
# ----------------------------------------------------------------------------

@pytest.fixture
def cmf_2d():
    return rt_pFq(1, 1, sp.Integer(1))


def test_orchestrator_post_filters_sign_violations(monkeypatch, cmf_2d):
    symbols = list(cmf_2d.matrices.keys())

    def _fake_harvest(self, _compute_n_samples, exact=False):
        return np.array([[1, 1], [-1, -1], [1, 0], [2, 2]])

    monkeypatch.setattr(RaycastPipelineSampler, "harvest", _fake_harvest)
    monkeypatch.setattr(
        config.extraction, "TRAJECTORY_CONSTRAINTS",
        {str(symbols[0]): 1, str(symbols[1]): 1},
    )

    s0 = symbols[0]
    shard = Shard(cmf_2d, e, [Hyperplane(s0, symbols)], [1],
                  Position({symbols[0]: sp.Integer(0), symbols[1]: sp.Integer(0)}),
                  Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(0)}))
    orch = ShardSamplingOrchestrator(shard, sampling_method="raycast")
    sampled = orch.sample_trajectories(lambda d: d)

    # Only the strictly-positive same-sign directions survive.  sample_trajectories
    # returns a deterministically-ordered list; compare as sets (content, not order).
    assert set(sampled) == {
        Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)}),
        Position({symbols[0]: sp.Integer(2), symbols[1]: sp.Integer(2)}),
    }


def test_orchestrator_noop_without_constraints(monkeypatch, cmf_2d):
    symbols = list(cmf_2d.matrices.keys())
    monkeypatch.setattr(config.extraction, "TRAJECTORY_CONSTRAINTS", None)

    def _fake_harvest(self, _compute_n_samples, exact=False):
        return np.array([[1, -1], [-2, 2]])

    monkeypatch.setattr(RaycastPipelineSampler, "harvest", _fake_harvest)
    s0 = symbols[0]
    shard = Shard(cmf_2d, e, [Hyperplane(s0, symbols)], [1],
                  Position({symbols[0]: sp.Integer(0), symbols[1]: sp.Integer(0)}),
                  Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(0)}))
    orch = ShardSamplingOrchestrator(shard, sampling_method="raycast")
    assert orch._fixed == {}
    assert len(orch.sample_trajectories(lambda d: d)) == 2


# ----------------------------------------------------------------------------
# FlatlandGeometry confinement (search stage) — needs lattice backend (rama env)
# ----------------------------------------------------------------------------

def test_flatland_geometry_confined_to_ratio(monkeypatch, cmf_2d):
    from dreamer.search.methods.flatland.geometry import FlatlandGeometry

    symbols = list(cmf_2d.matrices.keys())
    monkeypatch.setattr(
        config.extraction, "TRAJECTORY_CONSTRAINTS",
        {str(symbols[0]): 1, str(symbols[1]): 1},
    )
    # Half-space recession cone v0 >= 0 (-v0 <= 0); the ratio v0 = v1 collapses it to 1D.
    A = np.array([[-1.0, 0.0]])
    b = np.array([0.0])
    shard = Shard.from_matrices(
        cmf_2d, e, A, b,
        Position({symbols[0]: sp.Integer(0), symbols[1]: sp.Integer(0)}),
    )
    geom = FlatlandGeometry(shard)

    assert geom.d_flat == 1  # one ratio equality removed one dimension
    # The 1D flatland basis spans the (1,1) ray; +z is in, -z (wrong sign) is out.
    z_in = np.array([1])
    if not geom.is_inside(z_in):
        z_in = np.array([-1])  # basis sign is arbitrary; pick the in-cone orientation
    assert geom.is_inside(z_in)
    assert not geom.is_inside(-z_in)
    v = geom.Z_reduced @ z_in
    assert v[0] == v[1] and v[0] != 0  # exactly on the ratio, non-zero
