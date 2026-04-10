import numpy as np
import sympy as sp
import pytest

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.extraction.samplers.raycast_sampler import RaycastPipelineSampler
from dreamer.extraction.samplers.sphere_sampler import PrimitiveSphereSampler
from dreamer.extraction.sampling_orchestrators.shard_sampler_orchestrator import ShardSamplingOrchestrator


pytestmark = pytest.mark.timeout(60)


@pytest.fixture
def cmf_2d():
    return rt_pFq(1, 1, sp.Integer(1))


@pytest.fixture
def symbols(cmf_2d):
    return list(cmf_2d.matrices.keys())


def _position(symbols, values):
    return Position({sym: sp.sympify(v) for sym, v in zip(symbols, values)})


def test_shard_does_not_expose_sample_trajectories_method(cmf_2d, symbols):
    shard = Shard(cmf_2d, e, [], [], _position(symbols, [0, 0]))
    assert not hasattr(shard, "sample_trajectories")


def test_shard_sampler_translates_pipeline_output(monkeypatch, cmf_2d, symbols):
    requested = [None]

    def _fake_harvest(self, compute_n_samples, exact=False):
        assert exact is False
        requested[0] = compute_n_samples(self.d_orig)
        return np.array([[1.9, -2.2], [3.0, 4.0]])

    monkeypatch.setattr(RaycastPipelineSampler, "harvest", _fake_harvest)

    s0 = symbols[0]
    hps = [Hyperplane(s0, symbols)]
    shard = Shard(cmf_2d, e, hps, [1], _position(symbols, [0, 0]), _position(symbols, [1, 0]))
    sampler = ShardSamplingOrchestrator(shard)

    sampled = sampler.sample_trajectories(lambda d: d * 4)

    assert requested[0] == 8
    assert sampled == {
        Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(-2)}),
        Position({symbols[0]: sp.Integer(3), symbols[1]: sp.Integer(4)}),
    }


def test_whole_space_shard_uses_zero_constraint_matrix(monkeypatch, cmf_2d, symbols):
    observed_dim = [None]

    def _fake_harvest(self, _compute_n_samples, exact=False):
        assert exact is False
        observed_dim[0] = self.d
        return np.array([[1, 0]])

    monkeypatch.setattr(PrimitiveSphereSampler, "harvest", _fake_harvest)

    shard = Shard(cmf_2d, e, [], [], _position(symbols, [0, 0]))
    trajectories = ShardSamplingOrchestrator(shard).sample_trajectories(lambda _d: 1)

    assert observed_dim[0] == 2
    assert len(trajectories) == 1


def test_shard_sampler_forwards_exact_flag(monkeypatch, cmf_2d, symbols):
    observed_exact = [None]

    def _fake_harvest(self, _compute_n_samples, exact=False):
        observed_exact[0] = exact
        return np.array([[1, 0]])

    monkeypatch.setattr(RaycastPipelineSampler, "harvest", _fake_harvest)

    s0 = symbols[0]
    hps = [Hyperplane(s0, symbols)]
    shard = Shard(cmf_2d, e, hps, [1], _position(symbols, [0, 0]), _position(symbols, [1, 0]))

    _ = ShardSamplingOrchestrator(shard).sample_trajectories(lambda d: d, exact=True)
    assert observed_exact[0] is True
