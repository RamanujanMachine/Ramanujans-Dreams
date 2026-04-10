import numpy as np
import pytest

from dreamer.extraction.samplers.raycast_sampler import RaycastPipelineSampler
from dreamer.extraction.samplers.raycaster import _guide_rays_mcmc, _guide_rays_mhs, RayCastingSamplingMethod


pytestmark = pytest.mark.timeout(60)


def _assert_directional_exploration(rays: np.ndarray) -> None:
    norms = np.linalg.norm(rays, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    # A healthy exploration should avoid collapsing into one narrow direction.
    mean_abs = np.abs(np.mean(rays, axis=0))
    assert np.max(mean_abs) < 0.35

    variances = np.var(rays, axis=0)
    assert float(np.min(variances)) > 0.03


def test_estimate_cone_fraction_empty_bounds_is_one():
    b = np.empty((0, 3), dtype=np.float64)
    assert RaycastPipelineSampler._estimate_cone_fraction(b, d_flat=3, samples=1000) == pytest.approx(1.0)


def test_calculate_r_max_matches_closed_form_formula():
    target_quota = 120
    fraction = 0.25
    d_flat = 2

    result = RaycastPipelineSampler._calculate_R_max(target_quota, fraction, d_flat)
    expected = np.sqrt((target_quota * 1.0) / (fraction * np.pi))

    assert result == pytest.approx(expected, rel=1e-12)


def test_mcmc_guide_rays_are_spread_and_unit_norm():
    np.random.seed(0)
    rays = _guide_rays_mcmc(
        d_flat=3,
        B=np.empty((0, 3), dtype=np.float64),
        start_pos=np.array([1.0, 0.0, 0.0]),
        target_rays=160,
        mix_steps=20,
    )

    _assert_directional_exploration(rays)


def test_mhs_guide_rays_are_spread_and_unit_norm():
    np.random.seed(0)
    rays = _guide_rays_mhs(
        d_flat=3,
        B=np.empty((0, 3), dtype=np.float64),
        start_pos=np.array([1.0, 0.0, 0.0]),
        target_rays=160,
        mix_steps=20,
    )

    _assert_directional_exploration(rays)


def test_stage2_harvest_deduplicates_hits(monkeypatch):
    raycaster = RayCastingSamplingMethod(
        Z_reduced=np.eye(2, dtype=np.int64),
        B_reduced=np.empty((0, 2), dtype=np.float64),
        d_orig=2,
    )

    monkeypatch.setattr(
        raycaster,
        "_generate_continuous_guide_rays",
        lambda _target: np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
    )

    def _fake_raycast(d_orig, d_flat, z_mat, b_mat, guide_rays, r_max, t_step=0.1, max_per_ray=5):
        assert d_orig == 2
        assert d_flat == 2
        assert guide_rays.shape[0] == 2
        raw = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.int64)
        counts = np.array([2, 2], dtype=np.int32)
        return raw, counts

    monkeypatch.setattr("dreamer.extraction.samplers.raycaster._raycast", _fake_raycast)

    result = raycaster.harvest(target_rays=2, R_max=2.0, max_per_ray=2)

    assert result.shape == (1, 2)
    assert np.array_equal(result[0], np.array([1, 0]))


def test_pipeline_harvest_returns_target_quota_when_first_sweep_exceeds(monkeypatch):
    a_prime = np.eye(3, dtype=np.float64)
    sampler = RaycastPipelineSampler(a_prime)

    monkeypatch.setattr(
        "dreamer.extraction.samplers.raycast_sampler.HyperSpaceConditioner.process",
        lambda self: (np.eye(3, dtype=np.int64), np.empty((0, 3), dtype=np.float64), np.eye(3, dtype=np.int64)),
    )
    monkeypatch.setattr("dreamer.extraction.samplers.raycast_sampler.RaycastPipelineSampler._verify_uniformity", lambda *args, **kwargs: None)

    captured = {"max_per_ray": 0}

    def _fake_harvest(self, target_rays, R_max, max_per_ray=1):
        captured["max_per_ray"] = max_per_ray
        rays = np.arange(0, target_rays * 3 * self.d_orig, dtype=np.int64).reshape(target_rays * 3, self.d_orig)
        return rays

    monkeypatch.setattr("dreamer.extraction.samplers.raycast_sampler.RayCastingSamplingMethod.harvest", _fake_harvest)

    rays = sampler.harvest(lambda d: 9)

    assert sampler.d_flat == 3
    assert rays.shape == (9, 3)
    assert captured["max_per_ray"] >= 1


def test_pipeline_harvest_expands_radius_when_yield_too_low(monkeypatch):
    a_prime = np.ones((2, 5), dtype=np.float64)
    sampler = RaycastPipelineSampler(a_prime)

    monkeypatch.setattr(
        "dreamer.extraction.samplers.raycast_sampler.HyperSpaceConditioner.process",
        lambda self: (np.eye(5, 4, dtype=np.int64), np.ones((1, 4), dtype=np.float64), np.eye(4, dtype=np.int64)),
    )
    monkeypatch.setattr("dreamer.extraction.samplers.raycast_sampler.RaycastPipelineSampler._estimate_cone_fraction", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("dreamer.extraction.samplers.raycast_sampler.RaycastPipelineSampler._verify_uniformity", lambda *args, **kwargs: None)

    rmax_calls = []

    def _fake_harvest(self, target_rays, R_max, max_per_ray=1):
        rmax_calls.append(R_max)
        if len(rmax_calls) == 1:
            return np.arange(0, 6 * self.d_orig, dtype=np.int64).reshape(6, self.d_orig)
        return np.arange(0, (target_rays + 8) * self.d_orig, dtype=np.int64).reshape(target_rays + 8, self.d_orig)

    monkeypatch.setattr("dreamer.extraction.samplers.raycast_sampler.RayCastingSamplingMethod.harvest", _fake_harvest)

    rays = sampler.harvest(lambda d: 20, exact=True)

    assert sampler.d_flat == 4
    assert len(rmax_calls) >= 2
    assert rmax_calls[1] > rmax_calls[0]
    assert rays.shape[0] == 20


def test_pipeline_harvest_exact_mode_uses_callable_quota(monkeypatch):
    a_prime = np.eye(3, dtype=np.float64)
    sampler = RaycastPipelineSampler(a_prime)

    monkeypatch.setattr(
        "dreamer.extraction.samplers.raycast_sampler.HyperSpaceConditioner.process",
        lambda self: (np.eye(3, dtype=np.int64), np.empty((0, 3), dtype=np.float64), np.eye(3, dtype=np.int64)),
    )
    monkeypatch.setattr("dreamer.extraction.samplers.raycast_sampler.RaycastPipelineSampler._verify_uniformity", lambda *args, **kwargs: None)


    def _fake_harvest(self, target_rays, R_max, max_per_ray=1):
        rays = np.arange(0, (target_rays + 6) * self.d_orig, dtype=np.int64).reshape(target_rays + 6, self.d_orig)
        return rays

    monkeypatch.setattr("dreamer.extraction.samplers.raycast_sampler.RayCastingSamplingMethod.harvest", _fake_harvest)

    rays = sampler.harvest(lambda d: 11, exact=True)
    assert rays.shape == (11, 3)
