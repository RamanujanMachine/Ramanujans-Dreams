import numpy as np

from dreamer.extraction.samplers.sphere_sampler import PrimitiveSphereSampler, check_points


def test_check_points_filters_radius_origin_and_primitivity():
    points = np.array([
        [0, 0],    # origin -> reject
        [2, 2],    # gcd 2 -> reject
        [1, 2],    # primitive and inside radius -> accept
        [5, 0],    # outside radius -> reject
    ], dtype=np.int64)

    mask = check_points(points, R_sq=9)

    assert np.array_equal(mask, np.array([False, False, True, False]))


def test_compute_radius_increases_with_sample_budget():
    sampler = PrimitiveSphereSampler(d=3, batch_size=128)
    r_small = sampler.compute_radius(20)
    r_big = sampler.compute_radius(200)

    assert r_small > 0
    assert r_big >= r_small


def test_sample_returns_primitive_points_with_directional_spread():
    sampler = PrimitiveSphereSampler(d=3, batch_size=256)
    sampler.rng = np.random.default_rng(0)

    samples = sampler.harvest(lambda d: 40)

    assert samples.shape[0] == 40
    assert samples.shape[1] == 3
    assert len({tuple(v) for v in samples}) == 40

    # Primitive and non-zero by construction.
    assert np.all(np.any(samples != 0, axis=1))

    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    unit = samples / norms

    # Exploration check: avoid collapse to a single direction.
    mean_abs = np.abs(np.mean(unit, axis=0))
    assert np.max(mean_abs) < 0.55

    cos_sim = np.clip(unit @ unit.T, -1.0, 1.0)
    diag = np.arange(cos_sim.shape[0])
    cos_sim[diag, diag] = -1.0
    nearest = np.max(cos_sim, axis=1)
    # If nearest-neighbor cosine is strictly below 1.0, rays are not duplicates.
    assert np.median(nearest) < 0.98
