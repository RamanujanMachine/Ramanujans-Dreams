"""Recession-cone invariants for the MCMC trajectory samplers.

After the strict->closed change, every MCMC sampler (Linear PT / Discrete / Raw-space)
must harvest directions that are
  * **non-zero** — the origin is never a trajectory, and
  * inside the **closed** recession cone ``A_prime v <= 0`` (non-strict: a ray with
    ``A_i v == 0`` runs parallel to facet ``i`` and is admissible),
  * **primitive** (gcd of components == 1).

These are seed-independent invariants, so the test is robust to the samplers' RNG.
"""
import numpy as np
import pytest

from dreamer.extraction.samplers.parallel_tempering_raycaster import ParallelTemperingSampler
from dreamer.extraction.samplers.discrete_raycaster import DiscreteMCMCSampler
from dreamer.extraction.samplers.raw_space_raycaster import RawSpaceMCMCSampler

# Cone = { v : A_prime v <= 0 }.  Rows e0,e1,e2 -> the negative orthant (full-dimensional,
# strict interior at e.g. (-1,-1,-1)); its faces are the coordinate planes, so short
# axis-aligned rays such as (-1, 0, 0) are *face-parallel* recession directions.
A_PRIME = np.eye(3, dtype=np.int64)

SAMPLERS = [ParallelTemperingSampler, DiscreteMCMCSampler, RawSpaceMCMCSampler]


def _gcd(v):
    g = 0
    for x in v:
        x = abs(int(x))
        while x:
            g, x = x, g % x
    return g


@pytest.mark.parametrize("sampler_cls", SAMPLERS, ids=lambda c: c.__name__)
def test_harvest_is_nonzero_primitive_and_in_closed_cone(sampler_cls):
    sampler = sampler_cls(A_PRIME, rng_seed=12345)
    rays = np.asarray(sampler.harvest(40), dtype=np.int64)

    assert rays.shape[0] > 0, "sampler harvested nothing on a fat full-dim cone"
    # non-zero: no harvested direction is the origin
    assert np.all(np.any(rays != 0, axis=1)), "harvested the zero vector"
    # closed cone: A_prime v <= 0 (tolerance for float matmul of the conditioned basis)
    assert np.all(A_PRIME @ rays.T <= 1e-6), "harvested a direction outside the closed cone"
    # primitive
    assert all(_gcd(r) == 1 for r in rays), "harvested a non-primitive direction"
