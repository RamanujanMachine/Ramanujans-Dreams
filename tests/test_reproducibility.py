"""Reproducibility guarantees for the RNG seeding utility and the samplers.

Covers the contract introduced by ``dreamer.utils.rand``:

* ``derive_seed`` / ``derive_rng`` are deterministic functions of
  ``(search_config.GLOBAL_SEED, *context)`` — same inputs ⇒ same stream;
  different context ⇒ (independent) different stream; ``GLOBAL_SEED = None`` ⇒
  nondeterministic.
* Context hashing is process-stable (``hashlib``-based, not the salted built-in
  ``hash``), so a seed derived in a spawned worker matches the parent.  We assert
  the value is stable across a fresh subprocess.
* Each MCMC sampler (PT / Discrete / Raw-space), the sphere sampler, and the
  raycast pipeline produce **identical** output when constructed with the same
  seed, and (generically) different output for a different seed.
"""
import subprocess
import sys

import numpy as np
import pytest

from dreamer.configs.search import search_config
from dreamer.utils import rand
from dreamer.utils.rand import derive_py_random, derive_rng, derive_seed

from dreamer.extraction.samplers.parallel_tempering_raycaster import ParallelTemperingSampler
from dreamer.extraction.samplers.discrete_raycaster import DiscreteMCMCSampler
from dreamer.extraction.samplers.raw_space_raycaster import RawSpaceMCMCSampler
from dreamer.extraction.samplers.sphere_sampler import PrimitiveSphereSampler

# Negative orthant in 3D — a fat, full-dimensional cone (see test_recession_cone_samplers).
A_PRIME = np.eye(3, dtype=np.int64)
MCMC_SAMPLERS = [ParallelTemperingSampler, DiscreteMCMCSampler, RawSpaceMCMCSampler]


@pytest.fixture(autouse=True)
def _fixed_master_seed():
    """Pin GLOBAL_SEED to a known value and restore it afterwards."""
    original = search_config.GLOBAL_SEED
    search_config.GLOBAL_SEED = 42
    yield
    search_config.GLOBAL_SEED = original


# ---------------------------------------------------------------------------
# 1. The seeding utility
# ---------------------------------------------------------------------------

class TestDeriveSeed:
    def test_same_context_same_seed(self):
        assert derive_seed("shard0", "pt") == derive_seed("shard0", "pt")
        rng_a = derive_rng("shard0", "ga", "log(2)")
        rng_b = derive_rng("shard0", "ga", "log(2)")
        assert np.array_equal(rng_a.random(10), rng_b.random(10))

    def test_different_context_different_seed(self):
        assert derive_seed("shard0", "pt") != derive_seed("shard1", "pt")
        assert derive_seed("shard0", "pt") != derive_seed("shard0", "ga")
        assert derive_seed("s", "ga", "e") != derive_seed("s", "ga", "pi")

    def test_master_seed_changes_stream(self):
        s_a = derive_seed("shard0", "pt")
        search_config.GLOBAL_SEED = 1234
        s_b = derive_seed("shard0", "pt")
        assert s_a != s_b

    def test_none_master_seed_is_nondeterministic(self):
        search_config.GLOBAL_SEED = None
        assert derive_seed("shard0", "pt") != derive_seed("shard0", "pt")
        # A derived Generator still works, just unseeded.
        assert derive_rng("shard0", "pt").random() is not None

    def test_py_random_is_reproducible(self):
        a = derive_py_random("shard0", "annealing", "e")
        b = derive_py_random("shard0", "annealing", "e")
        assert [a.random() for _ in range(10)] == [b.random() for _ in range(10)]

    def test_seed_is_process_stable(self):
        """Context hashing must not depend on PYTHONHASHSEED (spawned-worker parity)."""
        local = derive_seed("shard0", "pt")
        code = (
            "from dreamer.configs.search import search_config;"
            "search_config.GLOBAL_SEED = 42;"
            "from dreamer.utils.rand import derive_seed;"
            "print(derive_seed('shard0', 'pt'))"
        )
        # PYTHONHASHSEED=random in the child would break a built-in-hash implementation.
        out = subprocess.check_output([sys.executable, "-c", code], env={"PYTHONHASHSEED": "random", **_clean_env()})
        assert int(out.strip().splitlines()[-1]) == local


def _clean_env():
    import os
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    return env


# ---------------------------------------------------------------------------
# 1b. Constant-derived seeds must be content-stable, not identity-stable
# ---------------------------------------------------------------------------

class TestConstantSeedStability:
    """A ``Constant`` fed into the RNG context must seed by *name*, not identity.

    Regression guard for the bug where the search methods seeded from
    ``str(constant)``: ``Constant`` had no ``__str__``/``__repr__``, so the default
    ``<...Constant object at 0x...>`` (memory address) made every process derive a
    *different* seed — silently breaking reproducibility of Simulated Annealing,
    Genetic, Gradient-Ascent and Hybrid-SPSA searches across runs.

    Two independent guards: (a) ``str``/``repr`` are the content (name), address-
    free, so even the old ``str(constant)`` pattern is now stable; (b) two distinct
    ``Constant`` instances with the same name derive the identical RNG stream —
    which is what "same seed ⇒ same run" relies on.
    """

    def _fresh_constant(self):
        import sympy as sp
        from dreamer.utils.constants.constant import Constant
        # A new object each call (same name) — distinct identities, same content.
        return Constant("repro_probe", sp.pi)

    def test_str_and_repr_are_address_free_name(self):
        c = self._fresh_constant()
        assert str(c) == c.name
        assert repr(c) == c.name
        assert "0x" not in str(c) and "object at" not in str(c)

    def test_two_instances_have_equal_string_form(self):
        c1 = self._fresh_constant()
        c2 = self._fresh_constant()
        assert c1 is not c2
        assert str(c1) == str(c2)

    @pytest.mark.parametrize("method", ["annealing", "genetic", "gradient", "spsa_adam"])
    def test_seed_stream_stable_across_constant_instances(self, method):
        # Mirror exactly what the search methods derive: (shard_id, method, name).
        c1 = self._fresh_constant()
        c2 = self._fresh_constant()
        rng_a = derive_rng("shardX", method, c1.name)
        rng_b = derive_rng("shardX", method, c2.name)
        assert np.array_equal(rng_a.random(10), rng_b.random(10))
        # And the (previously broken) str-based context is now stable too.
        pa = derive_py_random("shardX", method, str(c1))
        pb = derive_py_random("shardX", method, str(c2))
        assert [pa.random() for _ in range(10)] == [pb.random() for _ in range(10)]


# ---------------------------------------------------------------------------
# 1c. The trajectory reservoir order must be process-stable
# ---------------------------------------------------------------------------

#: Subprocess program: sample a shard's trajectory reservoir under a fixed
#: GLOBAL_SEED and print the ordered result.  Run in a child with
#: ``PYTHONHASHSEED=random`` so any reliance on ``Position``'s (per-process-salted)
#: hash order surfaces as a different order.
_RESERVOIR_PROG = r'''
from dreamer.configs.search import search_config
search_config.GLOBAL_SEED = 42
search_config.SAMPLING_METHOD = "{method}"
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq
from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.extraction.sampling_orchestrators.shard_sampler_orchestrator import (
    ShardSamplingOrchestrator,
)

cmf = rt_pFq(1, 1, sp.Integer(1))
symbols = list(cmf.matrices.keys())
zero_shift = Position({{s: sp.Integer(0) for s in symbols}})
hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
interior = Position({{symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)}})
shard = Shard(cmf, e, hps, [1, 1], zero_shift, interior)

samples = ShardSamplingOrchestrator(shard).sample_trajectories(40)
order = [tuple(int(p[s]) for s in symbols) for p in samples]
print("RESERVOIR_ISLIST::" + str(isinstance(samples, list)))
print("RESERVOIR_ORDER::" + repr(order))
'''


def _reservoir_output(method: str) -> tuple[str, str]:
    """Run ``_RESERVOIR_PROG`` in a fresh ``PYTHONHASHSEED=random`` subprocess.

    :return: ``(is_list, order_repr)`` extracted from the child's stdout (robust
        to any interleaved log lines).
    """
    prog = _RESERVOIR_PROG.format(method=method)
    out = subprocess.check_output(
        [sys.executable, "-c", prog],
        env={"PYTHONHASHSEED": "random", **_clean_env()},
    ).decode()
    is_list = order = None
    for line in out.splitlines():
        if line.startswith("RESERVOIR_ISLIST::"):
            is_list = line.split("::", 1)[1]
        elif line.startswith("RESERVOIR_ORDER::"):
            order = line.split("::", 1)[1]
    assert is_list is not None and order is not None, f"child produced no reservoir:\n{out}"
    return is_list, order


@pytest.mark.parametrize("method", ["pt", "discrete"])
def test_reservoir_order_is_process_stable(method):
    """``sample_trajectories`` must return the same *ordered* reservoir every run.

    Regression guard: it used to return a ``Set[Position]``, whose iteration order
    depends on ``Position``'s per-process-salted hash (PYTHONHASHSEED).  Downstream
    seed-vector selection (the search reservoirs) sorts these and picks the first
    match, so a different order silently seeds the search differently — the root
    cause of two identical runs diverging.  We run two independent
    ``PYTHONHASHSEED=random`` subprocesses and require byte-identical order.
    """
    is_list_a, order_a = _reservoir_output(method)
    is_list_b, order_b = _reservoir_output(method)
    assert is_list_a == "True", "sample_trajectories must return a list, not a set"
    assert order_a == order_b, (
        f"reservoir order differs across processes for method={method} — "
        f"sampling is not process-stable:\n{order_a}\n{order_b}"
    )


# ---------------------------------------------------------------------------
# 2. Sampler reproducibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sampler_cls", MCMC_SAMPLERS, ids=lambda c: c.__name__)
def test_mcmc_sampler_same_seed_identical(sampler_cls):
    seed = derive_seed("orthant", "pt")
    rays_a = np.asarray(sampler_cls(A_PRIME, rng_seed=seed).harvest(40), dtype=np.int64)
    rays_b = np.asarray(sampler_cls(A_PRIME, rng_seed=seed).harvest(40), dtype=np.int64)
    assert rays_a.shape == rays_b.shape
    assert np.array_equal(rays_a, rays_b), "same-seed harvest was not reproducible"


@pytest.mark.parametrize("sampler_cls", MCMC_SAMPLERS, ids=lambda c: c.__name__)
def test_mcmc_sampler_different_seed_differs(sampler_cls):
    a = np.asarray(sampler_cls(A_PRIME, rng_seed=1).harvest(40), dtype=np.int64)
    b = np.asarray(sampler_cls(A_PRIME, rng_seed=2).harvest(40), dtype=np.int64)
    # Different seeds should (with overwhelming probability) give a different harvest.
    assert not (a.shape == b.shape and np.array_equal(a, b))


def test_sphere_sampler_reproducible():
    seed = derive_seed("noA", "sphere")
    a = np.asarray(PrimitiveSphereSampler(3, seed=seed).harvest(50))
    b = np.asarray(PrimitiveSphereSampler(3, seed=seed).harvest(50))
    assert np.array_equal(np.sort(a, axis=0), np.sort(b, axis=0))
