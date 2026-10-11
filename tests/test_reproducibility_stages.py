"""Per-stage and per-method reproducibility guarantees.

Complements ``tests/test_reproducibility.py`` (the RNG utility, the samplers, and
the reservoir-order guard) with determinism checks for whole pipeline stages:

* **Extraction** — the heuristic ray-shooter must discover the same shards, in the
  same order, with the same interior witnesses, for a fixed seed.  The wall-clock
  timeout is only a guardrail (it may truncate the tail, which is acceptable); the
  *RNG-level* order is what must be reproducible, so these tests pin a fixed ray
  budget (``max_seconds=None``) to exercise the deterministic path.
* **Each search method** — for a fixed seed + shard, a method must explore the
  identical sequence of trajectories.  This is the guarantee the ``Set[Position]``
  reservoir-order bug silently broke: the reservoir seeded each run differently.

The search methods are driven with a deterministic **stub** evaluator because CI
has no LIReC database (real identification is unavailable).  The stub marks every
direction "identified" with a δ that is a fixed function of the direction, which
isolates exactly what these tests target — the method's own control flow
(reservoir seeding, RNG draws, neighbour ordering).  The real walk/identification
is deterministic and covered elsewhere (``test_attributes_management``, the
sampler tests in ``test_reproducibility``).

Cross-process stability (the PYTHONHASHSEED dimension that the reservoir bug
exploited) is guarded for sampling by
``test_reproducibility.test_reservoir_order_is_process_stable`` and for extraction
by ``test_extraction_discovery_is_process_stable`` below; the per-method tests run
in-process (their process-level input, the reservoir, is already guarded).
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import sympy as sp

from dreamer.configs.search import search_config
from dreamer.extraction.hyperplanes import Hyperplane


@pytest.fixture(autouse=True)
def _fixed_master_seed():
    """Pin GLOBAL_SEED to a known value and restore it afterwards."""
    original = search_config.GLOBAL_SEED
    search_config.GLOBAL_SEED = 42
    yield
    search_config.GLOBAL_SEED = original


def _clean_env():
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    return env


# ---------------------------------------------------------------------------
# 1. Extraction stage — deterministic discovery order + interior witnesses
# ---------------------------------------------------------------------------

def _extract(seed: int, num_rays: int):
    from dreamer.extraction.v2.ray_extractor import RayShootingExtractor

    syms = list(sp.symbols("x:3"))
    hps = [Hyperplane(s, syms) for s in syms]
    # max_seconds=None ⇒ no wall-clock truncation: exercise the deterministic
    # (seeded, ray-budgeted) path the reproducibility guarantee applies to.
    return RayShootingExtractor(num_rays=num_rays, max_seconds=None, seed=seed).extract(hps)


def test_extraction_discovery_is_deterministic():
    """Same seed ⇒ same cells, same discovery order, same interior witnesses."""
    m1 = _extract(seed=0, num_rays=4000)
    m2 = _extract(seed=0, num_rays=4000)
    assert len(m1) > 1, "extraction found too few cells to be a meaningful check"
    # Dict preserves insertion (discovery) order — this asserts order, not just content.
    assert list(m1.keys()) == list(m2.keys())
    for k in m1:
        assert np.array_equal(m1[k], m2[k]), f"interior witness for cell {k} differs"


_EXTRACT_PROG = r'''
from dreamer.configs.search import search_config
search_config.GLOBAL_SEED = 42
import sympy as sp
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.v2.ray_extractor import RayShootingExtractor
syms = list(sp.symbols("x:3"))
hps = [Hyperplane(s, syms) for s in syms]
m = RayShootingExtractor(num_rays=4000, max_seconds=None, seed=0).extract(hps)
print("EXTRACT::" + repr([(k, tuple(int(x) for x in v)) for k, v in m.items()]))
'''


def _extract_subprocess() -> str:
    out = subprocess.check_output(
        [sys.executable, "-c", _EXTRACT_PROG],
        env={"PYTHONHASHSEED": "random", **_clean_env()},
    ).decode()
    for line in out.splitlines():
        if line.startswith("EXTRACT::"):
            return line.split("::", 1)[1]
    raise AssertionError(f"child produced no extraction output:\n{out}")


def test_extraction_discovery_is_process_stable():
    """Discovery order + witnesses are identical across two PYTHONHASHSEED=random
    processes (cell keys are content-stable int tuples, so a difference would mean
    the seeded ray sequence itself is not process-stable)."""
    assert _extract_subprocess() == _extract_subprocess()


# ---------------------------------------------------------------------------
# 2. Search methods — deterministic exploration under a stub evaluator
# ---------------------------------------------------------------------------

_METHODS = [
    ("dreamer.search.methods.annealing.annealing_scan", "SimulatedAnnealingSearch"),
    ("dreamer.search.methods.genetic_search.genetic_scan", "GeneticSearch"),
    ("dreamer.search.methods.gradient_ascent.grad_ascent_scan", "GradientAscentSearch"),
    ("dreamer.search.methods.gradient_ascent.spsa_adam_ascent", "HybridSPSASearch"),
    ("dreamer.search.methods.small_angle.small_angle_scan", "SmallAngleSearch"),
]

#: Subprocess program: run one search method under a deterministic stub evaluator
#: and print the ordered sequence of explored directions.  Run in a child with
#: ``PYTHONHASHSEED=random`` so the whole method path (including the reservoir it
#: seeds from) is exercised under a fresh hash salt — the only way to catch the
#: ``Set[Position]`` reservoir-order class of bug, which is stable *within* a
#: single process and would slip past a same-process repeat.
#:
#: The stub marks every direction identified with a δ drawn from a *rugged*
#: deterministic landscape — a ``hashlib`` (process-stable, unlike the salted
#: built-in ``hash``) pseudo-random function of the direction.  Ruggedness is
#: essential: a smooth landscape lets every method descend greedily and never
#: exercise its RNG (Metropolis accept, mutation, diffraction), which would make
#: this test vacuous — it must actually depend on the seeded RNG and on neighbour
#: ordering so that any nondeterminism there surfaces as a divergent path.  The
#: stub is patched in both the method module and ``parallel_eval`` (the
#: ``pool=None`` serial batch path uses the latter's binding).  Iteration/
#: reservoir budgets are capped so the child stays fast while still taking steps.
_METHOD_PROG = r'''
from dreamer.configs import config
from dreamer.configs.search import search_config
search_config.GLOBAL_SEED = 42
config.configure(search=dict(
    ANNEAL_MAX_ITERS=12, ANNEAL_MAX_TOTAL_STEPS=48, ANNEAL_RESERVOIR_SIZE=20,
    GA_GENERATIONS=3, GA_POPULATION_SIZE=8,
    GRAD_MAX_STEPS=12, GRAD_RESERVOIR_SIZE=20,
    SPSA_MAX_STEPS=12, SPSA_RESERVOIR_SIZE=20,
    SA_MAX_DEPTH=12, SA_PATIENCE=4, SA_RESERVOIR_SIZE=20,
))
import hashlib
import importlib
import numpy as np
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq
from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard

cmf = rt_pFq(1, 1, sp.Integer(1))
symbols = list(cmf.matrices.keys())
zero_shift = Position({{s: sp.Integer(0) for s in symbols}})
hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
interior = Position({{symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)}})
shard = Shard(cmf, e, hps, [1, 1], zero_shift, interior)

captured = []
def stub(z, **eval_ctx):
    key = tuple(int(v) for v in np.asarray(z))
    captured.append(key)
    # Rugged, process-stable landscape (hashlib, not the salted built-in hash),
    # so the method must exercise its seeded RNG / ordering rather than descend
    # greedily.  delta in [0, 1); always identified.
    h = hashlib.sha256(repr(key).encode()).digest()
    return int.from_bytes(h[:6], "little") / float(1 << 48), True

mod = importlib.import_module("{module}")
mod.evaluate_in_flatland = stub
import dreamer.search.methods.flatland.parallel_eval as pe
pe.evaluate_in_flatland = stub

getattr(mod, "{cls}")(shard, e).search()
print("SEQ::" + repr(captured))
'''


def _method_sequence(module_name: str, cls_name: str) -> str:
    prog = _METHOD_PROG.format(module=module_name, cls=cls_name)
    out = subprocess.check_output(
        [sys.executable, "-c", prog],
        env={"PYTHONHASHSEED": "random", **_clean_env()},
    ).decode()
    for line in out.splitlines():
        if line.startswith("SEQ::"):
            return line.split("::", 1)[1]
    raise AssertionError(f"child produced no exploration sequence:\n{out}")


@pytest.mark.parametrize(
    "module_name,cls_name", _METHODS, ids=[m[1] for m in _METHODS]
)
def test_search_method_exploration_is_process_stable(module_name, cls_name):
    """A method explores the identical trajectory sequence across two processes.

    End-to-end process-stability guard for each method's own control flow: the
    seeded RNG (Metropolis accept / mutation / diffraction) and neighbour ordering
    must be reproducible, so two independent ``PYTHONHASHSEED=random`` children
    produce byte-identical exploration sequences.  Verified to fail when genuine
    per-process entropy is injected into a method's RNG.  (The upstream
    reservoir-order source is guarded directly by
    ``test_reproducibility.test_reservoir_order_is_process_stable``; here the
    reservoir is a deterministic input.)
    """
    seq_a = _method_sequence(module_name, cls_name)
    seq_b = _method_sequence(module_name, cls_name)
    assert seq_a != "[]", f"{cls_name} explored nothing — stub/seed wiring is wrong"
    assert seq_a == seq_b, f"{cls_name} exploration diverged across processes"
