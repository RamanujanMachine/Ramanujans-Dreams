"""Centralised, reproducible RNG seeding.

All randomness in the samplers and search methods derives from a single **master
seed** (``search_config.GLOBAL_SEED``) plus a per-unit *context* (typically the
shard id, the method name, and — for search methods — the target constant).
The derived streams are:

* **reproducible** — same master seed + same context ⇒ same stream, regardless
  of execution order or of spawn-vs-fork multiprocessing.  Context parts are
  mixed with :mod:`hashlib` (not the per-process-salted built-in ``hash``), so a
  derivation in a spawned worker matches the parent.
* **independent** — different contexts derive statistically independent streams
  via :class:`numpy.random.SeedSequence`.

Set ``search_config.GLOBAL_SEED = None`` for nondeterministic (OS-entropy) runs.

IMPORTANT — multiprocessing invariant
-------------------------------------
Randomness must stay in the **main** process.  The search-stage worker pools
(``evaluate_batch`` / ``make_eval_pool``) and the analyzer execute only
*deterministic* work (δ-walks, attribute computation); the stochastic loops of
the samplers and search methods run in the parent.  Do **not** move an
RNG-driven step into a worker: under *spawn* every worker re-imports this module,
so any global RNG seeding would be identical across workers (all workers would
draw the same stream).  If a parallel region genuinely needs randomness, derive a
per-unit seed here in the parent and pass it in explicitly (this is what the
numba walk kernels do via ``rng_seed``).
"""

import hashlib
import random as _py_random
from typing import List, Optional

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng

# Legacy module-level default.  The *live* master seed is
# ``search_config.GLOBAL_SEED``; this constant is the fallback used only if the
# config cannot be imported (and is kept for backward-compatible imports).
GLOBAL_SEED = 42

_UINT64_MASK = (1 << 64) - 1


def _master_seed() -> Optional[int]:
    """Return the live master seed (``search_config.GLOBAL_SEED``).

    Imported lazily so this low-level module stays free of an import cycle
    (``dreamer.configs`` does not depend on ``dreamer.utils.rand``).  ``None``
    selects nondeterministic (OS-entropy) seeding.

    :return: The master seed, or ``None`` for nondeterministic runs.
    """
    try:
        from dreamer.configs.search import search_config
        return search_config.GLOBAL_SEED
    except Exception:
        return GLOBAL_SEED


def _context_entropy(context) -> List[int]:
    """Convert a context tuple into a list of stable 64-bit entropy integers.

    Integers pass through (masked to 64 bits); everything else is hashed with
    SHA-256 of its ``repr`` so the mapping is identical across processes and runs
    (unlike the built-in ``hash``, which is salted per process).

    :param context: Arbitrary hashable/representable context parts.
    :return: List of 64-bit integers suitable for :class:`SeedSequence`.
    """
    out: List[int] = []
    for part in context:
        if isinstance(part, (int, np.integer)):
            out.append(int(part) & _UINT64_MASK)
        else:
            digest = hashlib.sha256(repr(part).encode("utf-8")).digest()
            out.append(int.from_bytes(digest[:8], "little"))
    return out


def _seed_sequence(context) -> Optional[SeedSequence]:
    """Build the :class:`SeedSequence` for *context*, or ``None`` if nondeterministic.

    :param context: Context parts identifying the unit of work.
    :return: A :class:`SeedSequence`, or ``None`` when the master seed is ``None``.
    """
    master = _master_seed()
    if master is None:
        return None
    return SeedSequence([int(master) & _UINT64_MASK] + _context_entropy(context))


def derive_rng(*context) -> Generator:
    """Return an independent, reproducible NumPy :class:`Generator` for *context*.

    With a non-``None`` master seed the stream is fully determined by
    ``(GLOBAL_SEED, *context)``; with ``GLOBAL_SEED = None`` it is seeded from OS
    entropy (nondeterministic).

    :param context: Parts identifying the unit of work (e.g. ``shard_id, "ga",
        constant``).
    :return: A seeded (or entropy-seeded) :class:`numpy.random.Generator`.
    """
    return default_rng(_seed_sequence(context))


def derive_seed(*context) -> int:
    """Return a reproducible 32-bit integer seed for *context*.

    Suitable for seeding numba's RNG (``np.random.seed``), which the JIT walk
    kernels call.  Nondeterministic when the master seed is ``None``.

    :param context: Parts identifying the unit of work.
    :return: A 32-bit non-negative integer seed.
    """
    ss = _seed_sequence(context)
    if ss is None:
        return int(default_rng().integers(0, 2 ** 31 - 1))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def derive_py_random(*context) -> _py_random.Random:
    """Return a reproducible :class:`random.Random` instance for *context*.

    Use for code that needs the stdlib ``random`` interface (e.g. genetic-search
    crossover / selection) without touching the shared global ``random`` state.

    :param context: Parts identifying the unit of work.
    :return: A seeded :class:`random.Random`.
    """
    return _py_random.Random(derive_seed(*context))
