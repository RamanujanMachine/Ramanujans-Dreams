"""
Progress estimates for loops whose length is not a fixed count.

:class:`ClockEstimator` — a loop that stops when the **first** of several
budgets runs out (accepted moves, total steps, temperature, time ...) is as far
along as its most-consumed budget: ``fraction = max_i(value_i / limit_i)``.

Simulated-annealing schedules — the temperature is a deterministic function of
the accepted-move count ``k`` (``log``: ``T0/ln(k+1)``, ``linear``: ``T0/(k+1)``,
with ``T = T0`` at ``k = 0``), so reading ``k`` back from ``T``
(:func:`schedule_inverse`) carries no information beyond ``k`` itself.  What
the temperature *does* give is the move count at which ``T`` reaches ``Tmin``
(:func:`k_at_temperature`); the run ends at the earlier of that and
``max_iters`` (:func:`sa_k_end`).  With the common setting ``T0=1, Tmin=1e-4``
and the log schedule, ``Tmin`` would be reached only at ``k ≈ e^10000`` — the
iteration cap is the real clock, and the estimator reports that honestly.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

SCHEDULES = ("log", "linear")


def schedule_temperature(k: float, T0: float, schedule: str) -> float:
    """Temperature after ``k`` accepted moves (``log``: ``T0/ln(k+1)``, ``linear``: ``T0/(k+1)``)."""
    if schedule == "log":
        return T0 / math.log(k + 1) if k > 0 else T0
    return T0 / (k + 1)


def schedule_inverse(T: float, T0: float, schedule: str) -> float:
    """Accepted-move count ``k`` at which the schedule reaches temperature ``T``.

    ``inf`` when ``T`` is unreachable in floating point.  Note the ``log``
    schedule is not monotone at the start (``T0/ln 2 > T0`` at ``k = 1``); the
    inverse returns the solution of ``T = T0/ln(k+1)``.
    """
    if T <= 0:
        return math.inf
    if schedule == "log":
        x = T0 / T
        if x > 700:                        # exp overflow
            return math.inf
        return max(0.0, math.exp(x) - 1.0)
    return max(0.0, T0 / T - 1.0)


def k_at_temperature(Tmin: float, T0: float, schedule: str) -> float:
    """Accepted moves until the temperature first drops to ``Tmin`` or below."""
    k = schedule_inverse(Tmin, T0, schedule)
    return float(math.ceil(k - 1e-9)) if math.isfinite(k) else math.inf   # tolerate rounding


def sa_k_end(T0: float, Tmin: float, max_iters: Optional[float], schedule: str) -> float:
    """Accepted moves at which annealing stops: ``min(max_iters, k at Tmin)``."""
    k_t = k_at_temperature(Tmin, T0, schedule)
    if max_iters is None or max_iters <= 0:
        return k_t
    return min(float(max_iters), k_t)


class ClockEstimator:
    """``fraction = max(value / limit)`` over named budgets (clocks).

    >>> est = ClockEstimator(accepted=(120, 500), steps=(3000, 10_000))
    >>> round(est.fraction(), 2), est.dominant()
    (0.3, 'steps')
    """

    def __init__(self, **clocks: Tuple[float, float]):
        self.clocks: Dict[str, Tuple[float, float]] = {
            name: (float(v), float(lim)) for name, (v, lim) in clocks.items()
            if lim is not None and math.isfinite(float(lim)) and float(lim) > 0
        }

    def _ratios(self) -> Dict[str, float]:
        return {n: min(1.0, max(0.0, v / lim)) for n, (v, lim) in self.clocks.items()}

    def fraction(self) -> Optional[float]:
        """Most-consumed budget's fraction, or ``None`` with no finite budget."""
        r = self._ratios()
        return max(r.values()) if r else None

    def dominant(self) -> Optional[str]:
        """Name of the budget that will run out first."""
        r = self._ratios()
        return max(r, key=r.get) if r else None


def sa_estimator(k: float, total_steps: float, *, T0: float, Tmin: float,
                 max_iters: Optional[float], max_total_steps: Optional[float],
                 schedule: str) -> ClockEstimator:
    """Progress clocks of a simulated-annealing run.

    :param k: Accepted moves so far (the schedule's clock).
    :param total_steps: Loop iterations so far (accepted + rejected).
    """
    return ClockEstimator(
        accepted=(k, sa_k_end(T0, Tmin, max_iters, schedule)),
        steps=(total_steps, max_total_steps if max_total_steps else math.inf),
    )
