"""Tests for :mod:`runmon.estimators`.

  - The schedule functions are exactly the project's SA schedule
    (``dreamer.search.methods.annealing.annealing_scan._get_temp``) and their
    inverse recovers ``k`` from ``T`` for both schedules.
  - Move count at ``Tmin``: linear ``T0/Tmin - 1``; log ``e^(T0/Tmin) - 1``
    (infinite for the default ``T0=1, Tmin=1e-4``), so ``max_iters`` is the
    real stop with the defaults.
  - ``ClockEstimator`` takes the most-consumed budget and ignores infinite /
    missing limits.
"""

import math

import pytest

from dreamer.search.methods.annealing.annealing_scan import _get_temp
from runmon import estimators as es


@pytest.mark.parametrize("schedule", es.SCHEDULES)
@pytest.mark.parametrize("k", [0, 1, 2, 5, 37, 499, 10_000])
def test_schedule_matches_project_sa(schedule, k):
    assert es.schedule_temperature(k, 1.7, schedule) == pytest.approx(_get_temp(1.7, k, schedule))


@pytest.mark.parametrize("schedule", es.SCHEDULES)
@pytest.mark.parametrize("k", [1, 2, 5, 37, 499])
def test_inverse_recovers_k(schedule, k):
    T = es.schedule_temperature(k, 1.0, schedule)
    assert es.schedule_inverse(T, 1.0, schedule) == pytest.approx(k, rel=1e-9)


def test_k_at_tmin():
    assert es.k_at_temperature(1e-4, 1.0, "linear") == 9999
    assert _get_temp(1.0, 9999, "linear") <= 1e-4 < _get_temp(1.0, 9998, "linear")
    assert es.k_at_temperature(1e-4, 1.0, "log") == math.inf       # e^10000
    k = es.k_at_temperature(0.25, 1.0, "log")                       # e^4 - 1 ≈ 53.6 -> 54
    assert k == 54 and _get_temp(1.0, 54, "log") <= 0.25 < _get_temp(1.0, 53, "log")


def test_sa_k_end_defaults_are_capped_by_max_iters():
    assert es.sa_k_end(1.0, 1e-4, 500, "log") == 500
    assert es.sa_k_end(1.0, 1e-4, 500, "linear") == 500
    assert es.sa_k_end(1.0, 0.25, 500, "linear") == 3
    assert es.sa_k_end(1.0, 0.25, None, "log") == 54


def test_clock_estimator():
    est = es.ClockEstimator(accepted=(120, 500), steps=(3000, 10_000), never=(5, math.inf))
    assert est.fraction() == pytest.approx(0.3) and est.dominant() == "steps"
    assert es.ClockEstimator(a=(5, 0), b=(1, None)).fraction() is None
    assert es.ClockEstimator(a=(900, 500)).fraction() == 1.0


def test_sa_estimator():
    est = es.sa_estimator(250, 1000, T0=1.0, Tmin=1e-4, max_iters=500,
                          max_total_steps=10_000, schedule="log")
    assert est.fraction() == pytest.approx(0.5) and est.dominant() == "accepted"
    est = es.sa_estimator(10, 9000, T0=1.0, Tmin=1e-4, max_iters=500,
                          max_total_steps=10_000, schedule="log")
    assert est.dominant() == "steps"                 # stuck rejecting: the step cap wins
