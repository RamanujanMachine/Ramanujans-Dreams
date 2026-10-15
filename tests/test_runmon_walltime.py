"""Tests for :mod:`runmon.walltime` — when will the scheduler kill this job?

``QSTAT_ZEUS`` is real ``qstat -f`` output shape from the Zeus PBS cluster
(job 4797556, run from inside the job, 2026-10-15).

  - ``[[HH:]MM:]SS`` parsing (hours beyond 24), rejects garbage.
  - PBS: end = now + limit - used, from ``qstat -f $PBS_JOBID``.
  - Precedence: explicit env end time > Torque ``PBS_WALLTIME`` > PBS qstat > Slurm.
  - Slurm ``squeue`` end time; ``UNLIMITED`` / failures give no deadline.
  - Outside a batch job: no deadline, and no command is run.
"""

import subprocess
from datetime import datetime

import pytest

from runmon import walltime as wt

QSTAT_ZEUS = """Job Id: 4797556.zeus-master
    Job_Name = zeus_smoke
    resources_used.cpupercent = 129
    resources_used.walltime = 00:01:40
    job_state = R
    exec_host = z001/1*4
    Resource_List.walltime = 00:10:00
    stime = Thu Oct 15 18:19:58 2026
"""


class FakeRunner:
    def __init__(self, stdout="", returncode=0, exc=None):
        self.stdout, self.returncode, self.exc, self.calls = stdout, returncode, exc, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if self.exc:
            raise self.exc
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, "")


@pytest.mark.parametrize("text, seconds", [
    ("00:10:00", 600), ("336:00:00", 336 * 3600), ("05:30", 330), ("42", 42),
    ("1:02:03", 3723), ("", None), ("ab:cd", None), ("1:2:3:4", None), ("-1:00", None),
])
def test_parse_hms(text, seconds):
    assert wt.parse_hms(text) == seconds


def test_parse_qstat():
    assert wt.parse_qstat_walltime(QSTAT_ZEUS) == {"limit": 600, "used": 100}
    assert wt.parse_qstat_walltime("Job Id: 1\n") == {"limit": None, "used": None}


def test_pbs_deadline_from_qstat():
    run = FakeRunner(QSTAT_ZEUS)
    d = wt.detect({"PBS_JOBID": "4797556.zeus-master"}, runner=run, now=1000.0)
    assert d.source == "pbs" and d.limit == 600
    assert d.end == pytest.approx(1000.0 + 600 - 100)
    assert d.remaining(1100.0) == pytest.approx(400)
    assert run.calls == [["qstat", "-f", "4797556.zeus-master"]]


@pytest.mark.parametrize("runner", [FakeRunner(exc=FileNotFoundError("qstat")),
                                    FakeRunner(returncode=1),
                                    FakeRunner("Job Id: 1\n    job_state = R\n")])
def test_pbs_unavailable_gives_none(runner):
    assert wt.detect({"PBS_JOBID": "1"}, runner=runner) is None


def test_explicit_end_wins():
    env = {"RUNMON_WALLTIME_END": "2026-10-16T12:00:00", "PBS_JOBID": "1"}
    run = FakeRunner(QSTAT_ZEUS)
    d = wt.detect(env, runner=run)
    assert d.source == "env" and d.end == datetime(2026, 10, 16, 12).timestamp()
    assert run.calls == []
    assert wt.detect({"RUNMON_WALLTIME_END": "1760600000"}).end == 1760600000


def test_torque_env():
    d = wt.detect({"PBS_WALLTIME": "3600", "PBS_JOBID": "1"}, runner=FakeRunner(), start=500.0)
    assert d.source == "torque" and d.end == 4100.0


def test_slurm():
    d = wt.detect({"SLURM_JOB_ID": "7"}, runner=FakeRunner("2026-10-16T12:00:00\n"))
    assert d.source == "slurm" and d.end == datetime(2026, 10, 16, 12).timestamp()
    assert wt.detect({"SLURM_JOB_ID": "7"}, runner=FakeRunner("UNLIMITED\n")) is None


def test_no_job_runs_nothing():
    run = FakeRunner(QSTAT_ZEUS)
    assert wt.detect({}, runner=run) is None
    assert run.calls == []
