"""Run limits: heavy jobs queue one at a time, every heavy or scheduled job has a time
limit, pre-kickoff runs never wait on a heavy job, and a hung step dumps its stacks."""

from __future__ import annotations

import subprocess
import sys

import dagster_defs as d


def test_jobs_carry_limits_and_prekick_skips_the_heavy_queue():
    jobs = {j.name: j.tags for j in d.defs.jobs}
    for name in ("projections_refresh", "status_refresh", "ros_refresh"):
        assert jobs[name]["ffa/heavy"] == "true" and int(jobs[name]["dagster/max_runtime"]) > 0
    pre = jobs["pre_kickoff_refresh"]
    assert "ffa/heavy" not in pre and int(pre["dagster/priority"]) > 0
    assert int(pre["dagster/max_runtime"]) <= 30 * 60  # ends before the window does
    sched = {s.name: s.cron_schedule for s in d.defs.schedules}
    assert sched["daily_ros"] == "45 17 * * 1-6"  # not inside Sunday's early window


def test_hang_dump_writes_stacks_to_stderr():
    code = ("import time, dagster_defs as d\n"
            "with d._hang_dump(1):\n    time.sleep(2.5)\n")  # fmt: skip
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    assert "Thread" in r.stderr and "time.sleep" not in r.stdout


def test_hang_dump_is_cancelled_when_the_step_finishes():
    code = ("import time, dagster_defs as d\n"
            "with d._hang_dump(1):\n    pass\ntime.sleep(2)\n")  # fmt: skip
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "Thread" not in r.stderr
