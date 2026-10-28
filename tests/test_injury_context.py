"""The injury story: what this season's reports show and which way they are moving."""

from __future__ import annotations

import polars as pl

from ffa.model import injury_context as ic

LP, FP, DNP = (
    "Limited Participation in Practice",
    "Full Participation in Practice",
    "Did Not Participate In Practice",
)


def test_spans_and_runs():
    assert ic._spans([1, 2, 3, 6, 7, 9]) == "wk 1–3, 6–7, 9"
    got = ic._runs([(4, "Out"), (5, "Out"), (6, "Questionable")])
    assert got == "Out wk 4–5, Questionable wk 6"


def test_returning_from_last_seasons_injury_reads_as_returning_not_new():
    """Oct 28: Charbonnet, last season's ACL, PUP weeks 1-6, Out but practicing weeks
    4-7, Questionable now. The assistant called him done for the season."""
    reports = [{"week": w, "status": "Out", "practice": LP if w < 7 else FP} for w in (4, 5, 6, 7)]
    line, returning = ic.story(
        week=8, reports=reports, reserve_weeks=[1, 2, 3, 4, 5, 6], latest_roster="ACT",
        games=0, prev_games=16, prev_snap=0.48,
        current={"injury_status": "Questionable", "injury": "Knee - ACL",
                 "injury_source": "Sleeper"},
    )  # fmt: skip
    assert returning
    assert "On a reserve list (IR/PUP) wk 1–6, since back on the active roster" in line
    assert "Out wk 4–7; practice: limited wk 4–6, full wk 7" in line
    assert "Trend: improving (Out → Questionable), and practicing: on track to return" in line
    assert "the injury predates this season" in line
    assert "last season 48% of snaps in 16 games" in line


def test_trends():
    out = [{"week": 6, "status": "Out", "practice": DNP}, {"week": 7, "status": "Out",
                                                           "practice": DNP}]  # fmt: skip
    base = dict(week=8, reserve_weeks=[], latest_roster="ACT", games=5, prev_games=None,
                prev_snap=None)  # fmt: skip
    line, _ = ic.story(reports=out, current={"injury_status": "Out"}, **base)
    assert "Trend: not practicing" in line and "First listed week 6" in line
    working = [{"week": 7, "status": "Out", "practice": LP}]
    line, _ = ic.story(reports=working, current={"injury_status": "Out"}, **base)
    assert "still listed Out but practicing" in line
    q = [{"week": 7, "status": "Questionable", "practice": LP}]
    line, _ = ic.story(reports=q, current={"injury_status": "Out"}, **base)
    assert "worsening (Questionable → Out)" in line
    line, _ = ic.story(reports=q, current={}, **base)
    assert "no designation so far this week after Questionable in week 7" in line


def test_nothing_to_say_for_a_healthy_regular():
    line, returning = ic.story(week=8, reports=[], reserve_weeks=[], latest_roster="ACT",
                               games=7, prev_games=17, prev_snap=0.9, current={})  # fmt: skip
    assert line is None and not returning


def test_old_reports_dont_count_as_a_current_stretch():
    old = [{"week": 2, "status": "Questionable", "practice": LP}]
    line, _ = ic.story(week=8, reports=old, reserve_weeks=[], latest_roster="ACT", games=6,
                       prev_games=None, prev_snap=None, current={})  # fmt: skip
    assert line is None


def test_contexts_from_frames_and_sleeper_note():
    injuries = pl.DataFrame(
        {
            "gsis_id": ["z"] * 2,
            "season": [2026] * 2,
            "week": [6, 7],
            "report_status": ["Out", "Out"],
            "practice_status": [LP, FP],
        }
    )
    rosters = pl.DataFrame(
        {
            "gsis_id": ["z", "z", "h"],
            "season": [2026] * 3,
            "week": [1, 7, 7],
            "status": ["RES", "ACT", "ACT"],
        }
    )
    pg = pl.DataFrame({"key": ["z", "z", "h"], "season": [2025, 2025, 2026], "week": [1, 2, 3],
                       "snaps": [30, 30, 50], "snap_pct": [0.5, 0.5, 0.8],
                       "played": [True, True, True]})  # fmt: skip
    cur = pl.DataFrame({"key": ["z"], "injury_status": ["Questionable"], "practice": [None],
                        "injury": ["Knee - ACL"], "designation": ["Questionable"],
                        "note": ["Activated from PUP;  expected to play"], "listing": [None],
                        "injury_source": ["Sleeper"]})  # fmt: skip
    out = ic.contexts(season=2026, week=8, keys=["z", "h"], injuries=injuries, rosters=rosters,
                      player_games=pg, current=cur)  # fmt: skip
    got = {r["key"]: r for r in out.iter_rows(named=True)}
    assert (
        got["z"]["returning"]
        and 'Sleeper note: "Activated from PUP; expected to play"' in (got["z"]["injury_context"])
    )
    assert got["h"]["injury_context"] is None
