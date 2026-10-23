"""Game statuses: Sleeper as a second source, practice-aware availability, and the case
that exposed the gap - a starter Out last week, not practicing, no designation yet."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest  # noqa: F401

from ffa.ingest import status
from ffa.model.inputs import _statuses
from ffa.model.sim_params import SimParams, is_rest
from ffa.service import startsit
from tests.test_api import warehouse  # noqa: F401  (fixture)

DNP = "Did Not Participate In Practice"
LIMITED = "Limited Participation in Practice"
FETCHED = datetime(2026, 10, 16, 22, 5, tzinfo=UTC)


def _injuries(rows):
    cols = ["season", "week", "gsis_id", "report_status", "practice_status",
            "report_primary_injury", "practice_primary_injury"]  # fmt: skip
    return pl.DataFrame(
        [dict(zip(cols, r, strict=True)) for r in rows],
        schema={c: pl.Int32 if c in ("season", "week") else pl.Utf8 for c in cols},
    ).with_columns(pl.lit(FETCHED).alias("_ingested_at"), pl.lit("PHI").alias("team"))


BARKLEY = "00-0034844"
# Week 5: Out. Week 6, Thursday: did not practice, no designation yet.
INJURIES = _injuries([
    (2026, 5, BARKLEY, "Out", DNP, "Hamstring", "Hamstring"),
    (2026, 6, BARKLEY, None, DNP, None, "Hamstring"),
    (2026, 6, "00-VET", None, DNP, None, "Not injury related - resting player"),
])  # fmt: skip

SIM = SimParams(p_play={
    f"pending|{DNP}|role|prev_out": 0.01, f"pending|{DNP}|role|prev_ok": 0.27,
    f"pending|{DNP}|any|any": 0.16, f"pending|{LIMITED}|any|any": 0.66,
    "Questionable|any|any": 0.63,
})  # fmt: skip


def test_play_probability_uses_practice_before_a_status_exists():
    assert SIM.play_probability(None, DNP, True, prev_out=True, final=False) == 0.01
    assert SIM.play_probability(None, DNP, True, prev_out=False, final=False) == 0.27
    assert SIM.play_probability(None, LIMITED, False, final=False) == 0.66  # falls back
    # Final report out, no designation: cleared to play despite the practice
    assert SIM.play_probability(None, DNP, True, prev_out=True, final=True) == 1.0
    assert SIM.play_probability(None, "Full Participation in Practice", True) == 1.0
    assert SIM.play_probability(None, None, True, prev_out=True) == 1.0  # off the report
    assert SIM.play_probability("Out", None, True) == 0.0
    assert SIM.play_probability("Questionable", DNP, True) == 0.63


def test_rest_days_are_not_injuries():
    assert is_rest("Not injury related - resting player") and is_rest("Rest")
    assert not is_rest("Hamstring") and not is_rest(None)
    st = _statuses(INJURIES, None, 2026, 6)
    vet = st.filter(pl.col("key") == "00-VET").row(0, named=True)
    assert vet["practice"] == "Full Participation in Practice"


def test_barkley_without_second_source_is_a_real_risk():
    row = _statuses(INJURIES, None, 2026, 6).filter(pl.col("key") == BARKLEY).row(0, named=True)
    assert row["injury_status"] is None and row["practice"] == DNP and row["prev_out"]
    assert row["injury_source"] == "nflverse"
    p = SIM.play_probability(row["injury_status"], row["practice"], True, row["prev_out"],
                             final=False)  # fmt: skip
    assert p == 0.01


def test_final_report_judged_from_designations():
    from ffa.model.inputs import _final_reports

    inj = _injuries([
        (2026, 6, "a", None, DNP, None, "Hamstring"),
        (2026, 6, "b", "Questionable", LIMITED, "Knee", "Knee"),
        (2026, 6, "c", None, DNP, None, "Ankle"),
    ]).with_columns(pl.Series("team", ["PHI", "DAL", "DAL"]))  # fmt: skip
    f = dict(_final_reports(inj, 2026, 6).select("team", "final_report").iter_rows())
    assert f == {"PHI": False, "DAL": True}


def _sleeper(statuses):
    players = pl.DataFrame(
        {
            "sleeper_id": [s[0] for s in statuses],
            "gsis_id": [None] * len(statuses),
            "injury_status": [s[1] for s in statuses],
            "injury_body_part": [s[2] for s in statuses],
            "practice_participation": [s[3] for s in statuses],
            "news_updated": ["1792185120000"] * len(statuses),  # 2026-10-16 21:12 UTC
        }
    )
    xw = pl.DataFrame({"sleeper_id": ["4866", "9"], "gsis_id": [BARKLEY, "00-IR"]})
    return status.sleeper_status(players, xw, FETCHED)


def test_sleeper_status_mapping():
    st = _sleeper([("4866", "Out", "Hamstring", "DNP"), ("9", "IR", "Knee", None),
                   ("77", "Questionable", None, None)])  # fmt: skip
    by = {r["key"]: r for r in st.iter_rows(named=True)}
    assert by[BARKLEY]["status"] == "Out" and by[BARKLEY]["practice"] == DNP
    assert by[BARKLEY]["updated_at"] == datetime(2026, 10, 16, 21, 12, tzinfo=UTC)
    assert by["00-IR"]["status"] == "Out"  # IR counts as out
    assert "77" not in by and len(by) == 2  # no crosswalk match: dropped, not guessed
    assert status.practice("Limited") == LIMITED and status.practice("") is None


def test_more_severe_source_wins_and_is_credited():
    sl = _sleeper([("4866", "Out", "Hamstring", "DNP")])
    row = _statuses(INJURIES, sl, 2026, 6).filter(pl.col("key") == BARKLEY).row(0, named=True)
    assert row["injury_status"] == "Out" and row["injury_source"] == "Sleeper"
    # nflverse more severe than Sleeper -> nflverse kept
    inj = _injuries([(2026, 6, BARKLEY, "Doubtful", DNP, "Hamstring", "Hamstring")])
    q = _sleeper([("4866", "Questionable", "Hamstring", None)])
    row = _statuses(inj, q, 2026, 6).row(0, named=True)
    assert row["injury_status"] == "Doubtful" and row["injury_source"] == "nflverse"


def _view(key, name, mean, p_play, status=None, pos="RB"):
    if_active = mean / p_play if p_play else None
    return startsit.PlayerView(
        key=key, name=name, team="PHI", position=pos, opponent="CAR",
        proj_if_active=if_active, median_if_active=if_active, p10_if_active=0,
        p25_if_active=0, p75_if_active=if_active, p90_if_active=if_active,
        play_probability=p_play, expected_points=mean, active_sims=2000 if p_play else 0,
        forced_active_batch=False, expected_stats_if_active={}, injury_status=status,
        injury_source="Sleeper" if status else None, injury_updated_at="Fri Oct 16 4:12 PM CDT",
        injury_fetched_at="Fri Oct 16 5:05 PM CDT", expected_stats={}, points_by_rule={},
        drivers=[], mean=mean, median=mean, p10=0, p90=mean * 2,
    )  # fmt: skip


def test_lineup_never_starts_a_ruled_out_player():
    views = {"a": _view("a", "Out Guy", 0.0, 0.0, "Out"), "b": _view("b", "Backup", 5.8, 1.0)}
    lineup = startsit.best_lineup(views, ["RB"])
    assert lineup[0]["key"] == "b"


def test_injury_line_carries_source_and_time():
    line = startsit.injury_line({
        "injury_status": "Out", "practice": DNP, "injury": "Hamstring", "p_play": 0.0,
        "injury_source": "Sleeper", "injury_updated_at": datetime(2026, 10, 16, 21, 12, tzinfo=UTC),
        "injury_fetched_at": FETCHED,
    })  # fmt: skip
    assert line.startswith("Injury: OUT (hamstring), did not practice [Sleeper, updated Fri Oct 16")
    assert "4:12 PM CDT; fetched Fri Oct 16 5:05 PM CDT]" in line
    pending = startsit.injury_line({"practice": DNP, "p_play": 0.01, "prev_out": True,
                                    "injury_fetched_at": FETCHED})  # fmt: skip
    assert pending.startswith("No game status yet, but did not practice this week; was Out last")
    assert "plays in 1% of simulations" in pending
    assert startsit.injury_line({"p_play": 1.0}) is None


def test_compare_flags_out_player_first_and_never_recommends_him(warehouse):  # noqa: F811
    from ffa.scoring.rules import default_rules
    from ffa.warehouse import duck

    w = startsit.load_week(duck.connect(warehouse), 2026, 6, ["00-A", "00-B"])
    # Alpha (the higher projection) is ruled out by Sleeper after the sims were stored.
    w.inputs = w.inputs.with_columns(
        pl.when(pl.col("key") == "00-A").then(0.0).otherwise(pl.col("p_play")).alias("p_play"),
        pl.when(pl.col("key") == "00-A").then(pl.lit("Out")).otherwise(pl.col("injury_status"))
        .alias("injury_status"),
    )  # fmt: skip
    c = startsit.compare(w, default_rules())
    assert c.recommendation == "00-B"
    assert c.notes[0].startswith("Alpha Back is ruled OUT")
    assert not c.close_call


def test_build_passes_report_state_to_availability(monkeypatch):
    """Regression: the final-report flag must reach play_probability (it once defaulted)."""
    import inspect

    from ffa.model import inputs

    src = inspect.getsource(inputs.build)
    assert 'final=r["final_report"]' in src and 'prev_out=r["prev_out"]' in src


def test_drivers_show_simulated_role_not_just_usual_share():
    """Regression: a backup filling in showed his backup share while the simulation gave
    him the starter's carries."""
    usage = {
        "player": {"rush_att": 16.4, "rec_tgt": 3.2, "rush_td": 0.24, "rec_td": 0.09},
        "team": {"rush_att": 27.9, "rec_tgt": 29.0, "rush_td": 0.72, "rec_td": 1.0},
        "absent_car": ["Saquon Barkley (1% to play)"], "absent_tgt": [],
    }  # fmt: skip
    lines = startsit._usage_lines({"car_share_s": 0.399, "tgt_share_s": 0.089}, "RB", usage)
    assert lines[0] == ("Projected 16.4 carries, 59% of the team's rushing attempts with "
                        "Saquon Barkley (1% to play) unlikely to play (usual share of designed "
                        "runs 39.9%)")  # fmt: skip
    assert lines[1] == "Projected rushing TDs: 0.24 of the team's 0.72 (33%)"
    assert lines[2].startswith("Projected 3.2 targets, 11% of team targets (usual share 8.9%)")


def test_api_drivers_include_simulated_volume(warehouse):  # noqa: F811
    from ffa.scoring.rules import default_rules
    from ffa.warehouse import duck

    w = startsit.load_week(duck.connect(warehouse), 2026, 6, ["00-A"])
    v = startsit.player_views(w, default_rules())["00-A"]
    assert any(d.startswith("Projected ") and "carries" in d for d in v.drivers)


def test_inactive_and_ir_players_are_explained():
    assert startsit.injury_line({"roster_status": "RES", "p_play": 0.0}).startswith(
        "On injured reserve [team roster"
    )
    assert startsit.injury_line({"roster_status": "INA", "p_play": 0.0}).startswith(
        "Inactive this week"
    )


def test_inheritance_moves_an_absent_starters_role_to_his_position():
    import numpy as np

    from ffa.model.sim import _inherit, _propensity

    shares = np.array([0.5, 0.3, 0.1])  # RB1 (out), RB2, QB
    pos = np.array(["RB", "RB", "QB"])
    active = np.array([[False], [True], [True]])
    assert np.allclose(_inherit(shares, active, pos, 0.0)[:, 0], [0, 0.3, 0.1])  # old model
    w = _inherit(shares, active, pos, 1.0)[:, 0]
    assert np.allclose(w, [0, 0.8, 0.1])  # all of RB1's share stays with the RBs
    # TD rate: RB2 inherits RB1's goal-line share along with his carries
    td = np.array([0.6, 0.1, 0.3])
    p0 = _propensity(td, shares, active, pos, 0.0)[:, 0]
    p1 = _propensity(td, shares, active, pos, 1.0)[:, 0]
    assert p1[1] > p0[1] and np.isclose(p1[2], p0[2])


def test_inheritance_is_on_by_default_and_reaches_old_param_files(tmp_path):
    import json

    from ffa.model.sim_params import INHERIT_KEEP

    assert INHERIT_KEEP == 0.5 and SimParams().inherit_keep == 0.5
    f = tmp_path / "sim_params.json"  # fitted before the field existed
    f.write_text(json.dumps({"p_play": {}}))
    assert SimParams.load(f).inherit_keep == 0.5


def test_backup_takes_most_of_an_out_starters_goal_line_work():
    """PHI 2026 week 6 shape: RB1 out, RB2 his backup, QB with a high TD rate per carry.
    Shares as estimated that week."""
    import numpy as np

    from ffa.model.sim import _inherit, _propensity

    pos = np.array(["RB", "QB", "RB", "RB"])  # Barkley, Hurts, Pierce, Steele
    car = np.array([0.479, 0.094, 0.399, 0.068])
    td = np.array([0.419, 0.342, 0.265, 0.059])
    active = np.array([[False], [True], [True], [True]])
    keep = SimParams().inherit_keep
    carries = _inherit(car, active, pos, keep)[:, 0]
    rate = _propensity(td, car, active, pos, keep)[:, 0]
    td_weight = carries * rate  # expected TD weight per player
    old = _inherit(car, active, pos, 0.0)[:, 0] * _propensity(td, car, active, pos, 0.0)[:, 0]
    pierce, hurts = 2, 1
    assert td_weight[pierce] / td_weight.sum() > old[pierce] / old.sum()
    assert td_weight[pierce] > td_weight[hurts]


def test_last_weeks_status_carries_until_the_new_report():
    """Regression: Monday/Tuesday, before the week's first report, a player Out last week
    was projected at 100% (absent from this week's report read as healthy)."""
    from ffa.model.inputs import _final_reports, _previous_status

    inj = _injuries([
        (2026, 6, "out_guy", "Out", DNP, "Knee", "Knee"),
        (2026, 6, "q_guy", "Questionable", LIMITED, "Ankle", "Ankle"),
        (2026, 4, "old_out", "Out", DNP, "Hand", "Hand"),  # Out in week 4, fine in week 6
        (2026, 6, "healthy", None, "Full Participation in Practice", None, None),
    ])  # fmt: skip
    prev = {r["key"]: r for r in _previous_status(inj, 2026, 7).iter_rows(named=True)}
    assert prev["out_guy"]["prev_status"] == "Out" and prev["out_guy"]["prev_week"] == 6
    assert prev["q_guy"]["prev_status"] == "Questionable"
    assert "old_out" not in prev and "healthy" not in prev
    assert _final_reports(inj, 2026, 7).height == 0  # week 7 report not out: unpublished

    sim = SimParams(p_play={"carry|Out|role": 0.38, "carry|Questionable|any": 0.9})
    assert sim.play_probability(None, None, True, prev_status="Out", published=False) == 0.38
    assert sim.play_probability(None, None, False, prev_status="Questionable",
                                published=False) == 0.9  # fmt: skip
    # once the team's report is out, absence from it means healthy
    assert sim.play_probability(None, None, True, prev_status="Out", published=True) == 1.0
    # an old parameter file without the table still doesn't say 100%
    assert SimParams().play_probability(None, None, True, prev_status="Out",
                                        published=False) < 0.5  # fmt: skip


def test_carried_status_is_explained():
    line = startsit.injury_line({"p_play": 0.31, "report_published": False,
                                 "prev_status": "Out", "prev_week": 6})  # fmt: skip
    assert line.startswith("No injury report yet this week; was Out in week 6 [nflverse")
    assert "plays in 31% of simulations" in line


def test_projection_run_fetches_sleeper_when_missing(tmp_path, monkeypatch):
    """Regression: the Sleeper status feed never ran because its schedule was off.
    The projection run now fetches it when missing or over a day old."""
    import dagster as dg

    import dagster_defs as d

    calls = []
    monkeypatch.setattr(d.settings, "warehouse_dir", str(tmp_path))
    monkeypatch.setattr(d, "_fetch_sleeper_status", lambda log: calls.append(1) or {})
    monkeypatch.setattr(d.projections_mod, "upcoming_week", lambda con, season: 7)
    monkeypatch.setattr(d.projections_mod, "load_models", lambda path: None)
    monkeypatch.setattr(
        d.projections_mod, "run_week",
        lambda *a, **k: d.projections_mod.RunResult(2026, 7, 1, 1),
    )  # fmt: skip
    monkeypatch.setattr(d.snapshots, "take", lambda *a, **k: (None, 0))
    monkeypatch.setattr(d, "_league_scoring", lambda log: {})
    d.weekly_projections(dg.build_asset_context())
    assert calls == [1]  # never fetched -> fetched
    (tmp_path / "dims").mkdir()
    pl.DataFrame({"sleeper_id": ["1"]}).write_parquet(tmp_path / "dims" / "sleeper_players.parquet")
    d.weekly_projections(dg.build_asset_context())
    assert calls == [1]  # fresh -> not fetched again (Sleeper asks for once a day)


def test_schedules_run_by_default():
    import dagster as dg

    from dagster_defs import defs

    assert all(s.default_status == dg.DefaultScheduleStatus.RUNNING for s in defs.schedules)


def test_questionable_from_sleeper_keeps_the_official_practice_note():
    """Midweek: Sleeper lists Questionable with no practice note, the official report has
    practice but no designation yet. The note must survive, or every Questionable
    player falls back to the same 'practice unknown' rate."""
    full = "Full Participation in Practice"
    inj = _injuries([(2026, 6, BARKLEY, None, LIMITED, None, "Hamstring"),
                     (2026, 6, "00-IR", None, full, None, "Knee")])  # fmt: skip
    sl = _sleeper([("4866", "Questionable", "Hamstring", None),
                   ("9", "Questionable", "Knee", "DNP")])  # fmt: skip
    rows = {r["key"]: r for r in _statuses(inj, sl, 2026, 6).iter_rows(named=True)}
    assert rows[BARKLEY]["injury_status"] == "Questionable"
    assert rows[BARKLEY]["injury_source"] == "Sleeper"
    assert rows[BARKLEY]["practice"] == LIMITED
    assert rows["00-IR"]["practice"] == full  # the official report outranks Sleeper's note
    sp = SimParams(p_play={f"Questionable|{LIMITED}|role": 0.75,
                           "Questionable|any|role": 0.73})  # fmt: skip
    assert sp.play_probability("Questionable", rows[BARKLEY]["practice"], True) == 0.75


def test_game_day_check_names_the_inactives_time():
    ko = datetime(2026, 10, 25, 17, 0, tzinfo=UTC)  # Sun 12:00 PM CDT
    fri = datetime(2026, 10, 23, 22, 0, tzinfo=UTC)
    line = startsit.game_day_check(0.75, "Questionable", ko, now=fri)
    assert line.startswith("Questionable: check inactives")
    assert "Sun Oct 25 10:30 AM" in line and "once a day" in line
    assert startsit.game_day_check(1.0, None, ko, now=fri) is None  # healthy
    assert startsit.game_day_check(0.0, "Out", ko, now=fri) is None  # ruled out
    assert startsit.game_day_check(0.75, "Questionable", ko, now=ko) is None  # kicked off
    assert "confirm he is active" in startsit.game_day_check(0.3, None, None)


def test_questionable_player_gets_the_check_in_views(warehouse):  # noqa: F811
    from ffa.scoring.rules import default_rules
    from ffa.warehouse import duck

    w = startsit.load_week(duck.connect(warehouse), 2026, 6, ["00-A", "00-B"])
    w.inputs = w.inputs.with_columns(
        pl.when(pl.col("key") == "00-A").then(0.75).otherwise(1.0).alias("p_play"),
        pl.when(pl.col("key") == "00-A").then(pl.lit("Questionable"))
        .otherwise(None).alias("injury_status"),
    )  # fmt: skip
    v = startsit.player_views(w, default_rules())
    assert v["00-A"].game_day_check.startswith("Questionable:")
    assert v["00-B"].game_day_check is None
