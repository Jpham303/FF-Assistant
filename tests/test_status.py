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
    ).with_columns(pl.lit(FETCHED).alias("_ingested_at"))


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
    f = dict(_final_reports(inj, 2026, 6).iter_rows())
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
    return startsit.PlayerView(
        key=key, name=name, team="PHI", position=pos, opponent="CAR", mean=mean,
        median=mean, p10=0, p90=mean * 2, play_probability=p_play, injury_status=status,
        injury_source="Sleeper" if status else None, injury_updated_at="Fri Oct 16 4:12 PM CDT",
        injury_fetched_at="Fri Oct 16 5:05 PM CDT", expected_stats={}, points_by_rule={},
        drivers=[],
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
