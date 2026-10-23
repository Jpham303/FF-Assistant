"""Live accuracy tracking: snapshots frozen before kickoff, actuals, metrics."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from ffa.tracking import metrics, report, snapshots
from ffa.warehouse import duck, store
from tests.test_api import SEASON, WEEK, warehouse  # noqa: F401  (fixture)

THU = datetime(2026, 10, 16, 0, 15, tzinfo=UTC)  # Thu 8:15 PM ET
SUN = datetime(2026, 10, 18, 17, 0, tzinfo=UTC)  # Sun 1:00 PM ET


@pytest.fixture
def wh(warehouse):  # noqa: F811
    sched = pl.DataFrame({
        "season": [SEASON] * 2, "week": [WEEK] * 2, "home_score": [None, None],
        "game_id": ["g_thu", "g_sun"], "gameday": ["2026-10-15", "2026-10-18"],
        "gametime": ["20:15", "13:00"], "home_team": ["DAL", "JAX"], "away_team": ["GB", "HOU"],
    }).with_columns(pl.col("season", "week").cast(pl.Int32))  # fmt: skip
    store.write(sched, warehouse, "schedules")
    return warehouse


LEAGUES = {"default": snapshots.leagues_scoring(None)["default"]}


def test_kickoffs_are_eastern_times_by_team(wh):
    ko = {r["team"]: r["kickoff"] for r in snapshots.kickoffs(duck.connect(wh), SEASON, WEEK)
          .iter_rows(named=True)}  # fmt: skip
    assert ko["DAL"] == ko["GB"] == THU and ko["JAX"] == SUN


def test_snapshot_is_write_once_and_skips_identical_forecasts(wh, tmp_path):
    con = duck.connect(wh)
    models = tmp_path / "models"
    models.mkdir()
    (models / "x.json").write_text("{}")
    t1, t2 = datetime(2026, 10, 15, 12, tzinfo=UTC), datetime(2026, 10, 15, 13, tzinfo=UTC)
    p1, n = snapshots.take(con, wh, models, SEASON, WEEK, LEAGUES, now=t1)
    assert p1 is not None and n == 5
    p2, _ = snapshots.take(con, wh, models, SEASON, WEEK, LEAGUES, now=t2)
    assert p2 is None  # identical forecast: the earlier snapshot stands for it
    df = snapshots.all_snapshots(wh, SEASON, WEEK)
    with pytest.raises(FileExistsError):
        snapshots.write(df, wh, t1)
    row = df.filter(pl.col("key") == "00-A").row(0, named=True)
    assert row["proj_if_active"] >= row["expected_points"] - 1e-9
    assert row["p10"] <= row["p50"] <= row["p90"] and len(row["quantiles"]) == 21
    assert row["kickoff"] == THU and row["model_version"] and row["league"] == "default"
    b = df.filter(pl.col("key") == "00-B").row(0, named=True)  # 80% to play
    assert b["play_probability"] == 0.8 and b["proj_if_active"] >= b["expected_points"]


def test_scoring_uses_last_snapshot_before_each_kickoff():
    base = {"league": "default", "key": "a", "kickoff": THU}
    snaps = pl.DataFrame([
        base | {"taken_at": datetime(2026, 10, 15, 12, tzinfo=UTC), "v": 1},
        base | {"taken_at": datetime(2026, 10, 15, 23, 30, tzinfo=UTC), "v": 2},  # last before
        base | {"taken_at": datetime(2026, 10, 16, 1, 0, tzinfo=UTC), "v": 3},  # after kickoff
        {"league": "default", "key": "b", "kickoff": None,
         "taken_at": datetime(2026, 10, 15, 12, tzinfo=UTC), "v": 9},  # no kickoff: never
    ])  # fmt: skip
    got = snapshots.scoring_snapshot(snaps)
    assert got["key"].to_list() == ["a"] and got["v"].to_list() == [2]


def _rows():
    return pl.DataFrame({
        "league": ["default"] * 4, "position": ["RB"] * 4, "key": list("abcd"),
        "proj_if_active": [10.0, 8.0, 6.0, 5.0], "expected_points": [10.0, 8.0, 3.0, 5.0],
        "p10": [4.0, 3.0, 2.0, 1.0], "p25": [7.0, 5.0, 4.0, 3.0],
        "p75": [13.0, 11.0, 8.0, 7.0], "p90": [16.0, 14.0, 10.0, 9.0],
        "was_active": [True, True, False, True], "actual": [12.0, 2.0, 0.0, 5.0],
        "last4": [11.0, 9.0, 7.0, None], "season_avg": [9.0, 10.0, 6.0, None],
    })  # fmt: skip


def test_metrics_arithmetic():
    r = metrics.metrics(_rows()).filter(pl.col("scope") == "all")
    get = {(x["source"], x["metric"]): (x["value"], x["n"]) for x in r.iter_rows(named=True)}
    # active: a (10 vs 12), b (8 vs 2), d (5 vs 5)
    assert get[("model", "mae_if_active")] == (pytest.approx((2 + 6 + 0) / 3), 3)
    assert get[("model", "bias")][0] == pytest.approx((-2 + 6 + 0) / 3)
    assert get[("model", "coverage_80")][0] == pytest.approx(2 / 3)  # b at 2 < p10 3
    assert get[("model", "mae_expected")] == (pytest.approx((2 + 6 + 3 + 0) / 4), 4)
    # baselines only where they exist (a, b), model scored on the same players
    assert get[("last4", "mae_if_active")] == (pytest.approx((1 + 7) / 2), 2)
    assert get[("model_vs_last4", "mae_if_active")] == (pytest.approx((2 + 6) / 2), 2)


def test_pooling_ignores_missing_values():
    res = pl.DataFrame({
        "league": ["d"] * 3, "position": ["K"] * 3, "scope": ["all"] * 3,
        "source": ["model"] * 3, "metric": ["mae_if_active"] * 3,
        "value": [4.0, None, 2.0], "n": [10, 5, 30], "week": [1, 2, 3],
    })  # fmt: skip
    p = metrics.pooled(res).row(0, named=True)
    assert p["value"] == pytest.approx((40 + 60) / 40) and p["n"] == 40


def test_report_renders_without_errors():
    res = metrics.metrics(_rows()).with_columns(
        pl.lit(2026).alias("season"), pl.lit(6).alias("week"), pl.lit("abc").alias("model_version")
    )
    md = report.markdown(res, 2026, 6)
    assert "| RB | 3 | 2.67 |" in md and "Model version(s): abc" in md


def test_sensor_fires_only_shortly_before_a_kickoff(wh, monkeypatch):
    from datetime import timedelta

    import dagster as dg

    import dagster_defs as d

    monkeypatch.setattr(d.settings, "warehouse_dir", str(wh))
    monkeypatch.setattr(d.settings, "current_season", SEASON)

    def runs(at):
        monkeypatch.setattr(d, "_now", lambda: at)
        res = d.pre_kickoff_snapshot(dg.build_sensor_context())
        return [r.run_key for r in (res or [])]

    assert runs(THU - timedelta(minutes=30)) == [f"prekick-{THU.isoformat()}"]
    assert runs(THU - timedelta(hours=3)) == []  # too early
    assert runs(THU + timedelta(minutes=5)) == []  # game started: too late to freeze
    assert runs(SUN - timedelta(minutes=10)) == [f"prekick-{SUN.isoformat()}"]


def _rows2():
    """a, d: scored; b: left early (10% of a usual 60%); c: given 50%, sat (availability
    miss); e: given 10%, sat (as projected)."""
    return pl.DataFrame({
        "league": ["default"] * 5, "position": ["RB"] * 5, "key": list("abcde"),
        "proj_if_active": [10.0, 8.0, 6.0, 5.0, 4.0],
        "expected_points": [10.0, 8.0, 3.0, 5.0, 0.4],
        "play_probability": [1.0, 1.0, 0.5, 1.0, 0.1],
        "injury_status": [None, None, "Questionable", None, "Doubtful"],
        "p10": [4.0, 3.0, 2.0, 1.0, 1.0], "p25": [7.0, 5.0, 4.0, 3.0, 2.0],
        "p75": [13.0, 11.0, 8.0, 7.0, 6.0], "p90": [16.0, 14.0, 10.0, 9.0, 8.0],
        "was_active": [True, True, False, True, False], "actual": [12.0, 2.0, 0.0, 5.0, 0.0],
        "snap_pct": [0.7, 0.1, None, 0.5, None], "usual_snap_pct": [0.7, 0.6, 0.5, 0.5, 0.2],
        "last4": [11.0, 9.0, 7.0, 6.0, None], "season_avg": [9.0, 10.0, 6.0, 8.0, None],
        "sleeper": [9.0, 7.0, 0.0, 6.0, 0.5],
    })  # fmt: skip


def test_phase2_buckets_sleeper_and_availability():
    r = metrics.metrics(_rows2())
    a = r.filter(pl.col("scope") == "all")
    get = {(x["source"], x["metric"]): (x["value"], x["n"]) for x in a.iter_rows(named=True)
           if x["position"] == "RB"}  # fmt: skip
    assert get[("misses", "availability_miss")][0] == pytest.approx(0.2)
    assert get[("misses", "early_exit")][0] == pytest.approx(0.2)
    assert get[("misses", "model")][0] == pytest.approx(0.4)
    # Scored on model misses only (a, d); the all-active figure keeps the exit (b).
    assert get[("model", "mae_if_active")] == (pytest.approx(1.0), 2)
    assert get[("model", "mae_if_active_all")] == (pytest.approx(8 / 3), 3)
    assert get[("sleeper", "mae_if_active")] == (pytest.approx(2.0), 2)
    assert get[("model_vs_sleeper", "mae_expected")][0] == pytest.approx(11.4 / 5)
    assert get[("sleeper", "mae_expected")][0] == pytest.approx(9.5 / 5)
    # Close pairs by season average (ab, ac, ad, bd, cd): both miss only b-d.
    assert get[("model_vs_sleeper", "start_sit")] == (pytest.approx(0.8), 5)
    assert get[("sleeper", "start_sit")] == (pytest.approx(0.8), 5)
    assert get[("model_vs_sleeper", "spearman")][0] > 0.5
    q = {x["metric"]: x["value"] for x in r.iter_rows(named=True)
         if x["source"] == "availability:Questionable"}  # fmt: skip
    assert q == {"said": 0.5, "happened": 0.0, "brier": 0.25}


def test_report_shows_sleeper_misses_and_availability():
    res = metrics.metrics(_rows2()).with_columns(
        pl.lit(2026).alias("season"), pl.lit(8).alias("week"), pl.lit("v").alias("model_version")
    )
    md = report.markdown(res, 2026, 8)
    assert "model / Sleeper" in md and "80.0% / 80.0%" in md
    assert "| RB | 5 | 20.0% | 20.0% | 40.0% | 20.0% |" in md
    assert "| Questionable | 1 | 50.0% | 0.0% | 0.250 |" in md
