"""If-active projection spec: acceptance criteria."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from ffa.model.sim import simulate
from ffa.scoring import score
from ffa.scoring.rules import default_rules
from ffa.service import ifactive, startsit
from ffa.service.projections import forced_active
from tests.test_sim import params, week

RULES = default_rules()


def _week(**p_play):
    w = week()
    players = w.players
    # a backup RB behind A_RB1, to see the starter's absence reach a teammate
    backup = players.filter(pl.col("key") == "A_RB1").with_columns(
        pl.lit("A_RB2").alias("key"), pl.lit("A_RB2").alias("name"),
        pl.lit(0.15).alias("car_share_s"), pl.lit(0.1).alias("rush_td_share_s"),
    )  # fmt: skip
    players = pl.concat([players, backup])
    for k, p in p_play.items():
        players = players.with_columns(
            pl.when(pl.col("key") == k).then(p).otherwise(pl.col("p_play")).alias("p_play")
        )
    return type(w)(w.season, w.week, w.teams, players)


def _summary(inp, n=4000, seed=3, forced=None):
    sims = simulate(inp, params(), n=n, seed=seed)
    sims = sims.with_columns(ifactive.active_flag(sims).alias("active"))
    pts = score(sims, RULES).select("key", "sim", "fpts", "active")
    play = dict(inp.players.select("key", "p_play").iter_rows())
    fp = None
    if forced is not None and forced.height:
        fp = score(forced, RULES).select(pl.col("forced_key").alias("key"), "sim", "fpts")
    return {r["key"]: r for r in ifactive.summarize(pts, play, fp).iter_rows(named=True)}, pts


def test_identity_expected_equals_play_times_if_active():
    s, _ = _summary(_week(A_RB1=0.73))
    for r in s.values():
        if r["proj_if_active"] is not None:
            assert r["expected_points"] == pytest.approx(
                r["play_probability"] * r["proj_if_active"], abs=0.01)  # fmt: skip


def test_healthy_player_if_active_equals_expected_and_ranges_match():
    s, pts = _summary(_week())
    r = s["B_WR1"]
    f = pts.filter(pl.col("key") == "B_WR1")["fpts"]
    assert r["play_probability"] == 1.0 and r["active_sims"] == 4000
    assert r["proj_if_active"] == pytest.approx(r["expected_points"], abs=1e-9)
    assert r["p10_if_active"] == pytest.approx(f.quantile(0.1, "linear"))
    assert r["p90_if_active"] == pytest.approx(f.quantile(0.9, "linear"))


def test_not_playing_has_null_if_active_and_zero_expected():
    s, _ = _summary(_week())  # A_WR2 is out (p_play 0) in the fixture
    r = s["A_WR2"]
    assert r["proj_if_active"] is None and r["p90_if_active"] is None
    assert r["expected_points"] == 0 and r["play_probability"] == 0


def test_low_play_chance_uses_a_forced_active_batch_and_is_stable():
    inp = _week(A_RB1=0.05)
    models = SimpleNamespace(sim=params())
    p50 = []
    for seed in (11, 12):
        forced = forced_active(inp, models, n=1000, seed=seed)
        assert set(forced["forced_key"].unique()) == {"A_RB1"}
        s, _ = _summary(inp, n=2000, seed=seed, forced=forced)
        r = s["A_RB1"]
        assert r["forced_active_batch"] and r["active_sims"] >= ifactive.MIN_ACTIVE
        assert r["expected_points"] == pytest.approx(0.05 * r["proj_if_active"])
        p50.append(r["median_if_active"])
    # Seed-to-seed sd of the median is ~0.24 at this size (12 seeds), so the difference of
    # two has sd ~0.33: allow 3 sd.
    assert abs(p50[0] - p50[1]) < 1.0


def test_teammate_projection_rises_when_the_starter_sits():
    """Week 7 case (Montgomery out, Marks up), on the fixture's RB pair."""
    _, pts = _summary(_week(A_RB1=0.5), n=6000)
    starter = pts.filter(pl.col("key") == "A_RB1").select("sim", pl.col("active").alias("on"))
    backup = pts.filter(pl.col("key") == "A_RB2").join(starter, on="sim")
    sits = backup.filter(~pl.col("on"))["fpts"].mean()
    plays = backup.filter(pl.col("on"))["fpts"].mean()
    assert sits > plays + 1.0


def _risky_vs_safe(n=1000, p_risky=0.4):
    rng = np.random.default_rng(0)
    on = rng.random(n) < p_risky
    sims = pl.DataFrame({
        "key": ["R"] * n + ["S"] * n, "name": ["Risky"] * n + ["Safe"] * n,
        "team": ["AAA"] * n + ["BBB"] * n, "position": ["WR"] * 2 * n,
        "sim": list(range(n)) * 2,
        "rec_yd": np.concatenate([np.where(on, 150.0, 0.0), np.full(n, 100.0)]),
        "active": np.concatenate([on, np.ones(n, bool)]),
    })  # fmt: skip
    inputs = pl.DataFrame({"key": ["R", "S"], "team": ["AAA", "BBB"], "p_play": [p_risky, 1.0],
                           "injury_status": ["Questionable", None]})  # fmt: skip
    teams = pl.DataFrame({"team": ["AAA", "BBB"], "opponent": ["BBB", "AAA"]})
    return startsit.Week(2026, 7, sims, inputs, teams)


def test_start_sit_reports_talent_alone_and_explains_disagreement():
    c = startsit.compare(_risky_vs_safe(), RULES)
    pw = {(x["a"], x["b"]): x for x in c.pairwise}
    assert pw[("R", "S")]["p_a_beats_b"] < 0.5  # sitting risk counted: Safe wins
    assert pw[("R", "S")]["p_a_beats_b_if_both_active"] == 1.0  # talent alone: Risky
    assert c.recommendation == "S"
    note = next(n for n in c.notes if "better play if both are active" in n)
    assert note.startswith("Risky is the better play if both are active (100%), but Safe wins")
    assert "Risky's 60% chance of sitting" in note


def test_no_note_when_both_measures_agree():
    c = startsit.compare(_risky_vs_safe(p_risky=0.95), RULES)
    assert not any("better play if both are active" in n for n in c.notes)


def test_deprecated_mean_equals_expected_points():
    v = startsit.player_views(_risky_vs_safe(), RULES)
    assert v["R"].mean == v["R"].expected_points
    assert v["R"].proj_if_active == pytest.approx(15.0)
    assert v["R"].expected_points == pytest.approx(0.4 * 15.0)
