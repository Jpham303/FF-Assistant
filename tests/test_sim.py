import numpy as np
import polars as pl
import pytest

from ffa.model.inputs import WeekInputs
from ffa.model.sim import simulate
from ffa.model.sim_params import SimParams

N = 3000


def params() -> SimParams:
    return SimParams(
        p_play={"Questionable|any|role": 0.75},
        fg_distance={"2023": {"30": 0.5, "45": 0.3, "55": 0.2},
                     "2025": {"40": 0.5, "52": 0.5}},
        fg_make={"2023": [6.0, -0.1], "2025": [6.5, -0.1]},
        dst={"def_td_per_takeaway": 0.09, "def_st_td_per_game": 0.04, "blk_kick_per_game": 0.07,
             "safe_per_game": 0.03, "def_st_fum_rec_per_game": 0.06, "def_st_ff_per_game": 0.06,
             "ff_per_fum_rec": 1.6},
    )  # fmt: skip


def week() -> WeekInputs:
    teams = pl.DataFrame(
        {
            "game_id": ["G", "G"], "team": ["AAA", "BBB"], "opponent": ["BBB", "AAA"],
            "is_home": [True, False], "spread_for": [3.0, -3.0], "total_line": [47.0, 47.0],
            "implied": [25.0, 22.0], "opp_implied": [22.0, 25.0], "mu_plays": [64.0, 62.0],
            "sd_plays": [8.0, 8.0], "p_pass": [0.6, 0.58], "mu_off_td": [2.8, 2.4],
            "p_pass_td": [0.62, 0.6], "mu_fg_att": [1.9, 1.8], "pace_s": [64.0, 62.0],
            "proe_s": [0.0, 0.0],
            **{c: [1.0, 1.0] for c in ("off_sack_mult", "off_int_mult", "off_fum_mult",
                                       "def_sack_mult", "def_int_mult", "def_fum_mult",
                                       "opp_def_sack_mult", "opp_def_int_mult",
                                       "opp_def_fum_mult")},
        }
    )  # fmt: skip

    def p(key, team, pos, **kw):
        opp = "BBB" if team == "AAA" else "AAA"
        base = {
            "key": key, "name": key, "team": team, "opponent": opp, "game_id": "G",
            "position": pos, "p_play": 1.0, "tgt_share_s": 0.0, "car_share_s": 0.0,
            "db_share_s": 0.0, "rec_td_share_s": 0.0, "rush_td_share_s": 0.0,
            "catch_rate_s": 0.65, "ypr_s": 11.0, "ypc_s": 4.3, "int_rate_s": 0.025,
            "scramble_rate_s": 0.03, "fum_rate_s": 0.01, "m_tgt": 1.0, "m_catch": 1.0,
            "m_ypr": 1.0, "m_ypc": 1.0,
        }  # fmt: skip
        return base | kw

    players = pl.DataFrame(
        [
            p("A_QB1", "AAA", "QB", db_share_s=0.95, car_share_s=0.08, rush_td_share_s=0.1),
            p("A_QB2", "AAA", "QB", db_share_s=0.05),
            p("A_WR1", "AAA", "WR", tgt_share_s=0.30, rec_td_share_s=0.35),
            p("A_WR2", "AAA", "WR", tgt_share_s=0.20, rec_td_share_s=0.2, p_play=0.0),
            p(
                "A_RB1",
                "AAA",
                "RB",
                tgt_share_s=0.10,
                car_share_s=0.65,
                rec_td_share_s=0.05,
                rush_td_share_s=0.7,
            ),
            p("A_K", "AAA", "K"),
            p("B_QB1", "BBB", "QB", db_share_s=1.0),
            p("B_WR1", "BBB", "WR", tgt_share_s=0.4, rec_td_share_s=0.4),
            p("B_RB1", "BBB", "RB", car_share_s=0.8, rush_td_share_s=0.8),
        ]
    )
    return WeekInputs(2026, 6, teams, players)


@pytest.fixture(scope="module")
def lines() -> pl.DataFrame:
    return simulate(week(), params(), n=N, seed=7)


def wide(lines: pl.DataFrame, stat: str) -> dict[str, np.ndarray]:
    return {k: g.sort("sim")[stat].to_numpy() for (k,), g in lines.group_by("key")}


def test_inactive_player_produces_nothing(lines):
    wr2 = lines.filter(pl.col("key") == "A_WR2")
    assert (
        wr2.select(pl.col("rec_tgt", "rec", "rec_yd", "rec_td").abs().sum()).sum_horizontal().item()
        == 0
    )


def test_the_starter_plays_and_the_backup_only_finishes_some_games(lines):
    att = wide(lines, "pass_att")
    team = att["A_QB1"] + att["A_QB2"]
    assert (att["A_QB1"] > 0).mean() > 0.99  # an active starter is never benched outright
    # In-game exits, from the fitted distribution: the starter took under 75% of his
    # team's dropbacks in 5.6% of 2016-2023 games he played.
    assert (att["A_QB1"] / team < 0.75).mean() == pytest.approx(0.056, abs=0.02)
    assert (att["A_QB1"] / team).mean() == pytest.approx(0.96, abs=0.02)


def test_inflated_backup_share_does_not_bench_the_starter():
    """Maye/Cousins bug: a backup's shrunk dropback share (0.6) was used as his odds of
    starting, so an active starter scored 0 in a large share of simulations."""
    w = week()
    players = w.players.with_columns(
        pl.when(pl.col("key") == "A_QB2").then(0.6).otherwise(pl.col("db_share_s"))
        .alias("db_share_s")
    )  # fmt: skip
    out = simulate(WeekInputs(w.season, w.week, w.teams, players), params(), n=N, seed=3)
    att = wide(out, "pass_att")
    assert (att["A_QB1"] == 0).mean() < 0.01
    assert np.quantile(att["A_QB1"], 0.1) > 20


def test_backup_starts_when_the_starter_is_out():
    w = week()
    players = w.players.with_columns(
        pl.when(pl.col("key") == "A_QB1").then(0.0).otherwise(pl.col("p_play")).alias("p_play")
    )
    out = simulate(WeekInputs(w.season, w.week, w.teams, players), params(), n=500, seed=5)
    att = wide(out, "pass_att")
    assert (att["A_QB1"] == 0).all() and (att["A_QB2"] > 0).all()
    td = wide(out, "pass_td")
    assert td["A_QB2"].mean() > 1  # the team's passing TDs go to whoever played


def test_passing_equals_receiving_plus_other_pool(lines):
    py = wide(lines, "pass_yd")
    ry = wide(lines, "rec_yd")
    qb_yards = py["A_QB1"] + py["A_QB2"]
    receivers = ry["A_WR1"] + ry["A_WR2"] + ry["A_RB1"]
    assert (qb_yards >= receivers - 1e-3).all()  # the rest went to unmodeled receivers
    assert (qb_yards - receivers).mean() < 0.15 * qb_yards.mean()


def test_defense_is_built_from_the_opponents_game(lines):
    get = lambda s: wide(lines, s)  # noqa: E731
    a_tds = (
        get("pass_td")["A_QB1"]
        + get("pass_td")["A_QB2"]
        + sum(get("rush_td")[k] for k in ("A_QB1", "A_QB2", "A_WR1", "A_WR2", "A_RB1"))
    )
    expected_allowed = 6 * a_tds + get("xpm")["A_K"] + 3 * get("fgm")["A_K"]
    np.testing.assert_allclose(get("pts_allow")["DST_BBB"], expected_allowed)
    np.testing.assert_allclose(
        get("int")["DST_BBB"], get("pass_int")["A_QB1"] + get("pass_int")["A_QB2"]
    )
    fum = sum(get("fum_lost")[k] for k in ("A_QB1", "A_QB2", "A_WR1", "A_WR2", "A_RB1"))
    np.testing.assert_allclose(get("fum_rec")["DST_BBB"], fum)


def test_touchdowns_need_a_touch(lines):
    rec_td, tgt = wide(lines, "rec_td")["A_WR1"], wide(lines, "rec_tgt")["A_WR1"]
    assert not ((rec_td > 0) & (tgt == 0)).any()


def test_injured_teammates_share_goes_to_active_players(lines):
    # WR2 (0.20) is out. Half of it stays with the WRs (INHERIT_KEEP = 0.5): WR1 0.30 +
    # 0.10 = 0.40; the rest is spread by share over WR1 (0.40) and RB1 (0.10) -> 4 : 1.
    tgt = wide(lines, "rec_tgt")
    assert tgt["A_WR1"].mean() / tgt["A_RB1"].mean() == pytest.approx(4.0, rel=0.1)
    team_targets = 64 * 0.6 * 0.85
    assert tgt["A_WR1"].mean() == pytest.approx(team_targets * 0.98 * 0.8, rel=0.12)


def test_kicking_uses_latest_completed_season(lines):
    d, m = params().kicking(2026)
    assert set(d) == {"40", "52"} and m == [6.5, -0.1]
    assert params().kicking(2024)[0] == params().fg_distance["2023"]
    k = lines.filter(pl.col("key") == "A_K")
    assert (k["fgm_yds"] >= 40 * k["fgm"] - 1e-6).all()  # every make is 40 or 52 yards


def test_reproducible():
    a = simulate(week(), params(), n=200, seed=3)
    b = simulate(week(), params(), n=200, seed=3)
    assert a.equals(b)


def test_play_probability():
    sp = params()
    assert sp.play_probability(None, None, True) == 1.0
    assert sp.play_probability("Out", None, True) == 0.0
    assert sp.play_probability("Questionable", "Full Participation in Practice", True) == 0.75
