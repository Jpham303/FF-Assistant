import json
from pathlib import Path

import polars as pl
import pytest

from ffa.scoring import (
    breakdown,
    dst_stat_lines,
    from_sleeper,
    nflverse_standard,
    player_stat_lines,
    score,
    stat_lines,
)
from ffa.scoring.stats import DST_STATS, PLAYER_STATS

FIX = Path(__file__).parent / "fixtures" / "scoring"
LEAGUE_RULES = from_sleeper(json.loads((FIX / "sleeper_scoring.json").read_text()))

PLAYER_COLS = sorted({c for e in PLAYER_STATS.values() for c in e.meta.root_names()})
TEAM_COLS = sorted(
    {c for e in DST_STATS.values() for c in e.meta.root_names()}
    | {"passing_yards", "sack_yards_lost", "rushing_yards"}
)


def raw_player(**stats) -> dict:
    """One nflverse player_stats row: every column the engine reads, zero unless given."""
    row = {c: 0 for c in PLAYER_COLS} | {"fg_made_list": None}
    base = {"player_id": "G-1", "season": 2026, "week": 1, "team": "CIN",
            "opponent_team": "TB", "position": "WR"}  # fmt: skip
    return base | row | stats


def players(*rows: dict) -> pl.DataFrame:
    return player_stat_lines(pl.DataFrame(list(rows), infer_schema_length=None))


def points(df: pl.DataFrame, rules=LEAGUE_RULES) -> list[float]:
    return score(df, rules)["fpts"].to_list()


# ---------------------------------------------------------------- rule translation


def test_league_rules_translated():
    r = LEAGUE_RULES
    assert r.linear["rec"] == 0.5 and r.linear["pass_td"] == 4 and r.linear["pass_int"] == -1
    assert r.linear["fgm_yds"] == 0.1 and r.linear["fgmiss"] == -1
    assert "fgm" not in r.linear  # zero-valued settings are dropped
    assert len([b for b in r.bands if b.stat == "pts_allow"]) == 6  # 21-27 is worth 0
    assert set(r.approximate) == {"def_st_fum_rec", "def_st_ff"}
    assert set(r.unsupported) == {"st_fum_rec", "st_ff"}


def test_unknown_and_pbp_settings_are_reported_not_dropped():
    r = from_sleeper({"rec": 1, "rec_td_40p": 2, "brand_new_setting": 3, "tkl": 1})
    assert r.linear == {"rec": 1.0}
    assert r.unsupported["rec_td_40p"].startswith("needs play-by-play")
    assert r.unsupported["brand_new_setting"] == "unknown Sleeper setting"
    assert "IDP" in r.unsupported["tkl"]


def test_composite_keys_split():
    r = from_sleeper({"fgm_50p": 5, "fgm_60p": 1})
    assert r.linear == {"fgm_50_59": 5.0, "fgm_60p": 6.0}  # stacks, as Sleeper does


# ---------------------------------------------------------------- offense


def test_half_ppr_receiver():
    df = players(raw_player(receptions=7, targets=9, receiving_yards=84, receiving_tds=1,
                            fumbles_lost_total=1))  # fmt: skip
    assert points(df) == [pytest.approx(7 * 0.5 + 8.4 + 6 - 2)]


def test_quarterback_gross_passing_yards_and_rushing():
    df = players(raw_player(position="QB", passing_yards=301, passing_tds=2,
                            passing_interceptions=1, sacks_suffered=3, rushing_yards=22,
                            rushing_2pt_conversions=1))  # fmt: skip
    assert points(df) == [pytest.approx(301 * 0.04 + 8 - 1 + 2.2 + 2)]


def test_matches_nflverse_reference_formula():
    rows = [
        raw_player(receptions=5, receiving_yards=61, receiving_tds=1, fumbles_lost_total=1),
        raw_player(player_id="G-2", position="QB", passing_yards=250, passing_tds=3,
                   passing_interceptions=2, rushing_yards=31, passing_2pt_conversions=1),
        raw_player(player_id="G-3", position="RB", carries=18, rushing_yards=97,
                   rushing_tds=2, receptions=3, receiving_yards=12, special_teams_tds=1),
    ]  # fmt: skip
    assert points(players(*rows), nflverse_standard(1.0)) == pytest.approx(
        [5 + 6.1 + 6 - 2, 10 + 12 - 4 + 3.1 + 2, 9.7 + 12 + 3 + 1.2 + 6]
    )


def test_yardage_bonus_brackets_and_position_terms():
    rules = from_sleeper({"bonus_rec_yd_100": 3, "bonus_rec_yd_200": 5, "bonus_rec_te": 0.5})
    df = players(
        raw_player(player_id="a", receiving_yards=99),
        raw_player(player_id="b", receiving_yards=100),
        raw_player(player_id="c", receiving_yards=200),
        raw_player(player_id="d", position="TE", receptions=6, receiving_yards=40),
    )
    assert points(df, rules) == [0, 3, 5, 3]


# ---------------------------------------------------------------- kicking


def test_kicker_points_per_made_yard_and_misses():
    df = players(raw_player(position="K", fg_made=2, fg_made_list="33;41", fg_made_distance=74,
                            fg_missed=1, fg_blocked=1, pat_made=3, pat_blocked=1))  # fmt: skip
    line = df.row(0, named=True)
    assert line["fgmiss"] == 2 and line["xpmiss"] == 1  # blocked kicks count as misses
    assert line["fgm_yds_over_30"] == 3 + 11
    assert points(df) == [pytest.approx(7.4 - 2 + 3 - 1)]


# ---------------------------------------------------------------- defense


def _team_rows() -> pl.DataFrame:
    def row(team, opp, **kw):
        base = {c: 0 for c in TEAM_COLS}
        return {"season": 2026, "week": 1, "team": team, "opponent_team": opp} | base | kw

    return pl.DataFrame(
        [
            row("CIN", "TB", def_sacks=4, def_interceptions=1, fumble_recovery_opp=1,
                passing_yards=250, sack_yards_lost=12, rushing_yards=100),
            row("TB", "CIN", def_sacks=1, def_tds=1, passing_yards=200, sack_yards_lost=30,
                rushing_yards=80),
            row("LA", "SEA", def_safeties=1),
            row("SEA", "LA"),
        ]
    )  # fmt: skip


def _schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2026, 2026], "week": [1, 1],
            "home_team": ["CIN", "SEA"], "away_team": ["TB", "LA"],
            "home_score": [27, 0], "away_score": [13, 35],
        }
    )  # fmt: skip


def test_defense_stat_lines():
    d = {r["key"]: r for r in dst_stat_lines(_team_rows(), _schedule()).iter_rows(named=True)}
    assert set(d) == {"DST_CIN", "DST_TB", "DST_LA", "DST_SEA"}
    assert d["DST_CIN"]["pts_allow"] == 13 and d["DST_TB"]["pts_allow"] == 27
    assert d["DST_CIN"]["yds_allow"] == 200 - 30 + 80  # TB's net offensive yards
    assert d["DST_LA"]["pts_allow"] == 0 and d["DST_SEA"]["pts_allow"] == 35


def test_defense_points():
    lines = dst_stat_lines(_team_rows(), _schedule())
    got = dict(zip(lines["key"], points(lines), strict=True))
    assert got["DST_CIN"] == 4 + 2 + 2 + 4  # sacks, INT, fumble recovery, 7-13 allowed
    assert got["DST_TB"] == 1 + 6 + 0  # 21-27 allowed is worth 0 in this league
    assert got["DST_LA"] == 2 + 10  # safety + shutout
    assert got["DST_SEA"] == -4


@pytest.mark.parametrize(
    ("allowed", "pts"),
    [(0, 10), (1, 7), (6, 7), (7, 4), (13, 4), (14, 1), (20, 1), (21, 0), (27, 0),
     (28, -1), (34, -1), (35, -4), (52, -4)],
)  # fmt: skip
def test_points_allowed_band_edges(allowed, pts):
    df = pl.DataFrame({"position": ["DEF"], "pts_allow": [float(allowed)]})
    assert points(df) == [pts]


def test_offensive_players_never_get_defense_bands():
    """A receiver's pts_allow is 0; he must not receive the shutout bonus."""
    lines = stat_lines(
        pl.DataFrame([raw_player(receptions=1)], infer_schema_length=None),
        _team_rows(),
        _schedule(),
    )
    wr = score(lines, LEAGUE_RULES).filter(pl.col("key") == "G-1")
    assert wr["fpts"].to_list() == [0.5]


def test_breakdown_sums_to_score():
    lines = stat_lines(
        pl.DataFrame(
            [raw_player(receptions=7, receiving_yards=84, receiving_tds=1),
             raw_player(player_id="K1", position="K", fg_made=1, fg_made_distance=45,
                        fg_made_list="45", pat_made=2)],
            infer_schema_length=None,
        ),
        _team_rows(),
        _schedule(),
    )  # fmt: skip
    totals = score(lines, LEAGUE_RULES).select("key", "fpts")
    parts = breakdown(lines, LEAGUE_RULES).group_by("key").agg(pl.col("points").sum())
    joined = totals.join(parts, on="key", how="left").with_columns(pl.col("points").fill_null(0))
    assert (joined["fpts"] - joined["points"]).abs().max() < 1e-9
    k = breakdown(lines.filter(pl.col("key") == "K1"), LEAGUE_RULES)
    assert dict(zip(k["term"], k["points"], strict=True)) == {"fgm_yds": 4.5, "xpm": 2.0}
