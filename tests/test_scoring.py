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
from ffa.scoring.defense import scoring_plays
from ffa.scoring.stats import PLAYER_STATS

FIX = Path(__file__).parent / "fixtures" / "scoring"
LEAGUE_RULES = from_sleeper(json.loads((FIX / "sleeper_scoring.json").read_text()))

PLAYER_COLS = sorted({c for e in PLAYER_STATS.values() for c in e.meta.root_names()})


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
    assert r.linear["def_st_fum_rec"] == 1 and r.linear["def_st_ff"] == 1
    assert r.approximate == {}
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


# ---------------------------------------------------------------- defense (play-by-play)

PBP_DEFAULTS = {
    "season": 2026, "week": 1, "home_team": "CIN", "away_team": "TB", "game_id": "G1",
    "posteam": None, "defteam": None, "play_type": "pass", "sack": 0, "interception": 0,
    "fumble_forced": 0, "fumble_lost": 0, "forced_fumble_player_1_team": None,
    "fumble_recovery_1_team": None, "safety": 0, "touchdown": 0, "return_touchdown": 0,
    "td_team": None, "punt_blocked": 0, "field_goal_result": None,
    "extra_point_result": None, "two_point_conv_result": None,
    "defensive_two_point_conv": 0, "two_point_attempt": 0, "qb_hit": 0,
    "tackled_for_loss": 0, "pass_defense_1_player_id": None, "return_yards": 0,
    "yards_gained": 0, "posteam_score": 0, "posteam_score_post": 0,
}  # fmt: skip


def game() -> pl.DataFrame:
    """CIN (home) 14, TB 11. Each play is a case the platforms score in a specific way."""
    plays = [
        # Strip-sack with no player credited for the forced fumble: still a sack.
        dict(posteam="TB", defteam="CIN", sack=1, yards_gained=-8, fumble_lost=1,
             fumble_recovery_1_team="CIN"),
        # Pick-six by CIN, then CIN's extra point: neither counts against TB's defense.
        dict(posteam="TB", defteam="CIN", interception=1, return_touchdown=1, touchdown=1,
             td_team="CIN"),
        dict(posteam="CIN", defteam="TB", play_type="extra_point", extra_point_result="good"),
        # TB punts; CIN muffs; TB recovers: special-teams recovery and forced fumble.
        dict(posteam="TB", defteam="CIN", play_type="punt", fumble_forced=1, fumble_lost=1,
             forced_fumble_player_1_team="TB", fumble_recovery_1_team="TB"),
        dict(posteam="TB", defteam="CIN", play_type="field_goal", field_goal_result="made"),
        dict(posteam="CIN", defteam="TB", touchdown=1, td_team="CIN", yards_gained=40),
        dict(posteam="CIN", defteam="TB", play_type="extra_point", extra_point_result="good"),
        # TB kickoff-return TD (TB is posteam on a kickoff it receives); CIN blocks the PAT.
        dict(posteam="TB", defteam="CIN", play_type="kickoff", touchdown=1,
             return_touchdown=1, td_team="TB", return_yards=100),
        dict(posteam="TB", defteam="CIN", play_type="extra_point", extra_point_result="blocked"),
        # TB punts, CIN's returner is tackled in his end zone: the kicking team scores.
        dict(posteam="TB", defteam="CIN", play_type="punt", safety=1, posteam_score=9,
             posteam_score_post=11),
    ]  # fmt: skip
    rows = [PBP_DEFAULTS | {"play_id": float(i)} | p for i, p in enumerate(plays, 1)]
    return pl.DataFrame(rows, infer_schema_length=None)


def schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {"game_id": ["G1"], "season": [2026], "week": [1], "home_team": ["CIN"],
         "away_team": ["TB"], "home_score": [14], "away_score": [11]}
    )  # fmt: skip


def defense() -> dict[str, dict]:
    return {r["key"]: r for r in dst_stat_lines(game(), schedule()).iter_rows(named=True)}


def test_scoring_events_add_up_to_the_final_score():
    totals = dict(scoring_plays(game()).group_by("scorer").agg(pl.col("points").sum()).iter_rows())
    assert totals == {"CIN": 14, "TB": 11}


def test_defense_counting_stats():
    cin, tb = defense()["DST_CIN"], defense()["DST_TB"]
    assert (cin["sack"], cin["sack_yd"], cin["int"], cin["ff"]) == (1, 8, 1, 0)
    assert (cin["fum_rec"], cin["def_td"], cin["blk_kick"]) == (1, 1, 1)
    assert (tb["def_st_fum_rec"], tb["def_st_ff"], tb["fum_rec"]) == (1, 1, 0)
    assert (tb["def_st_td"], tb["safe"], cin["safe"]) == (1, 1, 0)
    assert tb["def_kr_yd"] == 100 and tb["yds_allow"] == 40


def test_points_allowed_excludes_return_scores():
    d = defense()
    # TB scored 11, but 6 came on a kick return and 2 on a safety, neither against CIN's D.
    assert d["DST_CIN"]["pts_allow"] == 3
    # CIN scored 14; the pick-six and its extra point were not allowed by TB's defense.
    assert d["DST_TB"]["pts_allow"] == 7


def test_defense_points():
    lines = dst_stat_lines(game(), schedule())
    got = dict(zip(lines["key"], points(lines), strict=True))
    assert got["DST_CIN"] == 1 + 2 + 2 + 6 + 2 + 7  # sack, INT, rec, TD, block, 1-6 allowed
    assert got["DST_TB"] == 1 + 1 + 6 + 2 + 4  # ST rec, ST FF, return TD, safety, 7-13


def test_relocated_team_codes_are_normalized():
    old = game().with_columns(
        pl.col(c).replace({"TB": "STL"}) for c in ("away_team", "posteam", "defteam", "td_team")
    )
    sched = schedule().with_columns(pl.lit("STL").alias("away_team"))
    keys = set(dst_stat_lines(old, sched)["key"])
    assert keys == {"DST_CIN", "DST_LA"}


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
        pl.DataFrame([raw_player(receptions=1)], infer_schema_length=None), game(), schedule()
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
        game(),
        schedule(),
    )  # fmt: skip
    totals = score(lines, LEAGUE_RULES).select("key", "fpts")
    parts = breakdown(lines, LEAGUE_RULES).group_by("key").agg(pl.col("points").sum())
    joined = totals.join(parts, on="key", how="left").with_columns(pl.col("points").fill_null(0))
    assert (joined["fpts"] - joined["points"]).abs().max() < 1e-9
    k = breakdown(lines.filter(pl.col("key") == "K1"), LEAGUE_RULES)
    assert dict(zip(k["term"], k["points"], strict=True)) == {"fgm_yds": 4.5, "xpm": 2.0}
