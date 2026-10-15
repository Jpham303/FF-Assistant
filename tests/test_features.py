import polars as pl
import pytest

from ffa.features import player_game_log, team_game_log
from ffa.features.games import PBP_COLUMNS, _player_opportunity, _snaps, _vegas
from ffa.scoring.stats import PLAYER_STATS

PLAYER_COLS = sorted({c for e in PLAYER_STATS.values() for c in e.meta.root_names()})


def schedules() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["G1", "G2"], "season": [2026, 2026], "week": [1, 2],
            "game_type": ["REG", "REG"], "home_team": ["CIN", "TB"],
            "away_team": ["STL", "CIN"],  # STL: an old team code, must become LA
            "home_score": [27, None], "away_score": [20, None],
            "spread_line": [3.5, -2.0], "total_line": [47.5, 44.0], "roof": ["outdoors"] * 2,
        }
    )  # fmt: skip


def play(**kw) -> dict:
    base = {c: None for c in PBP_COLUMNS} | {
        "game_id": "G1", "season": 2026, "week": 1, "season_type": "REG",
        "posteam": "CIN", "defteam": "LA", "play_type": "pass", "qb_dropback": 1,
        "qb_scramble": 0, "pass_attempt": 1, "rush_attempt": 0, "sack": 0,
        "interception": 0, "fumble_lost": 0, "two_point_attempt": 0, "yardline_100": 50,
        "air_yards": 0, "epa": 0.0, "success": 0, "xpass": 0.6, "pass_touchdown": 0,
        "rush_touchdown": 0, "field_goal_attempt": 0, "extra_point_attempt": 0,
    }  # fmt: skip
    return base | kw


def pbp() -> pl.DataFrame:
    rows = [
        play(receiver_player_id="WR1", passer_player_id="QB1", air_yards=12, yardline_100=15),
        play(receiver_player_id="WR1", passer_player_id="QB1", air_yards=20, yardline_100=18,
             pass_touchdown=1),
        play(receiver_player_id="TE1", passer_player_id="QB1", air_yards=5),
        play(passer_player_id="QB1", sack=1),
        play(play_type="run", qb_dropback=0, pass_attempt=0, rush_attempt=1,
             rusher_player_id="RB1", yardline_100=3, rush_touchdown=1),
        play(play_type="run", qb_dropback=0, pass_attempt=0, rush_attempt=1,
             rusher_player_id="RB1", yardline_100=40),
        play(play_type="run", qb_scramble=1, pass_attempt=0, rush_attempt=1,
             rusher_player_id="QB1"),
        # Excluded: a two-point try and a kneel-down are not fantasy opportunities.
        play(play_type="run", qb_dropback=0, pass_attempt=0, rush_attempt=1,
             rusher_player_id="RB1", two_point_attempt=1, yardline_100=2),
        play(play_type="qb_kneel", qb_dropback=0, pass_attempt=0, rusher_player_id="QB1"),
        play(posteam="LA", defteam="CIN", play_type="run", qb_dropback=0, pass_attempt=0,
             rush_attempt=1, rusher_player_id="LARB"),
    ]  # fmt: skip
    return pl.DataFrame(rows, infer_schema_length=None)


def test_vegas_lines_from_each_teams_side():
    v = {
        r["team"]: r
        for r in _vegas(schedules()).filter(pl.col("game_id") == "G1").iter_rows(named=True)
    }
    assert set(v) == {"CIN", "LA"}
    assert v["CIN"]["spread_for"] == 3.5 and v["LA"]["spread_for"] == -3.5
    assert v["CIN"]["implied"] == pytest.approx(25.5) and v["LA"]["implied"] == pytest.approx(22.0)
    assert v["CIN"]["opp_implied"] == v["LA"]["implied"]


def test_team_game_log():
    t = {
        r["team"]: r
        for r in team_game_log(pbp(), schedules())
        .filter(pl.col("game_id") == "G1")
        .iter_rows(named=True)
    }
    cin = t["CIN"]
    assert cin["plays"] == 7  # kneel and two-point try excluded
    assert (cin["dropbacks"], cin["designed_runs"], cin["scrambles"]) == (5, 2, 1)
    assert (cin["pass_att"], cin["sacks_taken"], cin["targets"]) == (3, 1, 3)
    assert (cin["pass_td"], cin["rush_td"], cin["air_yards"]) == (1, 1, 37)
    assert t["LA"]["opp_plays"] == 7 and cin["opp_plays"] == 1  # what each defense faced
    future = team_game_log(pbp(), schedules()).filter(pl.col("game_id") == "G2")
    assert future["plays"].is_null().all() and future["implied"].is_not_null().all()


def test_player_opportunity():
    o = {r["key"]: r for r in _player_opportunity(pbp()).iter_rows(named=True)}
    # Only the 20-yard throw from the 18 reaches the end zone.
    assert (o["WR1"]["targets"], o["WR1"]["rz_targets"], o["WR1"]["ez_targets"]) == (2, 2, 1)
    assert o["WR1"]["air_yards"] == 32
    assert (o["RB1"]["designed_carries"], o["RB1"]["rz_carries"], o["RB1"]["gl_carries"]) == (
        2,
        1,
        1,
    )
    assert (o["QB1"]["scrambles"], o["QB1"]["designed_carries"], o["QB1"]["dropbacks"]) == (1, 0, 5)


def test_snaps_keyed_through_crosswalk():
    snaps = pl.DataFrame(
        {"game_id": ["G1", "G1"], "pfr_player_id": ["SmitJo00", "Nobody00"],
         "offense_snaps": [50, 3], "offense_pct": [0.8, 0.05]}
    )  # fmt: skip
    ids = pl.DataFrame({"gsis_id": ["00-1"], "pfr_id": ["SmitJo00"]})
    out = _snaps(snaps, ids)
    assert out.rows() == [("G1", "00-1", 50.0, 0.8)]


def test_player_game_log_joins_everything():
    def raw(key, pos, **kw):
        row = {c: 0 for c in PLAYER_COLS} | {"fg_made_list": None}
        base = {"player_id": key, "season": 2026, "week": 1, "team": "CIN",
                "opponent_team": "STL", "position": pos}  # fmt: skip
        return base | row | kw

    stats = pl.DataFrame(
        [raw("WR1", "WR", targets=2, receptions=2, receiving_yards=30, receiving_tds=1),
         raw("RB1", "FB", carries=2, rushing_yards=12, rushing_tds=1)],
        infer_schema_length=None,
    )  # fmt: skip
    teams = team_game_log(pbp(), schedules())
    snaps = pl.DataFrame(
        {"game_id": ["G1"], "pfr_player_id": ["W1"], "offense_snaps": [55], "offense_pct": [0.9]}
    )
    ids = pl.DataFrame({"gsis_id": ["WR1"], "pfr_id": ["W1"]})
    inj = pl.DataFrame(
        {"season": [2026], "week": [1], "gsis_id": ["WR1"], "report_status": ["Questionable"],
         "report_primary_injury": ["Hamstring"]}
    )  # fmt: skip
    out = {
        r["key"]: r
        for r in player_game_log(pbp(), stats, snaps, ids, inj, teams).iter_rows(named=True)
    }
    wr, rb = out["WR1"], out["RB1"]
    assert wr["team"] == "CIN" and wr["opponent"] == "LA"
    assert (wr["targets"], wr["team_targets"], wr["rec"], wr["rec_td"]) == (2, 3, 2, 1)
    assert (wr["snaps"], wr["snap_pct"]) == (55, 0.9)
    assert (wr["injury_status"], wr["injury"]) == ("Questionable", "Hamstring")
    assert wr["implied"] == pytest.approx(25.5)
    assert rb["position"] == "RB" and rb["snaps"] == 0 and rb["injury_status"] is None
