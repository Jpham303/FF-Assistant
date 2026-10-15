import polars as pl

from ffa.ingest.ids import Resolver
from ffa.leagues.base import Platform, WeekResult
from ffa.scoring import from_sleeper
from ffa.scoring.check import awarded_frame, compare


def _resolver() -> Resolver:
    xw = pl.DataFrame(
        {
            "gsis_id": ["G-A", "G-B", "G-C"],
            "sleeper_id": ["1", "2", "3"],
            "yahoo_id": [None, None, None],
            "name_key": ["a", "b", "c"],
            "position": ["WR", "WR", "WR"],
            "team": ["CIN", "CIN", "CIN"],
            "last_season": [2026, 2026, 2026],
        },
        schema_overrides={"yahoo_id": pl.Utf8},
    )
    return Resolver(xw)


def test_compare_statuses():
    rules = from_sleeper({"rec": 0.5, "rec_yd": 0.1})
    lines = pl.DataFrame(
        {
            "key": ["G-A", "G-B"],
            "week": [1, 1],
            "position": ["WR", "WR"],
            "rec": [4.0, 6.0],
            "rec_yd": [50.0, 70.0],
        }
    )
    weeks = {
        1: [
            WeekResult(
                team_id="1",
                matchup_id=1,
                points=0,
                starters=[],
                # A matches (7.0); B is off by one (10.0 vs 11.0); C had no stat line
                # (inactive) and scored 0; 999 is a player we can't map.
                player_points={"1": 7.0, "2": 11.0, "3": 0.0, "999": 4.2},
            )
        ]
    }
    out, rep = compare(awarded_frame(weeks), lines, rules, _resolver(), Platform.SLEEPER)
    status = dict(zip(out["platform_id"], out["status"], strict=True))
    assert status == {"1": "match", "2": "mismatch", "3": "match", "999": "unresolved"}
    assert (rep.compared, rep.matched, rep.mismatched, rep.unresolved) == (3, 2, 1, 1)
    assert out.filter(pl.col("platform_id") == "2")["diff"].item() == -1.0
