"""Sleeper benchmark capture: parsing, league scoring, keys, write-once and dedupe."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from ffa.tracking import benchmarks as bm

NOW = datetime(2026, 10, 25, 16, 15, tzinfo=UTC)
KO = datetime(2026, 10, 25, 17, 0, tzinfo=UTC)
HALF = {"pass_yd": 0.04, "pass_td": 4, "rush_yd": 0.1, "rec": 0.5, "rec_yd": 0.1,
        "rec_td": 6, "sack": 1, "int": 2, "pts_allow_7_13": 4}  # fmt: skip
PPR = HALF | {"rec": 1.0}
RAW = [
    {"player_id": "4984", "team": "BUF", "player": {"position": "QB", "first_name": "Josh",
     "last_name": "Allen"}, "stats": {"pass_yd": 250.0, "pass_td": 2.0, "rush_yd": 30.0,
     "pts_ppr": 23.0, "adp_dd_ppr": 20}},
    {"player_id": "6794", "team": "LAR", "player": {"position": "WR"},
     "stats": {"rec": 6.0, "rec_yd": 80.0, "rec_td": 0.5}},
    {"player_id": "LAR", "team": "LAR", "player": {"position": "DEF"},
     "stats": {"sack": 2.5, "int": 1.0, "pts_allow_7_13": 0.25}},
    {"player_id": "9999", "team": "NYJ", "player": {"position": "RB"}, "stats": {"rush_yd": 50.0}},
    {"player_id": "4984", "team": "BUF", "player": {"position": "QB"}, "stats": {}},  # ADP-only row
]  # fmt: skip
XW = pl.DataFrame({"sleeper_id": ["4984", "6794"], "gsis_id": ["00-ALLEN", "00-PUKA"]})
KICK = pl.DataFrame({"team": ["BUF", "LA"], "kickoff": [KO, KO]},
                    schema={"team": pl.Utf8, "kickoff": pl.Datetime("us", "UTC")})  # fmt: skip
LEAGUES = {"default": HALF, "sleeper:1": PPR}


def test_scores_each_league_like_sleeper_and_keys_like_ours():
    df = bm.build(RAW, XW, KICK, LEAGUES, NOW)
    p = {(r["league"], r["key"]): r for r in df.iter_rows(named=True)}
    assert p[("default", "00-ALLEN")]["points"] == pytest.approx(250 * 0.04 + 8 + 3)
    assert p[("default", "00-PUKA")]["points"] == pytest.approx(3 + 8 + 3)
    assert p[("sleeper:1", "00-PUKA")]["points"] == pytest.approx(6 + 8 + 3)
    assert p[("default", "DST_LA")]["points"] == pytest.approx(2.5 + 2 + 1)  # LAR -> LA
    assert p[("default", "DST_LA")]["kickoff"] == KO
    assert not any(k == "9999" or k is None for _, k in p)  # unmatched: dropped
    assert df.height == 6 and df["content_hash"].n_unique() == 1


def test_take_is_write_once_skips_unchanged_and_keeps_raw(tmp_path):
    path, n = bm.take(tmp_path, 2026, 8, XW, KICK, LEAGUES, now=NOW, raw=RAW)
    assert path is not None and n == 6
    assert list((tmp_path / bm.RAW / "season=2026" / "week=08").glob("sleeper_*.json.gz"))
    again, _ = bm.take(tmp_path, 2026, 8, XW, KICK, LEAGUES, now=KO, raw=RAW)
    assert again is None  # identical: the earlier capture stands
    changed = [RAW[0] | {"stats": {"pass_yd": 300.0}}, *RAW[1:]]
    later, _ = bm.take(tmp_path, 2026, 8, XW, KICK, LEAGUES, now=KO, raw=changed)
    assert later is not None
    assert bm.all_benchmarks(tmp_path, 2026, 8)["taken_at"].n_unique() == 2


def test_empty_response_is_an_error_not_a_silent_blank(tmp_path):
    with pytest.raises(ValueError):
        bm.take(tmp_path, 2026, 8, XW, KICK, LEAGUES, now=NOW, raw=[])
