"""Traded players (Nov 11): Sleeper's current team wins over nflverse's lagging roster."""

from __future__ import annotations

import duckdb
import polars as pl
import pytest

from ffa.model import inputs
from ffa.service import availability, startsit


def _roster() -> pl.DataFrame:
    return pl.DataFrame({"key": ["moved", "stay", "fa"], "name": ["M", "S", "F"],
                         "team": ["NYJ", "DET", "MIA"], "position": ["WR"] * 3,
                         "roster_status": ["ACT"] * 3})  # fmt: skip


def test_sleepers_team_wins_and_the_old_team_is_kept():
    now = pl.DataFrame({"key": ["moved", "stay"], "team": ["DAL", "DET"]})
    got = {r["key"]: r for r in inputs.current_teams(_roster(), now).iter_rows(named=True)}
    assert (got["moved"]["team"], got["moved"]["traded_from"]) == ("DAL", "NYJ")
    assert (got["stay"]["team"], got["stay"]["traded_from"]) == ("DET", None)
    assert (got["fa"]["team"], got["fa"]["traded_from"]) == ("MIA", None)  # Sleeper silent
    plain = inputs.current_teams(_roster(), None)
    assert plain["team"].to_list() == ["NYJ", "DET", "MIA"]
    assert plain["traded_from"].null_count() == 3


class _PM:
    position_prior = {"tgt_share": {"WR": 0.15}, "car_share": {"WR": 0.01}}


def test_a_traded_players_shares_are_half_his_own_until_two_new_team_games():
    roster = inputs.current_teams(_roster(), pl.DataFrame({"key": ["moved"], "team": ["DAL"]}))
    now = pl.DataFrame({"key": ["moved", "stay"], "position": ["WR", "WR"],
                        "tgt_share_s": [0.25, 0.25], "car_share_s": [0.03, 0.03]})  # fmt: skip
    hist = pl.DataFrame({"key": ["moved"] * 3, "season": [2026] * 3, "week": [7, 8, 9],
                         "team": ["NYJ", "NYJ", "DAL"], "played": [True] * 3})  # fmt: skip
    got = {r["key"]: r for r in inputs._new_team_shares(now, roster, hist, _PM, 2026)
           .iter_rows(named=True)}  # fmt: skip
    assert got["moved"]["new_team_games"] == 1
    assert got["moved"]["tgt_share_s"] == pytest.approx(0.5 * 0.25 + 0.5 * 0.15)
    assert got["moved"]["car_share_s"] == pytest.approx(0.02)
    assert got["stay"]["tgt_share_s"] == 0.25 and got["stay"]["traded_from"] is None
    # Two games with the new team: his own estimates (they now include those games).
    hist2 = hist.vstack(pl.DataFrame({"key": ["moved"], "season": [2026], "week": [10],
                                      "team": ["DAL"], "played": [True]}))  # fmt: skip
    got2 = inputs._new_team_shares(now, roster, hist2, _PM, 2026).row(0, named=True)
    assert got2["new_team_games"] == 2 and got2["tgt_share_s"] == 0.25


def test_the_traded_flag_is_said():
    row = {"traded_from": "NYJ", "team": "DAL", "new_team_games": 0, "position": "WR"}
    line = startsit.traded_line(row)
    assert line.startswith("Traded: now on DAL (Sleeper), from NYJ; no games yet")
    assert "LOW CONFIDENCE" in line and startsit.traded_confidence(row) == "low"
    assert startsit.traded_confidence(row | {"new_team_games": 2}) == "normal"
    assert startsit.traded_line({"team": "DAL"}) is None
    assert "Traded: now on DAL" in startsit.drivers(row, None, "WR")[0]


def test_a_defense_and_a_traded_player_on_bye_read_bye(monkeypatch):
    """Nov 11: 'not on an NFL roster' for a D/ST (no roster row) and a traded player's
    bye taken from his old team."""
    con = duckdb.connect()
    con.sql("create table player_ids as select 'moved' as gsis_id, 's1' as sleeper_id, "
            "'NYJ' as team")  # fmt: skip
    con.sql("create table sleeper_players as select 's1' as sleeper_id, 'DAL' as team")
    monkeypatch.setattr(availability, "_latest_roster", lambda *a: pl.DataFrame(
        {"key": ["moved"], "team": ["NYJ"], "status": ["ACT"], "code": [None],
         "roster_week": [9]}))  # fmt: skip
    monkeypatch.setattr(availability, "_teams_playing", lambda *a: {"NYJ", "BUF"})
    monkeypatch.setattr(availability, "_sleeper", lambda *a: {})
    out = {
        r.key: r for r in availability.reasons(con, 2026, 10, {"moved": "M", "DST_KC": "KC D/ST"})
    }
    assert out["moved"].reason == "bye" and "DAL" in out["moved"].detail
    assert out["DST_KC"].reason == "bye" and "KC" in out["DST_KC"].detail
