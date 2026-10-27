"""Trade evaluation end to end on the API fixtures: league, rosters, a stored
rest-of-season run, Sleeper's schedule (mocked) and playoff odds."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from ffa.db.models import LeagueRow, RosterEntryRow, TeamRow
from ffa.scoring.rules import from_sleeper
from ffa.service import gamestate, ros
from ffa.trade import service
from ffa.warehouse import duck, store
from tests.test_api import SEASON, WEEK, client, db, warehouse  # noqa: F401  (fixtures)

PTS = {"00-A": 14.0, "00-B": 9.0, "00-C": 12.0, "00-D": 7.0, "00-Q": 18.0, "00-F": 11.0}
POS = {"00-A": "RB", "00-B": "RB", "00-C": "WR", "00-D": "WR", "00-Q": "QB", "00-F": "WR"}


def _schedule(_lid, w):
    return [{"roster_id": 8, "matchup_id": 1, "starters": []},
            {"roster_id": 9, "matchup_id": 1, "starters": []}]  # fmt: skip


@pytest.fixture
def trade_league(db, warehouse, monkeypatch):  # noqa: F811
    row = db.get(LeagueRow, "sleeper:1")
    row.settings = {"playoff_week_start": 9, "playoff_teams": 1, "last_week": 10}
    db.add(TeamRow(id="sleeper:1:9", league_id="sleeper:1", platform_team_id="9",
                   name="Theirs", wins=3, losses=2))  # fmt: skip
    db.add(RosterEntryRow(team_id="sleeper:1:9", platform_player_id="00-Q", player_key="00-Q",
                          name="Quinn Passer", position="QB", slot_type="starter"))  # fmt: skip
    me = db.get(TeamRow, "sleeper:1:8")
    me.wins, me.losses = 2, 3
    db.commit()
    rows = [{"league": lg, "key": k, "name": k, "position": POS[k], "team": "DAL",
             "target_week": w, "opponent": "GB", "play_probability": 1.0,
             "expected_points": PTS[k], "proj_if_active": PTS[k], "sd": 5.0,
             "playoff": w >= 9, "season": SEASON, "week": WEEK}
            for lg in ("sleeper:1", "default") for k in PTS for w in range(WEEK, 11)]  # fmt: skip
    store.write(pl.DataFrame(rows), warehouse, ros.DATASET, volatile=True,
                ingested_at=datetime.now(UTC))  # fmt: skip
    monkeypatch.setattr(gamestate, "fetch_matchups", _schedule)
    return db, row, warehouse


def test_evaluate_trade_both_sides_free_agent_and_playoff_odds(trade_league):
    db_, row, wh = trade_league
    con = duck.connect(wh)
    r = service.evaluate_trade(con, db_, row, from_sleeper(row.scoring), SEASON, WEEK,
                               give=["00-A"], get=["00-Q"], fetch=_schedule)  # fmt: skip
    weeks = list(range(WEEK, 11))
    assert r["weeks"] == weeks and r["playoff_weeks"] == [9, 10]
    # Waiver baseline first: you'd already add the free agent Foxtrot (WR 11) into your
    # open roster spot; he takes the flex from Bravo (9): +2 a week without any trade.
    # You (QB, RB, WR, FLEX), from that baseline: before 0 + Alpha 14 + Charlie 12 +
    # Foxtrot 11 = 37; after Quinn 18 + Bravo 9 + Charlie 12 + Foxtrot 11 = 50: +13 a
    # week, 5 weeks. (Without the baseline it read +11 from 35 to 46.) They lose Quinn's
    # 18 and start Alpha (14): -4 a week.
    moves = r["waiver_baseline"]["moves"]
    assert [(m["add"], m["drop"]) for m in moves] == [("00-F", None)]  # an open spot
    assert r["you"]["per_week"] == 13.0 and r["you"]["ros_change"] == 65.0
    assert r["them"]["per_week"] == -4.0 and r["them"]["team_name"] == "Theirs"
    assert r["playoff_odds"]["you"]["after"] > r["playoff_odds"]["you"]["before"]
    assert r["give"] == ["00-A"] and r["get"] == ["00-Q"]


def test_trade_needs_the_players_on_the_right_rosters(trade_league):
    db_, row, wh = trade_league
    with pytest.raises(ValueError, match="Not on your roster"):
        service.evaluate_trade(duck.connect(wh), db_, row, from_sleeper(row.scoring), SEASON,
                               WEEK, give=["00-Q"], get=["00-A"], fetch=_schedule)  # fmt: skip


def test_api_and_tool(trade_league, client, monkeypatch):  # noqa: F811
    from ffa import mcp_server

    r = client.post("/trade/evaluate", json={"give": ["00-A"], "get": ["00-Q"],
                                             "league": "sleeper:1"})  # fmt: skip
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["verdict"] in ("accept", "decline", "close call") and d["you"]["per_week"] == 13.0
    bad = client.post("/trade/evaluate", json={"give": ["00-Q"], "get": ["00-A"],
                                               "league": "sleeper:1"})  # fmt: skip
    assert bad.status_code == 422 and "Not on your roster" in bad.text
    mcp_server.set_client(client)
    try:
        from tests.test_mcp import call

        t = call("evaluate_trade", give=["Alpha Back"], get=["Quinn Passer"], league="sleeper:1")
        assert t["you"]["per_week"] == 13.0
    finally:
        mcp_server.set_client(None)


def test_trade_with_an_unprojected_player_is_refused_not_valued_at_zero(trade_league):
    db_, row, wh = trade_league
    # A player on the partner's roster with no rest-of-season rows (the bye-week bug).
    db_.add(RosterEntryRow(team_id="sleeper:1:9", platform_player_id="00-W", player_key="00-W",
                           name="Wash Ington", position="WR", slot_type="bench"))  # fmt: skip
    db_.commit()
    con = duck.connect(wh)
    with pytest.raises(LookupError, match="Wash Ington"):
        service.evaluate_trade(con, db_, row, from_sleeper(row.scoring), SEASON, WEEK,
                               give=["00-A"], get=["00-W"], fetch=_schedule)  # fmt: skip


def test_trade_targets_service_api_and_tool(trade_league, client):  # noqa: F811
    from ffa import mcp_server

    db_, row, wh = trade_league
    con = duck.connect(wh)
    r = service.trade_targets(con, db_, row, from_sleeper(row.scoring), SEASON, WEEK,
                              fetch=_schedule)  # fmt: skip
    assert "QB" in r["your_needs"] and r["searched"]["partner"] == "every team"
    for p in r["packages"]:  # each one scored like evaluate_trade
        assert p["you"]["ros_change"] > 0 and "verdict" in p and p["shape"]
    api = client.post("/trade/targets", json={"league": "sleeper:1", "position": "QB"})
    assert api.status_code == 200, api.text
    pk = api.json()["packages"]
    # Alpha (RB, 14) for Quinn (QB, 18) costs them 4 a week: past THEIR_MAX_LOSS, so no
    # QB package is realistic in this two-team league.
    assert all(p["get"] == ["00-Q"] for p in pk) and len(pk) <= 1
    bad = client.post("/trade/targets", json={"league": "sleeper:1", "partner": "Nobody"})
    assert bad.status_code == 422 and "Theirs" in bad.text
    mcp_server.set_client(client)
    try:
        from tests.test_mcp import call

        t = call("trade_targets", position="QB", league="sleeper:1")
        assert "packages" in t and "your_needs" in t
    finally:
        mcp_server.set_client(None)


def test_a_player_in_your_ir_slot_can_be_traded(trade_league):
    """Reported Oct 27: "Not on your roster: Alec Pierce" for a player in the IR slot."""
    db_, row, wh = trade_league
    db_.add(RosterEntryRow(team_id="sleeper:1:8", platform_player_id="00-F", player_key="00-F",
                           name="Foxtrot Wide", position="WR", slot_type="ir"))  # fmt: skip
    db_.commit()
    r = service.evaluate_trade(duck.connect(wh), db_, row, from_sleeper(row.scoring), SEASON,
                               WEEK, give=["00-F"], get=["00-Q"], fetch=_schedule)  # fmt: skip
    assert r["give"] == ["00-F"] and r["get"] == ["00-Q"]
    # Your IR-slot players count in your lineup in the weeks they're projected to play:
    # before = QB 0 + Alpha 14 + Charlie 12 + flex Foxtrot 11 = 37. After: Quinn 18 +
    # Alpha 14 + Charlie 12 + flex Bravo 9 = 53: +16 a week.
    assert r["you"]["per_week"] == 16.0


class _Con:
    """Stands in for the warehouse: returns these game logs for any query."""

    def __init__(self, pg=None, exc=None):
        self.pg, self.exc = pg, exc

    def sql(self, _q):
        if self.exc:
            raise self.exc
        return self

    def pl(self):
        return self.pg


def test_ranks_season_to_date_and_per_game_rest_of_season():
    from ffa.scoring.rules import default_rules
    from ffa.trade import context, core

    pg = pl.DataFrame({"key": ["a", "a", "b", "c"], "position": ["WR"] * 4,
                       "rec": [5, 5, 4, 1], "rec_yd": [100, 100, 60, 10]})  # fmt: skip
    players = {
        # a: best so far; b: fewer points so far but the best per game ahead
        "a": core.Player("a", "A", "WR", {9: (10.0, 3.0)}, per_game=11.0),
        "b": core.Player("b", "B", "WR", {9: (9.0, 3.0)}, per_game=15.0),
        "c": core.Player("c", "C", "WR", {9: (5.0, 3.0)}, per_game=6.0),
    }
    assert context._ranks(_Con(pg), default_rules(), SEASON, 9, players) is None
    assert [players[k].season_rank for k in "abc"] == [1, 2, 3]
    assert [players[k].ros_rank for k in "abc"] == [2, 1, 3]
    assert players["a"].season_points == 2 * (5 * 0.5 + 10.0)  # half PPR: 2 games


def test_rank_failure_is_reported_not_hidden():
    from ffa.scoring.rules import default_rules
    from ffa.trade import context, core

    players = {"a": core.Player("a", "A", "WR", {9: (10.0, 3.0)}, per_game=11.0)}
    err = context._ranks(_Con(exc=RuntimeError("boom")), default_rules(), SEASON, 9, players)
    assert err == "RuntimeError: boom" and players["a"].season_rank is None
    assert players["a"].ros_rank == 1  # projections still rank


def test_rest_of_season_drops_this_weeks_row_once_his_game_kicked_off(monkeypatch):
    from ffa.api import main
    from ffa.tracking import snapshots

    now = datetime(2026, 10, 26, 20, 0, tzinfo=UTC)  # Monday afternoon
    ko = pl.DataFrame({"team": ["PHI", "NE"], "kickoff": [
        datetime(2026, 10, 25, 17, 0, tzinfo=UTC),   # Sunday: played
        datetime(2026, 10, 27, 0, 15, tzinfo=UTC),   # Monday night: not yet
    ]})  # fmt: skip
    monkeypatch.setattr(snapshots, "kickoffs", lambda con, s, w: ko)
    snap = pl.DataFrame({"team": ["PHI", "PHI", "NE", "NE"], "target_week": [7, 8, 7, 8]})
    out = main._unplayed(None, snap, SEASON, 7, now=now)
    assert out.select("team", "target_week").rows() == [("PHI", 8), ("NE", 7), ("NE", 8)]
