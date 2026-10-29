"""MCP server tests: tools run against the in-process API on the synthetic warehouse."""

from __future__ import annotations

import json

import anyio
import pytest

from ffa import mcp_server
from tests.test_api import BYE, client, db, warehouse  # noqa: F401  (fixtures)


@pytest.fixture
def tools(client):  # noqa: F811
    mcp_server.set_client(client)
    yield
    mcp_server.set_client(None)


def call(name: str, **args):
    res = anyio.run(mcp_server.mcp.call_tool, name, args)
    assert not res.is_error, res
    if res.structured_content is not None:
        out = res.structured_content
        return out.get("result", out) if set(out) == {"result"} else out
    return json.loads(res.content[0].text)


def test_tools_listed_read_only():
    tools = anyio.run(mcp_server.mcp.list_tools)
    names = {t.name for t in tools}
    assert names == {"current_week", "search_players", "start_sit", "player_projections",
                     "list_leagues", "my_team", "rest_of_season", "evaluate_trade",
                     "trade_targets", "trade_packages", "trade_cards"}  # fmt: skip
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)
    assert "SMALL factor" in mcp_server.INSTRUCTIONS


def test_start_sit_by_name(tools):
    d = call("start_sit", players=["Alpha Back", "Bravo Back"])
    assert d["recommendation"] == "00-A"
    assert d["players"][0]["drivers"]


def test_start_sit_refuses_to_swap_names(tools):
    d = call("start_sit", players=["Alpha Back", "Echo Williams"])
    (p,) = d["unresolved"]
    assert "no projection this week" in p["problem"]
    d = call("start_sit", players=["Alpha Bak", "Bravo Back"])  # typo: ask, don't guess
    assert d["unresolved"][0]["candidates"][0]["key"] == "00-A"
    assert "recommendation" not in d


def test_league_scoring_and_my_team(tools):
    d = call("start_sit", players=["00-C", "00-D"], league="sleeper:1")
    assert d["scoring"] == "Test League (sleeper)"
    t = call("my_team")  # the only synced league is the default
    assert t["team"] == "Mine"
    assert {r["name"]: r["reason"] for r in t["not_playing"]}[BYE[1]] == "bye"


def test_api_errors_become_messages(tools):
    d = call("start_sit", players=["00-A", "00-B"], league="sleeper:404")
    assert "Unknown league" in d["error"]


def test_api_down_is_explained():
    import httpx

    mcp_server.set_client(httpx.Client(base_url="http://127.0.0.1:9", timeout=1))
    try:
        assert "unreachable" in call("current_week")["error"]
    finally:
        mcp_server.set_client(None)


def test_rest_of_season_explains_a_missing_run(tools):
    d = call("rest_of_season", players=["Alpha Back"])
    assert "No rest-of-season projections yet" in d["error"]


def test_rest_of_season_from_a_snapshot(tools, warehouse, monkeypatch):  # noqa: F811
    from datetime import UTC, datetime

    import polars as pl

    from ffa.service import ros
    from ffa.warehouse import store

    rows = pl.DataFrame({
        "league": ["default"] * 3, "key": ["00-A"] * 3, "name": ["Alpha Back"] * 3,
        "position": ["RB"] * 3, "team": ["DAL"] * 3, "target_week": [6, 7, 15],
        "opponent": ["GB", "PHI", "NYG"], "play_probability": [1.0, 0.9, 0.8],
        "expected_points": [12.0, 11.0, 10.0], "playoff": [False, False, True],
        "proj_if_active": [12.0, 12.2, 12.5],
        "season": [2026] * 3, "week": [6] * 3,
    })  # fmt: skip
    store.write(rows, warehouse, ros.DATASET, volatile=True, ingested_at=datetime.now(UTC))
    # The snapshot's own numbers (the live current week is tested in test_ros).
    monkeypatch.setattr(ros, "live_week", lambda *a, **k: pl.DataFrame())
    d = call("rest_of_season", players=["Alpha Back", "Echo Williams"])  # Echo: on bye
    p = {x["name"]: x for x in d["players"]}["Alpha Back"]
    assert p["total"] == 33.0 and p["playoff_weeks_total"] == 10.0 and len(p["by_week"]) == 3
    assert p["per_game_if_active"] == 12.2 and p["expected_games"] == 2.7
    assert d["not_found"] == ["00-E"]  # resolved despite the bye, but not in the run


def test_slow_api_is_explained(monkeypatch):
    import httpx

    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    mcp_server.set_client(httpx.Client(base_url="http://api", transport=httpx.MockTransport(slow)))
    try:
        assert "took more than 50 seconds" in call("current_week")["error"]
    finally:
        mcp_server.set_client(None)
