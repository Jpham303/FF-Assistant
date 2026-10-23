"""MCP server: lets AI assistants (Claude Desktop, etc.) use the FF Assistant API.

A thin translator. Every number comes from the API, which serves the backtested
simulation model, so an assistant answering from these tools gives the same answer as the
app. Tools are read-only.

Run: `uv run ffa-mcp` (stdio). The API must be running (FFA_API_URL, default
http://localhost:8000).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

INSTRUCTIONS = """\
Fantasy football projections and start/sit decisions from a simulation model that was
backtested on the full 2025 season (it beat season-average and opportunity-based
baselines on start/sit accuracy). Rules for answering with these tools:

1. Only cite numbers that a tool returned (projections, percentages, shares, ratings,
   betting lines). Never invent or recall stats, ranks or news from elsewhere and present
   them as the model's.
2. Lead with the recommendation, how likely it is to be right (p_a_beats_b) and the
   median_gap in points. Below 55% it is a coin flip; say so plainly rather than
   overselling. Then the top `decision_drivers` (what separates the two, in points, with
   their detail). When `matchup` is available, give the user's chance of beating that
   opponent with each option; on a coin flip the recommendation may come from that
   tiebreaker (floor when favored, ceiling when the underdog): relay its note.
3. Explain with the `drivers` each player comes with, biggest first: role (share of
   targets/carries), game environment (implied team total), availability (injury
   status), efficiency. Matchup ratings are a SMALL factor (they moved accuracy by
   0.1-0.3% in testing); never make a matchup the main reason.
4. Headline number: proj_if_active, the projection if he plays, with its range
   (p10_if_active-p90_if_active). Show play_probability and the injury status NEXT to
   it; never fold availability into the headline. expected_points (= play_probability x
   proj_if_active) is what lineups are ranked by: show it as secondary, labelled as such.
   Players not playing have no if-active projection: list them with 0 and the reason.
   A player under 100% comes with `game_day_check` (when inactives are announced for his
   game): relay it next to his play probability. `mean` is deprecated: don't use it.
   In start/sit, p_a_beats_b counts the chance of sitting; p_a_beats_b_if_both_active is
   talent alone. When they pick different players, relay the note that explains it.
5. Name the data behind the answer: `sources`, when projections ran, and for every injured
   or questionable player the injury source and time that come with him ("Out - Sleeper,
   updated Fri 4:12 PM"). The model does not read news. Game statuses come from nflverse
   and from Sleeper, fetched once a day (~5 PM CT; Sunday ~10:45 AM CT), so a late scratch
   or inactive after that is not reflected: say so, and tell the user to confirm game-day
   status before lineups lock. If current_week shows sleeper_status_fetched_at as null or
   more than a day old, say the second injury source is missing or stale.
6. A player with no game status yet but a missed or limited practice is a real risk
   (statuses usually arrive with Friday's final report); report his play probability.
   A player ruled out is never the recommendation; say clearly that he is out.
7. For players who aren't playing, give each one's `reason` and `detail` as returned
   (bye, out, injured_reserve, pup, suspended, released, not_on_nfl_roster, ...). Call it a
   bye only when reason is "bye". Point out players with may_return=true who are not on
   bye: they may come back later in the season.
8. If a player is not found or not playing this week, say so. Never substitute a
   similarly named player.
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
KEY = re.compile(r"^(\d{2}-\w+|DST_[A-Z]{2,3})$")  # gsis_id or team defense

logging.getLogger("httpx").setLevel(logging.WARNING)  # keep stderr quiet in clients

mcp = MCPServer("ff-assistant", instructions=INSTRUCTIONS)
_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            base_url=os.environ.get("FFA_API_URL", "http://localhost:8000"), timeout=60
        )
    return _client


def set_client(c: httpx.Client | None) -> None:
    """Point the tools at another client (tests use the in-process API)."""
    global _client
    _client = c


class ApiError(Exception):
    pass


def _call(method: str, path: str, **kw) -> Any:
    try:
        r = client().request(method, path, **kw)
    except httpx.HTTPError as e:
        raise ApiError(
            f"FF Assistant API unreachable at {client().base_url} ({e.__class__.__name__}). "
            "Is it running? (docker compose up)"
        ) from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(str(detail))
    return r.json()


def _resolve(players: list[str], week: int | None) -> tuple[list[str], list[dict]]:
    """Names or keys -> keys. Accepts a name only on an exact match that is playing this
    week; everything else comes back as a problem for the assistant to relay or clarify."""
    keys, problems = [], []
    for p in players:
        if KEY.match(p.strip()):
            keys.append(p.strip())
            continue
        params = {"q": p, "limit": 5} | ({"week": week} if week else {})
        found = _call("GET", "/players/search", params=params)
        exact = [r for r in found if r["match"] == 1.0]
        if len(exact) == 1 and exact[0]["projected"]:
            keys.append(exact[0]["key"])
        elif len(exact) == 1:
            r = exact[0]
            why = "has no projection this week: bye week or not on an active roster"
            problems.append({"query": p, "problem": f"{r['name']} ({r['team']}) {why}"})
        elif len(exact) > 1:
            problems.append({"query": p, "problem": "several players share this name; "
                             "ask which one and pass their key", "candidates": exact})  # fmt: skip
        else:
            problems.append({"query": p, "problem": "no exact match; ask the user which "
                             "player they meant", "candidates": found})  # fmt: skip
    return keys, problems


# ---------------------------------------------------------------- tools


@mcp.tool(annotations=READ_ONLY)
def current_week() -> dict:
    """The upcoming NFL week, whether projections exist for it, when they were generated,
    when each injury source was last fetched (nflverse, Sleeper), a caveat about game-day
    statuses, and the data sources to cite."""
    try:
        return _call("GET", "/week")
    except ApiError as e:
        return {"error": str(e)}


@mcp.tool(annotations=READ_ONLY)
def search_players(name: str, position: str | None = None) -> list[dict] | dict:
    """Find players (or team defenses, e.g. "JAX") by name; typos allowed.

    Each result: key, name, team, position, match (1.0 = exact name) and projected (False
    = not playing this week, usually a bye). Use this when a name is ambiguous or another
    tool reports no exact match. A close match is NOT the same player: confirm with the
    user before using it."""
    try:
        return _call("GET", "/players/search", params={"q": name, "position": position}
                     if position else {"q": name})  # fmt: skip
    except ApiError as e:
        return {"error": str(e)}


@mcp.tool(annotations=READ_ONLY)
def start_sit(players: list[str], league: str | None = None, week: int | None = None) -> dict:
    """Who to start among 2-6 players, scored with the league's own rules.

    players: names ("Javonte Williams", "JAX" for a defense) or keys from search_players.
    league: league id from list_leagues (e.g. "sleeper:123"); defaults to the user's
    configured league, else standard half-PPR.

    Returns the recommendation (by expected points; on a close call, by the user's chance
    of winning this week's matchup), close_call (true below 55%), pairwise p_a_beats_b
    (all simulations: the chance of sitting counts), p_a_beats_b_if_both_active (talent
    alone) and median_gap, decision_drivers (biggest point differences, with detail),
    matchup (opponent and P(win) with each candidate, when a league is used), notes
    (injuries, coin flips, tiebreaker, and when the odds disagree), and per player:
    proj_if_active with its range (the headline),
    play_probability and injury status (shown beside it), expected_points (what ranks),
    expected stats if active, points by scoring rule, and drivers (plain-language
    reasons, each with its number). If a name is not found or the
    player is not playing, returns `unresolved` instead; relay it, don't guess."""
    try:
        keys, problems = _resolve(players, week)
        if problems:
            return {"unresolved": problems, "resolved_keys": keys}
        body = {"players": keys} | ({"league": league} if league else {})
        return _call("POST", "/start-sit", json=body | ({"week": week} if week else {}))
    except ApiError as e:
        return {"error": str(e)}


@mcp.tool(annotations=READ_ONLY)
def player_projections(players: list[str], league: str | None = None,
                       week: int | None = None) -> list[dict] | dict:  # fmt: skip
    """Projection and reasoning for one or more players: proj_if_active with its range
    (the headline: points if he plays), play_probability and injury status beside it,
    expected_points (= play_probability x proj_if_active, what lineups rank by), expected
    stats if active, points by scoring rule, and drivers.
    Accepts names or keys; unresolved names come back as `unresolved`."""
    try:
        keys, problems = _resolve(players, week)
        if problems:
            return {"unresolved": problems, "resolved_keys": keys}
        params = {"keys": ",".join(keys)} | ({"league": league} if league else {})
        return _call("GET", "/projections", params=params | ({"week": week} if week else {}))
    except ApiError as e:
        return {"error": str(e)}


@mcp.tool(annotations=READ_ONLY)
def list_leagues() -> list[dict] | dict:
    """The user's synced fantasy leagues: id, name, platform, season and lineup slots."""
    try:
        return _call("GET", "/leagues")
    except ApiError as e:
        return {"error": str(e)}


@mcp.tool(annotations=READ_ONLY)
def my_team(league: str | None = None, week: int | None = None) -> dict:
    """The user's roster in a league: the lineup that maximizes expected points under
    the league's slots and scoring (each slot shows proj_if_active, play_probability,
    expected_points and injury status), the bench with full projections and drivers, and
    rostered players who are not playing, each with a reason (bye, out, injured reserve,
    PUP, suspended, released, not on an NFL roster...), its source and whether he may
    return; players in the league's IR slot are included. league defaults to the only synced
    league. Changing the actual lineup is up to the user, on their platform."""
    try:
        if league is None:
            leagues = _call("GET", "/leagues")
            if len(leagues) != 1:
                ids = ", ".join(lg["league"] for lg in leagues) or "none"
                return {"error": f"Specify league; synced leagues: {ids}"}
            league = leagues[0]["league"]
        params = {"week": week} if week else {}
        return _call("GET", f"/leagues/{league}/my-team", params=params)
    except ApiError as e:
        return {"error": str(e)}


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
