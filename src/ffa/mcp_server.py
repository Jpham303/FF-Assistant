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
9. Rest of season, the remaining schedule, playoff weeks, "how will X do the rest of the
   year", dynasty-free trade value: call rest_of_season. Never answer these from web
   search or memory; if the tool returns an error or times out, say so and retry once
   instead of guessing. Report per_game_if_active (points a game when he plays) next to
   the total (expected after injury risk; expected_games of the remaining games), and
   the playoff-week total. Opponents come from the official schedule in the tool.
10. Games that have kicked off are locked in Sleeper: my_team shows those slots as the
   user set them (locked=true), final games with their actual points; lineup_total
   (actual + projected) is what the matchup odds use, points_so_far what's banked. Never
   suggest moving a locked player; in start/sit, relay the note when one is locked.
11. Trade questions ("should I trade X for Y", "is this fair"): call evaluate_trade, never
   web search. Lead with the verdict and its reason, then your lineup change per week and
   over the rest of the season (and in playoff weeks), the other team's change (is it
   realistic for them?), playoff and title odds before/after, any free agent added (by name, with
   his points) or player dropped, and the flags in how_it_looks. p_ahead is approximate
   (players treated as independent); say so if it's near 50%. If your_trade_value has a
   flag, say how much rest-of-season value the user gives up: a lineup gain has to be
   large to justify it. waiver_baseline lists free
   pickups the user (or they) could make right now without trading: the trade is scored
   after those, so mention them; they're worth doing either way.
12. "Who should I target", "I need a RB, what's realistic", "what can I get for X":
   call trade_targets (never web search). It applies trade_packages' checks to the
   question asked. Start with the user's biggest needs, then each package, best grade
   first: what he gives and gets, the grade and its acceptance parts, his gain per week
   and theirs, the shape (straight swap, sweetener, consolidation), playoff and title odds
   change,
   and any free-agent add. Then near_misses, saying what each missed by (grade_note).
   An empty list means nothing passes the checks: say so, don't loosen them. It only
   suggests: sending an offer is up to the user, in Sleeper.
13. A player in rest_of_season's not_found has NO projection: say it's missing (a data
   problem to report), never treat him as worth 0 points or compare anyone against him.
14. Players share last names (Pierce: Alec, IND WR, and Dameon, PHI RB). A last name alone
   is resolved from context: the team, a teammate named alongside ("Barkley and Pierce"
   means the PHI back), the position, the user's roster. Look each one up with
   search_players and check the team before using any number; if context doesn't
   settle it, ask which one.
15. "What trades should I offer", "any trades out there", "who would take X": call
   trade_packages. Lead with league_wide's Likely and Fair offers, then the best Stretch
   and Long shot, then any Near miss (say what it missed by: grade_note). For each: what
   the user gives and gets, his gain a week and theirs, the grade and its acceptance
   parts (their lineup change x their stakes, how it looks, anyone they'd cut; stakes: a
   contender weighs lineup points more, an eliminated team hardly at all), reasons for both sides,
   playoff and title odds. skipped_pickups are waiver pickups the trade leaves no room for, not
   players he owns. Grades are a scored judgment, not a probability.
16. Injuries: judge how serious an injury is and when a player returns ONLY from what the
   tools return: the designation, play_probability, the rest-of-season chance to play
   by week, and `injury_context` (or the "Injury history this season" driver): reports
   and practice by week, reserve-list weeks, the trend, whether the injury predates the
   season, Sleeper's note. Never infer severity, a timeline or "out for the season" from
   the body part ("ACL", "Achilles"): an injury on file can be last season's on a player
   who is back. Say "out for the season" only if a tool says so (a season-ending reserve
   listing or note). No return date exists in this data: never invent one; say what the
   reports show (e.g. "practicing in full, upgraded to Questionable: on track to return").
17. Every trade, every tool: show BOTH sides' gain or loss, never only the user's: points
   a week, playoff odds change and title odds change, for the user and for the other
   team, e.g. a two-row table (You | Them) or "You: +1.2 a week, playoffs 54% -> 61%,
   title 9% -> 12%. Them: -0.4 a week, playoffs 40% -> 37%, title 5% -> 4%". Compact
   offers group these under `you` and `them`; the full form has per_week (you_per_week /
   them_per_week), playoff_odds and title_odds (before, after, change). Title odds
   simulate the league's bracket (seeding, byes, round lengths). If title_odds is null,
   say it's unavailable and why (notes).
18. Trades from trade_packages and trade_targets are ALTERNATIVES, each scored on its own
   against the user's current roster (after the free waiver moves): never present them
   as a sequence ("do this, then that"), never combine two, and never assume a player
   from one trade is available for another. Offers can share players for that reason.
   Generated trades send up to 3 players and receive up to 2; a trade the user names in
   evaluate_trade can be any size.
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
            # Under the 60 s a remote client waits, so a slow API answers with a reason
            # instead of a silent timeout.
            base_url=os.environ.get("FFA_API_URL", "http://localhost:8000"),
            timeout=50,
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
    except httpx.TimeoutException as e:
        raise ApiError(
            "The FF Assistant API took more than 50 seconds to answer. It is usually busy "
            "(a projection job running in Dagster) or just restarted after a deploy. Try "
            "again in a minute; if it keeps happening, check `docker stats`."
        ) from e
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


def _resolve(players: list[str], week: int | None,
             need_projection: bool = True) -> tuple[list[str], list[dict]]:  # fmt: skip
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
        if len(exact) == 1 and (exact[0]["projected"] or not need_projection):
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
    stats if active, points by scoring rule, drivers, and rest_of_season (expected points
    for every remaining week through the fantasy championship, the total and the playoff-
    week total; for trade questions, rest of season is what matters), and for an injured
    or returning player injury_context: this season's reports, practice, reserve-list
    weeks and trend (the basis for any statement about the injury).
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
def rest_of_season(players: list[str], league: str | None = None) -> dict:
    """Rest-of-season outlook for one or more players: projected fantasy points for every
    remaining week through the fantasy championship (opponent, expected points, chance to
    play), the rest-of-season total and the playoff-week total, in the league's scoring.
    Use for "rest of season", "ROS", "remaining schedule", "playoff schedule", "who will
    score more the rest of the way", and as the basis for trade questions. Injured or
    returning players carry injury_context (reports, practice, trend this season).
    players: names or keys; league: id from list_leagues (defaults to the user's league)."""
    try:
        keys, problems = _resolve(players, None, need_projection=False)  # byes are fine
        if problems:
            return {"unresolved": problems, "resolved_keys": keys}
        params = {"keys": ",".join(keys)} | ({"league": league} if league else {})
        return _call("GET", "/rest-of-season", params=params)
    except ApiError as e:
        return {"error": str(e)}


@mcp.tool(annotations=READ_ONLY)
def evaluate_trade(give: list[str], get: list[str], league: str | None = None,
                   playoff_weighting: bool = False) -> dict:  # fmt: skip
    """Should I make this trade? Evaluates a proposed trade in the user's league: how both
    teams' best weekly lineups change for every remaining week (per week, rest of season,
    fantasy playoff weeks), who gets dropped and which free agent fills a freed spot
    (named, with the points he adds), how it looks to managers (season-to-date positional
    rank for same-position trades, projected points otherwise, with stale-rank flags),
    playoff and title odds before and after for both teams, a verdict and the reasons.
    give: the user's players; get: players from ONE other team (names or keys, up to 4 a
    side). playoff_weighting: count playoff weeks double. Read-only: it never proposes
    anything in Sleeper."""
    try:
        keys_give, p1 = _resolve(give, None, need_projection=False)
        keys_get, p2 = _resolve(get, None, need_projection=False)
        if p1 or p2:
            return {"unresolved": p1 + p2}
        body = {"give": keys_give, "get": keys_get, "playoff_weighting": playoff_weighting}
        return _call("POST", "/trade/evaluate", json=body | ({"league": league} if league else {}))
    except ApiError as e:
        return {"error": str(e)}


@mcp.tool(annotations=READ_ONLY)
def trade_targets(give: list[str] | None = None, position: str | None = None,
                  partner: str | None = None, league: str | None = None,
                  playoff_weighting: bool = False) -> dict:  # fmt: skip
    """Who should I target in a trade? The trade package generator narrowed to the
    question: trades that raise the user's weekly lineup by at least the minimum gain
    (more when he gives up trade value), graded by how likely the other manager is to
    accept (Likely, Fair, Stretch, Long shot). Returns up to 5, best grade first (one per
    team unless partner is set), each fully scored like evaluate_trade, plus up to 3
    near misses and the user's needs by position. give: players the user is offering
    (names; empty = any); position: only targets at this position ("I need a RB");
    partner: only this team. Takes up to ~1 min league-wide. Read-only: never proposes anything."""
    try:
        keys_give, p1 = _resolve(give or [], None, need_projection=False)
        if p1:
            return {"unresolved": p1}
        body = {"give": keys_give, "position": position, "partner": partner,
                "playoff_weighting": playoff_weighting}  # fmt: skip
        if league:
            body["league"] = league
        return _call("POST", "/trade/targets", json=body)
    except ApiError as e:
        return {"error": str(e)}


@mcp.tool(annotations=READ_ONLY)
def trade_packages(partner: str | None = None, refresh: bool = False,
                   league: str | None = None, playoff_weighting: bool = False,
                   top: int = 12, full: bool = False) -> dict:  # fmt: skip
    """Trade offers to make: for every opponent in the user's league (or one `partner`
    team), the best offer in each acceptance grade (Likely, Fair, Stretch, Long shot)
    where the user gains, with both lineups' change per week, playoff and title odds,
    reasons for
    both sides (with numbers) and what the other manager sees (points per game so far,
    last 3 games, draft pick, injury). league_wide lists them all, Likely first, by the
    user's gain. Compact by default (the best `top` offers, the fields needed to
    decide); full=true for every field (large). Precomputed daily; refresh=true
    recomputes (can take a minute). Read-only: never proposes anything."""
    try:
        params = {
            "refresh": refresh,
            "playoff_weighting": playoff_weighting,
            "compact": not full,
            "top": top,
        }
        params |= {"partner": partner} if partner else {}
        params |= {"league": league} if league else {}
        return _call("GET", "/trade/packages", params=params)
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
