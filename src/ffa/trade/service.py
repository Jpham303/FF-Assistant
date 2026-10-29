"""Evaluate a trade end to end: league context, both sides, playoff odds before/after."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from ffa.db.models import LeagueRow
from ffa.scoring.rules import ScoringRules
from ffa.trade import context, core, packages, playoffs, search


def _weekly(ctx: context.Context, rosters: dict[str, list[str]]) -> dict:
    """team -> week -> (expected points, variance) of its best lineup: the remaining
    regular season and the playoff weeks (for title odds)."""
    out: dict[str, dict[int, tuple[float, float]]] = {}
    weeks = sorted(set(ctx.schedule) | set(ctx.league.playoff_weeks))
    for tid, keys in rosters.items():
        out[tid] = {}
        for w in weeks:
            if w == ctx.current_week:
                out[tid][w] = ctx.current.get(tid, (0.0, 0.0))
            else:
                mu, var, _ = core.lineup(ctx.league, keys, w)
                out[tid][w] = (mu, var)
    return out


def evaluate_trade(con, db, row: LeagueRow, rules: ScoringRules, season: int, week: int,
                   give: list[str], get: list[str], playoff_weighting: bool = False,
                   now: datetime | None = None, fetch=None) -> dict:  # fmt: skip
    ctx = context.build(con, db, row, rules, season, week, now=now, fetch=fetch)
    lg = ctx.league
    mine = lg.teams[ctx.me]
    not_mine = [k for k in give if k not in mine.roster]
    if not_mine:
        raise ValueError(f"Not on your roster: {', '.join(_names(lg, not_mine))}")
    owner = {k: tid for tid, t in lg.teams.items() if tid != ctx.me for k in t.roster}
    unowned = [k for k in get if k not in owner]
    if unowned:
        raise ValueError(f"Not on another team's roster: {', '.join(_names(lg, unowned))}")
    partners = {owner[k] for k in get}
    if len(partners) != 1:
        raise ValueError("The players you'd get must all be on one team")
    partner = partners.pop()
    # Never value a traded player at 0 because he's missing from the projections.
    unprojected = [k for k in give + get if lg.weeks and not lg.players[k].weekly]
    if unprojected:
        raise LookupError(
            f"No rest-of-season projection for {', '.join(_names(lg, unprojected))}, so this "
            "trade can't be valued. Re-run ros_projections in Dagster; if he's still missing, "
            "it's a bug (not a 0-point player).")  # fmt: skip
    waivers = _apply_waivers(ctx, playoff_weighting, frozenset(give + get))
    res = _evaluate(ctx, partner, give, get, playoff_weighting)
    res["waiver_baseline"] = _waiver_dict(ctx, waivers, {ctx.me, partner})
    res["scoring"], res["as_of"], res["notes"] = ctx.scoring, ctx.as_of, ctx.notes
    if res["playoff_odds"] is None:
        ctx.notes.append("Playoff odds unavailable: the league's playoff settings or schedule "
                         "aren't synced (run leagues_refresh).")  # fmt: skip
    elif res["title_odds"] is None:
        ctx.notes.append("Title odds unavailable: not every playoff week is projected yet "
                         "(they're added by the next rest-of-season run).")  # fmt: skip
    return res


def _evaluate(ctx: context.Context, partner: str, give: list[str], get: list[str],
              playoff_weighting: bool) -> dict:  # fmt: skip
    """One trade, fully scored: both sides, how it looks, verdict, playoff odds."""
    lg = ctx.league
    res = core.evaluate(lg, ctx.me, partner, give, get, playoff_weighting)
    res["you"]["team_name"] = lg.teams[ctx.me].name
    res["them"]["team_name"] = lg.teams[partner].name
    if ctx.schedule and ctx.playoff_teams:
        rosters = {tid: t.roster for tid, t in lg.teams.items()}

        def run(r: dict) -> dict:  # same seed both times: the change isn't simulation noise
            return playoffs.season(ctx.records, _weekly(ctx, r), ctx.schedule,
                                   int(ctx.playoff_teams), sorted(lg.playoff_weeks),
                                   ctx.playoff_round_type)  # fmt: skip

        before = run(rosters)
        rosters |= {ctx.me: res["you"]["roster_after"], partner: res["them"]["roster_after"]}
        after = run(rosters)

        def side(kind: str, tid: str) -> dict | None:
            b, a = before[tid][kind], after[tid][kind]
            if b is None or a is None:
                return None
            return {"before": b, "after": a, "change": round(a - b, 3)}

        res["playoff_odds"] = {
            "you": side("playoffs", ctx.me), "them": side("playoffs", partner),
            "regular_season_weeks_left": sorted(ctx.schedule),
        }  # fmt: skip
        title = {"you": side("title", ctx.me), "them": side("title", partner)}
        res["title_odds"] = (title | {"playoff_weeks": sorted(lg.playoff_weeks),
                                      "bracket": f"{ctx.playoff_teams} teams, fixed bracket, "
                                                 "top seeds get any byes"}
                             if title["you"] is not None else None)  # fmt: skip
    else:
        res["playoff_odds"] = None
        res["title_odds"] = None
    before = core.usable_value(lg, lg.teams[ctx.me].roster)
    after = core.usable_value(lg, res["you"]["roster_after"])
    res["usable_value"] = {
        "before_per_week": round(before, 1), "after_per_week": round(after, 1),
        "change_pct": round(100 * (after - before) / before, 1) if before > 0 else None,
        "basis": "your starters' points a week above the best free agent at each "
                 "position (bench depth doesn't count)",
    }  # fmt: skip
    res["give"], res["get"] = _names(lg, give), _names(lg, get)
    res["give_keys"], res["get_keys"] = list(give), list(get)
    for side in ("you", "them"):
        res[side].pop("roster_after", None)
    return res


GRADES_NOTE = {g: f"acceptance score at least {line} a week" for g, line in
               packages.THRESHOLDS} | {packages.NEAR_MISS: "missed a grade or a minimum by a "
                                       "hair (see grade_note)"}  # fmt: skip
ACCEPTANCE_NOTE = (
    "Acceptance score = their stakes (see `stakes`) x their lineup change a week + "
    f"{packages.LOOKS_WEIGHT} x how it "
    "looks to them (perceived points a game they come out ahead) - "
    f"{packages.CUT_WEIGHT} x perceived points a game of anyone they'd have to cut. "
    f"Your side: every offer gains you at least max({packages.OFFER_MIN_GAIN}, "
    f"{packages.VALUE_RATIO} x your_value_lost_per_week) a week, shown as "
    f"your_gain_needed ({packages.LONG_SHOT_MY_GAIN} at least for a long shot). "
    "your_value_lost_per_week is value over replacement you give minus what you get "
    "(each player's points a week above the best free agent at his position, best "
    "player first, depth counted less): above 0 you give more than you get, and "
    "the gain you need rises; below 0 you get more. Less than the minimum gain is "
    "within the model's noise. A scored grade, not a probability: there's no data "
    "on offers turned down. Completed trades in the league are logged to check it "
    "later."
)
GRADE_ORDER = {g: i for i, g in enumerate((*packages.GRADES, packages.NEAR_MISS))}
TARGETS_TOP, TARGETS_NEAR = 5, 3


def _offer_item(ctx: context.Context, tid: str, o: packages.Offer, team_needs: dict,
                picked: frozenset[str], playoff_weighting: bool,
                full: bool = False) -> dict:  # fmt: skip
    """One graded offer, fully scored like evaluate_trade, with reasons for both sides and
    what the other manager sees (trade_packages and trade_targets alike)."""
    lg, p = ctx.league, o.package
    res = _evaluate(ctx, tid, list(p.give), list(p.get), playoff_weighting)
    you, them = packages.reasons(lg, o, res, team_needs[tid])
    return {
        "grade": o.grade, "grade_note": o.note,
        "acceptance": {"score": round(o.score, 2), **o.parts},
        "partner": lg.teams[tid].name,
        "give": res["give"], "get": res["get"],
        "give_keys": res["give_keys"], "get_keys": res["get_keys"],
        "you_per_week": round(o.mine, 2), "them_per_week": round(o.theirs, 2),
        "you_playoff_weeks": res["you"]["playoff_weeks_change"],
        "reasons_you": you, "reasons_them": them,
        "what_they_see": _seen(lg, p.give, p.get, o.looks),
        "shape": p.tags or ["straight swap"],
        "their_needs_filled": sorted({lg.players[k].position for k in p.give
                                      if team_needs[tid].get(lg.players[k].position, 0) > 0}),
        "playoff_odds": res["playoff_odds"],
        "title_odds": res["title_odds"],
        "usable_value": res["usable_value"],
        **_moves(res["you"], picked, "you"), **_moves(res["them"], picked, "them"),
        "verdict": res["verdict"], "p_ahead": res["p_ahead"],
        # Target search: both sides in full, like evaluate_trade (trade_packages keeps
        # its payload small).
        **({"you": res["you"], "them": res["them"]} if full else {}),
    }  # fmt: skip


def trade_targets(con, db, row: LeagueRow, rules: ScoringRules, season: int, week: int,
                  give: list[str] | None = None, position: str | None = None,
                  partner: str | None = None, playoff_weighting: bool = False,
                  now: datetime | None = None, fetch=None) -> dict:  # fmt: skip
    """Target search: the best packages that raise your lineup and don't hurt theirs much,
    each scored like evaluate_trade."""
    ctx = context.build(con, db, row, rules, season, week, now=now, fetch=fetch)
    lg = ctx.league
    give = list(give or [])
    not_mine = [k for k in give if k not in lg.teams[ctx.me].roster]
    if not_mine:
        raise ValueError(f"Not on your roster: {', '.join(_names(lg, not_mine))}")
    unprojected = [k for k in give if lg.weeks and not lg.players[k].weekly]
    if unprojected:
        who = ", ".join(_names(lg, unprojected))
        raise LookupError(f"No rest-of-season projection for {who}; re-run ros_projections "
                          "in Dagster.")  # fmt: skip
    partners = None
    if partner:
        want = partner.lower()
        match = [t for t in lg.teams.values()
                 if t.id != ctx.me and want in (t.name.lower(), t.id.lower())]  # fmt: skip
        if not match:
            names = sorted(t.name for t in lg.teams.values() if t.id != ctx.me)
            raise ValueError(f"No team named {partner!r}. Teams: {', '.join(names)}")
        partners = [match[0].id]
    pos = position.upper() if position else None
    waivers = _apply_waivers(ctx, playoff_weighting, frozenset(give))
    lg = ctx.league
    picked = _picked_up(waivers)
    # The package generator, narrowed to the question (Oct 28, issue #2): every target
    # passes the same minimum gain, trade-value check and acceptance grade as
    # trade_packages, so the two never disagree on what's worth offering.
    st = _stakes(ctx)
    offers = packages.generate(lg, ctx.me, playoff_weighting=playoff_weighting,
                               partners=partners, untradeable=picked, give=tuple(give),
                               position=pos, stakes=st)  # fmt: skip
    graded, near = [], []
    for tid, offs in offers.items():
        for o in offs:
            (near if o.grade == packages.NEAR_MISS else graded).append((tid, o))
    near.sort(key=lambda x: -x[1].mine)
    needs = search.needs(lg)
    # Best trades first (Oct 29): every candidate scored (title odds included), then
    # Likely first and the rest together, by your title-odds gain.
    items = sorted((_offer_item(ctx, tid, o, needs, picked, playoff_weighting, full=True)
                    for tid, o in graded), key=packages.item_rank)  # fmt: skip
    if not partners:  # league-wide: each team's best, so the list isn't one roster
        seen: set[str] = set()
        items = [x for x in items if not (x["partner"] in seen or seen.add(x["partner"]))]
    items = items[:TARGETS_TOP]
    near_items = [_offer_item(ctx, tid, o, needs, picked, playoff_weighting, full=True)
                  for tid, o in near[:TARGETS_NEAR]]  # fmt: skip
    mine = needs[ctx.me]
    return {
        "your_needs": {pos_: v for pos_, v in sorted(mine.items(), key=lambda kv: -kv[1])},
        "needs_note": "Points a week your starters at each position trail (+) or lead (-) the "
                      "league's middle team there, rest of season.",
        "packages": items,
        "near_misses": near_items,
        "grades": GRADES_NOTE,
        "acceptance_note": ACCEPTANCE_NOTE,
        "stakes": _stakes_dict(ctx, st),
        "waiver_baseline": _waiver_dict(ctx, waivers, {ctx.me}),
        "searched": {"give": _names(lg, give), "position": pos,
                     "partner": lg.teams[partners[0]].name if partners else "every team",
                     "rule": "the package generator's checks (as trade_packages): your gain at "
                             "least your_gain_needed, graded Long shot or better; Likely "
                             "first, then the rest, each by your title-odds gain (then your "
                             "gain a week); one per team unless a partner is named; "
                             f"up to {search.MAX_GIVE} players sent and {search.MAX_GET} received"},
        "weeks": lg.weeks, "scoring": ctx.scoring, "as_of": ctx.as_of, "notes": ctx.notes,
    }  # fmt: skip


WAIVER_NOTE = (
    "Scored after the free waiver moves each team could already make (best free agent added, "
    "least valuable player dropped, when it adds at least {min} points a week): a trade gets "
    "credit only for what it adds beyond those."
)


def _stakes(ctx: context.Context) -> dict[str, float] | None:
    """team id -> what a lineup point is worth to it (playoffs.stakes), or None."""
    lg = ctx.league
    if not (ctx.schedule and ctx.playoff_teams):
        return None
    rosters = {tid: t.roster for tid, t in lg.teams.items()}
    return playoffs.stakes(ctx.records, _weekly(ctx, rosters), ctx.schedule,
                           int(ctx.playoff_teams), sorted(lg.playoff_weeks),
                           ctx.playoff_round_type)  # fmt: skip


def _stakes_dict(ctx: context.Context, st: dict[str, float] | None) -> dict:
    lg = ctx.league
    if st is None:
        return {"note": "unavailable (no synced schedule or projected playoff weeks): every "
                        "team's lineup change counts at 1.0"}  # fmt: skip
    return {"note": STAKES_NOTE,
            "by_team": {lg.teams[t].name: w for t, w in sorted(st.items(), key=lambda kv: -kv[1])
                        if t in lg.teams}}  # fmt: skip


STAKES_NOTE = (
    "What a point a week is worth to each team, against the league's average team (1.0): "
    "how much its title odds rise with 1 more point every remaining week. The other "
    "team's lineup change is multiplied by this in the acceptance score: a title "
    "contender weighs it more, an eliminated team hardly at all (then how the deal "
    "looks is what's left)."
)


def _apply_waivers(ctx: context.Context, playoff_weighting: bool,
                   protect: frozenset[str]) -> list[core.Waiver]:  # fmt: skip
    """Swap ctx.league for the league after every team's helpful waiver moves, worst
    record first (waiver priority)."""
    order = sorted(ctx.league.teams, key=lambda t: ctx.records.get(t, (0.0, 0.0)))
    ctx.league, moves = core.with_waivers(ctx.league, playoff_weighting, protect, order)
    return moves


def _picked_up(moves: list[core.Waiver]) -> frozenset[str]:
    """Free agents the baseline assumed picked up: not on anyone's roster, so not trade
    chips (Oct 27: the generator offered LV and NO D/ST the user doesn't own)."""
    return frozenset(m.add.key for m in moves)


def _waiver_dict(ctx: context.Context, moves: list[core.Waiver], teams: set[str]) -> dict:
    lg = ctx.league
    return {
        "note": WAIVER_NOTE.format(min=core.WAIVER_MIN),
        "moves": [{"team": lg.teams[m.team].name,
                   "add": m.add.name, "add_position": m.add.position,
                   "points_added": round(m.add.points, 1),
                   "drop": m.drop.name if m.drop else None}
                  for m in moves if m.team in teams],
    }  # fmt: skip


def _names(lg: core.League, keys: list[str]) -> list[str]:
    return [lg.players[k].name if k in lg.players else k for k in keys]


def trade_packages(con, db, row: LeagueRow, rules: ScoringRules, season: int, week: int,
                   playoff_weighting: bool = False, partner: str | None = None,
                   now: datetime | None = None, fetch=None) -> dict:  # fmt: skip
    """The package generator: for every opponent (or `partner`), the best offer in each
    acceptance grade where you gain, each scored like evaluate_trade, with reasons for
    both sides and what the other manager sees."""
    t0 = datetime.now(UTC)
    ctx = context.build(con, db, row, rules, season, week, now=now, fetch=fetch)
    lg = ctx.league
    partners = None
    if partner:
        want = partner.lower()
        match = [t for t in lg.teams.values()
                 if t.id != ctx.me and want in (t.name.lower(), t.id.lower())]  # fmt: skip
        if not match:
            names = sorted(t.name for t in lg.teams.values() if t.id != ctx.me)
            raise ValueError(f"No team named {partner!r}. Teams: {', '.join(names)}")
        partners = [match[0].id]
    waivers = _apply_waivers(ctx, playoff_weighting, frozenset())
    lg = ctx.league
    team_needs = search.needs(lg)
    picked = _picked_up(waivers)
    st = _stakes(ctx)
    offers = packages.generate(lg, ctx.me, playoff_weighting=playoff_weighting,
                               partners=partners, untradeable=picked, stakes=st)  # fmt: skip
    by_team, flat = [], []
    for tid, offs in offers.items():
        items = [_offer_item(ctx, tid, o, team_needs, picked, playoff_weighting) for o in offs]
        flat += items
        by_team.append({"team": lg.teams[tid].name, "packages": items})
    flat.sort(key=packages.item_rank)
    for t in by_team:
        t["packages"].sort(key=packages.item_rank)
    return {
        "grades": GRADES_NOTE,
        "acceptance_note": ACCEPTANCE_NOTE,
        "stakes": _stakes_dict(ctx, st),
        "by_team": by_team,
        "league_wide": flat,
        "waiver_baseline": _waiver_dict(ctx, waivers, {ctx.me}),
        "weeks": lg.weeks, "scoring": ctx.scoring, "as_of": ctx.as_of, "notes": ctx.notes,
        "generated_at": t0.isoformat(),
        "payload_version": PAYLOAD_VERSION,
        "seconds": round((datetime.now(UTC) - t0).total_seconds(), 1),
    }  # fmt: skip


def _moves(side: dict, picked: frozenset[str], who: str) -> dict:
    """Drops and adds, with the waiver baseline's pickups split out: 'dropping' one only
    means the pickup no longer fits (Oct 27: Gadsden showed as dropped from a roster he
    was never on)."""
    drops = [d for d in side["drops"] if d["key"] not in picked]
    skipped = [{"name": d["name"], "position": d["position"], "points": d["points"],
                "note": "a waiver pickup the trade leaves no room for (not a player you "
                        "own)" if who == "you" else "a waiver pickup the trade leaves them "
                        "no room for"} for d in side["drops"] if d["key"] in picked]  # fmt: skip
    return {f"drops_{who}": drops, f"adds_{who}": side["adds"], f"skipped_pickups_{who}": skipped}


def _seen(lg: core.League, give, get, looks: float) -> dict:
    """What the other manager sees, per player, and the verdict on it."""

    def card(k: str) -> dict:
        p = lg.players[k]
        return {"name": p.name, "position": p.position, "ppg_so_far": p.ppg,
                "last3": p.last3, "season_rank": p.season_rank, "draft_pick": p.draft_pick,
                "injury": p.injury,
                **({"injury_context": p.injury_context} if p.injury_context else {}),
                "perceived_per_game": round(packages.perceived(p), 1)}  # fmt: skip

    return {"they_get": [card(k) for k in give], "they_give": [card(k) for k in get],
            "looks_for_them": round(looks, 2),
            "reads_as": "better for them" if looks > packages.LOOKS_EVEN
            else "even" if looks >= -packages.LOOKS_EVEN else "worse for them"}  # fmt: skip


DATASET = "trade_packages"  # precomputed daily after rest of season
# Bumped when an offer gains fields the tools rely on: a saved run in an older format is
# recomputed instead of served (Oct 28: a board saved before title odds existed showed
# none until the next daily run).
PAYLOAD_VERSION = 5  # 3: stakes; 4: keys, usable value (cards); 5: best-first ranking


def save_all(warehouse_dir, season: int, week: int, payloads: dict[str, dict],
             now: datetime | None = None) -> None:  # fmt: skip
    """Every league's run in one write (a week's latest write is what's read back)."""
    import json

    import polars as pl

    from ffa.warehouse import store

    texts = [json.dumps(p, default=str) for p in payloads.values()]
    df = pl.DataFrame({"league": list(payloads), "season": [season] * len(payloads),
                       "week": [week] * len(payloads), "payload": texts})  # fmt: skip
    store.write(df, warehouse_dir, DATASET, volatile=True, ingested_at=now or datetime.now(UTC))


def saved_packages(con, league_id: str, season: int, week: int) -> dict | None:
    """The latest precomputed run for this league and week, or None."""
    import json

    from ffa.warehouse import duck

    try:
        t = duck.week(con, DATASET, season, week)
    except duck.MissingPartition:
        return None
    rows = con.sql(f"select payload, _ingested_at from {t} where league = ? "
                   "order by _ingested_at desc limit 1", params=[league_id]).fetchall()  # fmt: skip
    if not rows:
        return None
    payload = json.loads(rows[0][0])
    if payload.get("payload_version", 1) < PAYLOAD_VERSION:
        return None  # an older format: recompute
    return payload | {"precomputed_at": str(rows[0][1])}


def trade_cards(con, db, row: LeagueRow, rules: ScoringRules, season: int, week: int,
                position: str | None = None, top: int = 8, refresh: bool = False,
                now: datetime | None = None, fetch=None) -> dict:  # fmt: skip
    """Trade cards (Oct 29): the package generator's offers seen from the players you'd
    get, best target first. Built from the saved trade_packages run (or a fresh one)."""
    from ffa.scoring import score
    from ffa.trade import cards

    payload = None if refresh else saved_packages(con, row.id, season, week)
    if payload is None:
        payload = trade_packages(con, db, row, rules, season, week, now=now, fetch=fetch)
    ctx = context.build(con, db, row, rules, season, week, now=now, fetch=fetch)
    lg = ctx.league
    owners = {k: t.name for tid, t in lg.teams.items() if tid != ctx.me for k in t.roster}
    notes = list(payload.get("notes", []))
    try:
        pg = score(con.sql(f"select * from player_games where season between "
                           f"{int(season) - 1} and {int(season)}").pl(), rules)  # fmt: skip
        tg = con.sql(f"select * from team_games where season = {int(season)}").pl()
    except Exception as exc:  # noqa: BLE001  (said in the notes, never silently)
        notes.append(f"Game logs unavailable ({type(exc).__name__}): cards leave out schedule "
                     "rank, last box score and recent usage.")  # fmt: skip
        pg = pl.DataFrame(schema={"key": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
                                  "position": pl.Utf8, "fpts": pl.Float64})  # fmt: skip
        tg = pl.DataFrame(schema={"season": pl.Int32, "week": pl.Int32, "team": pl.Utf8,
                                  "opponent": pl.Utf8, "game_type": pl.Utf8})  # fmt: skip
    built = cards.build(lg, payload.get("league_wide", []), pg, tg, season, week,
                        owners=owners, last_week=max(lg.weeks) if lg.weeks else week,
                        position=position.upper() if position else None, top=top)  # fmt: skip
    return {
        "cards": built,
        "how_to_read": CARDS_NOTE,
        "stakes": payload.get("stakes"),
        "as_of": payload.get("as_of"), "precomputed_at": payload.get("precomputed_at"),
        "generated_at": payload.get("generated_at"), "notes": notes,
    }  # fmt: skip


CARDS_NOTE = (
    "One card per player you could get, best first: his best deal graded Likely first, "
    "then the rest, each by your title-odds gain. deals: offers that bring him in; could_say_yes: those graded Likely or Fair. "
    "confidence: 1-5 from p_ahead, your chance of coming out ahead over the rest of the "
    "season (5: 90%+, 4: 75-90%, 3: 60-75%, 2: 50-60%, 1: under 50%). acceptance: the "
    "grade in words (likely, plausible, a stretch, a long shot). playoff_points_per_week: "
    "your lineup change a week in the fantasy playoff weeks. usable_value_change_pct: "
    "change in your starters' points a week above the best free agent at each position. "
    "Each deal is an alternative scored on its own."
)


COMPACT_TOP = 12


def compact(payload: dict, top: int | None = COMPACT_TOP) -> dict:
    """A short form for chat clients (Oct 27: the full run was 220K characters and got
    cut off): the best `top` offers league-wide, each with what's needed to decide (the
    trade, both sides' change, the grade and its parts, 2 reasons a side, your playoff
    odds), and no per-team copy."""

    def pct(v: dict | None) -> str | None:
        """ "before -> after (change)" in percent: the change written out, so a chat client
        states it (Oct 28: offers showed before/after only, and it never reached the
        user)."""
        if not v or v.get("before") is None or v.get("after") is None:
            return None
        b, a = 100 * v["before"], 100 * v["after"]
        return f"{b:.1f}% -> {a:.1f}% ({a - b:+.1f} pts)"

    def side(x: dict, who: str) -> dict:
        """One team's gain or loss, grouped (Oct 28: show both sides of every trade)."""
        return {
            "per_week": x[f"{who}_per_week"],
            "playoff_odds": pct((x.get("playoff_odds") or {}).get(who)),
            "title_odds": pct((x.get("title_odds") or {}).get(who)),
        }

    def short(x: dict) -> dict:
        return {
            "grade": x["grade"], "grade_note": x.get("grade_note"),
            "partner": x["partner"], "give": x["give"], "get": x["get"],
            "you": side(x, "you"), "them": side(x, "them"),
            "acceptance": x.get("acceptance"),
            "reads_as": (x.get("what_they_see") or {}).get("reads_as"),
            "reasons_you": x["reasons_you"][:2], "reasons_them": x["reasons_them"][:2],
            "shape": x.get("shape"),
            "cuts_them": [d["name"] for d in x.get("drops_them", [])],
            "drops_you": [d["name"] for d in x.get("drops_you", [])],
            "adds": [a["name"] for a in x.get("adds_you", [])],
            "skipped_pickups_you": [d["name"] for d in x.get("skipped_pickups_you", [])],
        }  # fmt: skip

    flat = payload.get("league_wide", [])
    keep = flat[:top] if top else flat
    out = {k: v for k, v in payload.items() if k not in ("by_team", "league_wide")}
    return out | {
        "league_wide": [short(x) for x in keep],
        "shown": f"{len(keep)} of {len(flat)} offers (Likely first, by your gain); "
                 "compact=false for every field, top=N for more",
        "teams_with_offers": sorted({x["partner"] for x in flat}),
    }  # fmt: skip
