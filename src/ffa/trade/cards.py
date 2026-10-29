"""Trade cards: the package generator's offers seen from the players you'd get (Oct 29).

One card per target: who he is and whose roster he's on, how many offers bring him in and
how many the other manager could say yes to, a one-line case (role, schedule left, last
box score), what to watch (injury story, usage against his norm, price tier), and his
best deal: what you give and get, a 1-5 confidence from your chance of coming out ahead,
the acceptance grade, your title odds before and after, your lineup change a week in the
playoff weeks, and the change in your usable value (starters above replacement).

Every number comes from the model or the league's own data; the words come from fixed
rules over those numbers, so nothing on a card is a guess.
"""

from __future__ import annotations

import math

import polars as pl

from ffa.trade import core

GRADE_ORDER = {"Likely": 0, "Fair": 1, "Stretch": 2, "Long shot": 3}
SAYS_YES = {"Likely", "Fair"}
ACCEPT_WORD = {"Likely": "likely", "Fair": "plausible", "Stretch": "a stretch",
               "Long shot": "a long shot"}  # fmt: skip
# Your chance of coming out ahead over the rest of the season -> confidence 1-5.
CONFIDENCE = ((0.90, 5, "confident"), (0.75, 4, "favorable"), (0.60, 3, "lean"),
              (0.50, 2, "coin flip"), (0.0, 1, "risky"))  # fmt: skip
TIER_SIZE = 12  # QB1 / RB1 / WR1 / TE1 = top 12 at the position, then 2, 3...
USAGE_GAP = 0.15  # recent use this far off his norm is worth a line
MIN_BASE_GAMES = 4
HARD_SCHEDULE = 0.75  # in the bottom quarter of schedules: worth a line
# Half-PPR-style weights only to rank schedules (the league's own scoring isn't needed
# to say which defenses give up the most to a position).
_PTS = {"pass_yd": 0.04, "pass_td": 4, "pass_int": -1, "rush_yd": 0.1, "rush_td": 6,
        "rec": 0.5, "rec_yd": 0.1, "rec_td": 6, "fum_lost": -2}  # fmt: skip


def confidence(p_ahead: float | None) -> dict:
    p = float(p_ahead or 0.0)
    for cut, level, label in CONFIDENCE:
        if p >= cut:
            return {"level": level, "label": label, "p_ahead": round(p, 3)}
    return {"level": 1, "label": "risky", "p_ahead": round(p, 3)}


def tier(rank: int | None) -> int | None:
    return None if not rank else math.ceil(rank / TIER_SIZE)


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


Ranks = dict[tuple[str, str], tuple[int, int]]


def schedule_ranks(player_games: pl.DataFrame, team_games: pl.DataFrame, season: int,
                   week: int, last_week: int) -> Ranks:  # fmt: skip
    """(NFL team, position) -> (rank, of): his team's remaining schedule (weeks `week` to
    `last_week`) by the points its opponents have allowed to the position per game this
    season; 1 = easiest."""
    pg = player_games.filter((pl.col("season") == season) & (pl.col("week") < week))
    cols = [c for c in _PTS if c in pg.columns]
    if pg.height == 0 or "opponent" not in pg.columns or not cols:
        return {}
    pts = sum((pl.col(c).fill_null(0) * w for c, w in _PTS.items() if c in cols), pl.lit(0.0))
    allowed = (
        pg.with_columns(pts.alias("_p"))
        .group_by("opponent", "position", "week").agg(pl.col("_p").sum())
        .group_by("opponent", "position").agg(pl.col("_p").mean().alias("allowed"))
    )  # fmt: skip
    games = team_games.filter((pl.col("season") == season) & (pl.col("week") >= week)
                              & (pl.col("week") <= last_week)
                              & (pl.col("game_type") == "REG"))  # fmt: skip
    games = games.select("team", "opponent").drop_nulls()
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        a = allowed.filter(pl.col("position") == pos).select("opponent", "allowed")
        sos = (games.join(a, on="opponent", how="inner").group_by("team")
               .agg(pl.col("allowed").mean()).sort("allowed", descending=True))  # fmt: skip
        n = sos.height
        for i, t in enumerate(sos["team"].to_list(), 1):
            out[(t, pos)] = (i, n)
    return out


def usage(player_games: pl.DataFrame, key: str, position: str, season: int,
          week: int) -> dict | None:  # fmt: skip
    """His last two games' looks (targets; for a RB, carries + targets) against his norm:
    last season a game (4+ games), else this season before those two."""
    if position not in ("RB", "WR", "TE"):
        return None
    look = (pl.col("targets").fill_null(0) + pl.col("designed_carries").fill_null(0)
            if position == "RB" else pl.col("targets").fill_null(0))  # fmt: skip
    word = "carries + targets" if position == "RB" else "targets"
    played = (pl.col("played").fill_null(False) if "played" in player_games.columns
              else pl.lit(True))  # fmt: skip
    g = (player_games.filter((pl.col("key") == key) & played)
         .with_columns(look.alias("_l")).sort("season", "week"))  # fmt: skip
    cur = g.filter((pl.col("season") == season) & (pl.col("week") < week))
    if cur.height < 2:
        return None
    last2 = [int(x) for x in cur["_l"].tail(2).to_list()]
    prev = g.filter(pl.col("season") == season - 1)
    if prev.height >= MIN_BASE_GAMES:
        base, basis = float(prev["_l"].mean()), "last season"
    elif cur.height - 2 >= 2:
        base, basis = float(cur["_l"].head(cur.height - 2).mean()), "earlier this season"
    else:
        return None
    return {"last_two": last2, "per_game_norm": round(base, 1), "norm": basis, "word": word}


def watch(p: core.Player, use: dict | None, sched: tuple[int, int] | None) -> list[str]:
    """What to check before offering, each with the number it rests on."""
    out = []
    if p.injury or p.injury_context:
        head = f"{p.injury}" if p.injury else "Injury history"
        story = "; ".join((p.injury_context or "").split("; ")[:3])
        out.append(f"{head}: {story}" if story else f"{head}: check this week's report")
    if use:
        recent = sum(use["last_two"]) / 2
        a, b = use["last_two"]
        what = (f"{a} {use['word']} in each of his last two games" if a == b
                else f"{a} and {b} {use['word']} in his last two games")  # fmt: skip
        if recent < (1 - USAGE_GAP) * use["per_game_norm"]:
            out.append(f"{what}, under the {use['per_game_norm']} a game he averaged "
                       f"{use['norm']}")  # fmt: skip
        elif recent > (1 + USAGE_GAP) * use["per_game_norm"]:
            out.append(f"{what}, up from the {use['per_game_norm']} a game he averaged "
                       f"{use['norm']}")  # fmt: skip
    vt, pt = tier(p.ros_rank), tier(p.season_rank)
    if vt and pt and vt != pt:
        pos = p.position
        if vt > pt:
            out.append(
                f"Ranked {pos}{p.season_rank} so far but projects {pos}{p.ros_rank} a "
                f"game the rest of the way: pay a {pos}{vt} price, not a {pos}{pt} one"
            )
        else:
            out.append(
                f"Ranked {pos}{p.season_rank} so far but projects {pos}{p.ros_rank} a "
                f"game the rest of the way: a {pos}{pt} price for {pos}{vt} production"
            )
    if sched and sched[0] > HARD_SCHEDULE * sched[1]:
        out.append(f"One of the hardest {p.position} schedules left ({ordinal(sched[0])} of "
                   f"{sched[1]})")  # fmt: skip
    return out


def summary(p: core.Player, sched: tuple[int, int] | None, last_game: float | None) -> str:
    bits = []
    t = tier(p.ros_rank)
    if t:
        bits.append(f"{p.position}{t} role ({p.position}{p.ros_rank} by projected points a "
                    "game)")  # fmt: skip
    if sched:
        r, n = sched
        where = "easiest" if r == 1 else f"{ordinal(r)}-easiest"
        bits.append(f"the {where} {p.position} schedule left (of {n})")
    line = " on ".join(bits[:2]) if len(bits) == 2 else (bits[0] if bits else p.name)
    if last_game is not None:
        line += f", and his last box score was {last_game:.1f}"
    return line[0].upper() + line[1:] + "."


def _side_players(lg: core.League, keys: list[str]) -> list[dict]:
    return [{"name": lg.players[k].name, "position": lg.players[k].position}
            for k in keys if k in lg.players]  # fmt: skip


def _pct(v: dict | None) -> str | None:
    if not v or v.get("before") is None or v.get("after") is None:
        return None
    return f"{100 * v['before']:.1f}% -> {100 * v['after']:.1f}%"


def deal(lg: core.League, x: dict, n_playoff_weeks: int) -> dict:
    """One offer as it sits on a card."""
    title = (x.get("title_odds") or {}).get("you")
    playoffs = (x.get("playoff_odds") or {}).get("you")
    them_title = (x.get("title_odds") or {}).get("them")
    uv = (x.get("usable_value") or {}).get("change_pct")
    pw = x.get("you_playoff_weeks")
    return {
        "confidence": confidence(x.get("p_ahead")),
        "grade": x["grade"], "acceptance": ACCEPT_WORD.get(x["grade"], x["grade"].lower()),
        "partner": x["partner"],
        "you_give": _side_players(lg, x["give_keys"]),
        "you_get": _side_players(lg, x["get_keys"]),
        "title_odds": _pct(title),
        "title_odds_change_pts": None if not title else round(100 * title["change"], 1),
        "playoff_odds": _pct(playoffs),
        "playoff_points_per_week": (round(pw / n_playoff_weeks, 1)
                                    if pw is not None and n_playoff_weeks else None),
        "usable_value_change_pct": uv,
        "you_per_week": x["you_per_week"], "them_per_week": x["them_per_week"],
        "their_title_odds": _pct(them_title),
    }  # fmt: skip


def _deal_key(x: dict) -> tuple:
    title = (x.get("title_odds") or {}).get("you") or {}
    return (GRADE_ORDER.get(x["grade"], 9), -(title.get("change") or 0.0), -x["you_per_week"])


def build(lg: core.League, offers: list[dict], player_games: pl.DataFrame,
          team_games: pl.DataFrame, season: int, week: int, *,
          owners: dict[str, str], last_week: int, position: str | None = None,
          top: int = 8) -> list[dict]:  # fmt: skip
    """Cards for the best `top` targets, best first: ranked by your title-odds gain in
    his best deal (then grade, then your gain a week). offers: trade_packages'
    league_wide items (near misses left out). owners: player key -> team name."""
    by_target: dict[str, list[dict]] = {}
    for x in offers:
        if x["grade"] not in GRADE_ORDER or not x.get("get_keys"):
            continue
        tk = max(x["get_keys"], key=lambda k: lg.players[k].ros(lg.weeks) if k in lg.players
                 else 0.0)  # fmt: skip
        if position and lg.players[tk].position != position:
            continue
        by_target.setdefault(tk, []).append(x)
    for xs in by_target.values():
        xs.sort(key=_deal_key)

    def rank_key(item):
        _k, xs = item
        title = (xs[0].get("title_odds") or {}).get("you") or {}
        return (-(title.get("change") or 0.0), GRADE_ORDER[xs[0]["grade"]], -xs[0]["you_per_week"])

    ranked = sorted(by_target.items(), key=rank_key)[:top]
    sched = schedule_ranks(player_games, team_games, season, week, last_week)
    played = (pl.col("played").fill_null(False) if "played" in player_games.columns
              else pl.lit(True))  # fmt: skip
    cards = []
    n_po = len(lg.playoff_weeks)
    for i, (k, xs) in enumerate(ranked, 1):
        p = lg.players[k]
        s = sched.get((p.nfl_team or "", p.position))
        last = last_game(player_games.filter(played), k, season, week)
        use = usage(player_games, k, p.position, season, week)
        cards.append({
            "rank": i, "name": p.name, "position": p.position, "nfl_team": p.nfl_team,
            "owner": owners.get(k), "deals": len(xs),
            "could_say_yes": sum(x["grade"] in SAYS_YES for x in xs),
            "summary": summary(p, s, last),
            "watch": watch(p, use, s),
            "facts": {"ros_rank": p.ros_rank, "season_rank": p.season_rank,
                      "ros_per_game": p.per_game, "ppg_so_far": p.ppg, "last_game": last,
                      "schedule_rank": list(s) if s else None, "usage": use,
                      "injury": p.injury, "injury_context": p.injury_context},
            "best_deal": deal(lg, xs[0], n_po),
            "other_deals": [deal(lg, x, n_po) for x in xs[1:3]],
        })  # fmt: skip
    return cards


def last_game(player_games: pl.DataFrame, key: str, season: int, week: int) -> float | None:
    g = player_games.filter((pl.col("key") == key) & (pl.col("season") == season)
                            & (pl.col("week") < week)).sort("week")  # fmt: skip
    if g.height == 0 or "fpts" not in g.columns:
        return None
    return round(float(g["fpts"][-1]), 1)
