"""Each injured or returning player's injury story this season, in one line.

A designation alone misleads: "Questionable (knee - ACL)" reads as a fresh, serious injury
when it is last season's ACL on a player who has practiced for a month and is about to
return (Oct 28: Charbonnet). The story comes from data already stored: this season's
official reports (designation and practice by week), the weekly rosters (weeks on a
reserve list), his games played, and Sleeper's current entry (designation, practice,
note, listing). Nothing here is news text, so nothing says when a player will return;
it says what the reports show and which way they are moving.
"""

from __future__ import annotations

import polars as pl

SEVERITY = {"Questionable": 1, "Doubtful": 2, "Out": 3}
PRACTICE_WORD = {
    "Did Not Participate In Practice": "none", "Limited Participation in Practice": "limited",
    "Full Participation in Practice": "full",
}  # fmt: skip
BYE_GAP = 2  # report weeks this far apart still count as one stretch (a bye between)
MIN_PREV_GAMES = 2
NOTE_CHARS = 200


def _spans(weeks: list[int]) -> str:
    """[1, 2, 3, 6, 7] -> 'wk 1–3, 6–7'."""
    if not weeks:
        return ""
    out, start, prev = [], weeks[0], weeks[0]
    for w in weeks[1:] + [None]:
        if w is not None and w == prev + 1:
            prev = w
            continue
        out.append(f"{start}" if start == prev else f"{start}–{prev}")
        if w is not None:
            start = prev = w
    return "wk " + ", ".join(out)


def _runs(pairs: list[tuple[int, str]]) -> str:
    """[(4, 'Out'), (5, 'Out'), (6, 'Questionable')] -> 'Out wk 4–5, Questionable wk 6'."""
    out: list[str] = []
    i = 0
    while i < len(pairs):
        label, weeks = pairs[i][1], [pairs[i][0]]
        while i + 1 < len(pairs) and pairs[i + 1][1] == label:
            i += 1
            weeks.append(pairs[i][0])
        out.append(f"{label} {_spans(weeks)}")
        i += 1
    return ", ".join(out)


def _stretch(rows: list[dict], week: int) -> list[dict]:
    """His latest unbroken stretch of report entries (byes allowed), if it reaches the
    last couple of weeks."""
    rows = sorted(rows, key=lambda r: r["week"])
    if not rows or rows[-1]["week"] < week - BYE_GAP:
        return []
    keep = [rows[-1]]
    for r in reversed(rows[:-1]):
        if keep[0]["week"] - r["week"] > BYE_GAP:
            break
        keep.insert(0, r)
    return keep


def story(*, week: int, reports: list[dict], reserve_weeks: list[int],
          latest_roster: str | None, games: int, prev_games: int | None,
          prev_snap: float | None, current: dict) -> tuple[str | None, bool]:  # fmt: skip
    """(the line, returning). reports: this season's official report entries before
    `week` ({week, status, practice}); reserve_weeks: weeks on a reserve list this
    season; games: games played this season; prev_*: last season; current: this week's
    combined entry (injury_status, practice, injury, designation, note, listing,
    injury_source)."""
    parts: list[str] = []
    run = _stretch(reports, week)
    # Early on everyone has no games yet; from week 3 none means he has been missing.
    returning = week >= 3 and games == 0 and (prev_games or 0) >= MIN_PREV_GAMES
    cur = current.get("injury_status")
    if not (run or reserve_weeks or cur or returning or current.get("note")):
        return None, False
    first = min([*reserve_weeks, *(r["week"] for r in run)], default=None)
    if reserve_weeks:
        where = ("still there" if latest_roster == "RES"
                 else "since back on the active roster")  # fmt: skip
        parts.append(f"On a reserve list (IR/PUP) {_spans(sorted(reserve_weeks))}, {where}")
    if run:
        st = [(r["week"], r["status"]) for r in run if r.get("status")]
        pr = [(r["week"], PRACTICE_WORD.get(r.get("practice") or "", ""))
              for r in run if r.get("practice")]  # fmt: skip
        bits = []
        if st:
            bits.append(_runs(st))
        if pr:
            bits.append(f"practice: {_runs(pr)}")
        if bits:
            parts.append("Official reports: " + "; ".join(bits))
    if cur or current.get("practice"):
        p = PRACTICE_WORD.get(current.get("practice") or "")
        src = f" [{current['injury_source']}]" if current.get("injury_source") else ""
        parts.append(f"This week: {cur or 'no designation'}"
                     + (f", practice {p}" if p else "") + src)  # fmt: skip
    # Which way it's moving.
    last = next((r for r in reversed(run) if r.get("status")), None)
    practicing = [
        r
        for r in run
        if r.get("practice")
        in ("Limited Participation in Practice", "Full Participation in Practice")
    ]
    if last and cur is None and latest_roster != "RES":
        parts.append(
            f"Trend: no designation so far this week after {last['status']} in week "
            f"{last['week']} (reports come out Wednesday to Friday)"
        )
    elif last and cur and SEVERITY.get(cur, 0) < SEVERITY.get(last["status"], 0):
        parts.append(
            f"Trend: improving ({last['status']} → {cur})"
            + (", and practicing" if practicing else "")
            + ": on track to return"
        )
    elif last and cur and SEVERITY.get(cur, 0) > SEVERITY.get(last["status"], 0):
        parts.append(f"Trend: worsening ({last['status']} → {cur})")
    elif cur == "Out" and practicing and practicing[-1]["week"] >= week - BYE_GAP:
        parts.append("Trend: still listed Out but practicing: working back")
    elif cur == "Out" and run and not practicing:
        parts.append("Trend: not practicing")
    if first == 1:
        parts.append("Listed since week 1: the injury predates this season, it is not new")
    elif first is not None:
        parts.append(f"First listed week {first}")
    if current.get("injury"):
        parts.append(f"Injury on file: {current['injury'].lower()}")
    if returning:
        parts.append(f"No games yet this season; last season {prev_snap * 100:.0f}% of snaps "
                     f"in {prev_games} games (his role for the weeks ahead)")  # fmt: skip
    if current.get("listing"):
        parts.append(f"Sleeper lists him as {current['listing']}")
    if current.get("note"):
        note = " ".join(str(current["note"]).split())
        parts.append(f'Sleeper note: "{note[:NOTE_CHARS]}{"…" if len(note) > NOTE_CHARS else ""}"')
    return "; ".join(parts), returning


def contexts(*, season: int, week: int, keys: list[str], injuries: pl.DataFrame,
             rosters: pl.DataFrame, player_games: pl.DataFrame,
             current: pl.DataFrame) -> pl.DataFrame:  # fmt: skip
    """key -> injury_context (None when there is nothing to say), returning."""
    want = set(keys)
    inj = injuries.filter((pl.col("season") == season) & (pl.col("week") < week)
                          & pl.col("gsis_id").is_in(list(want)))  # fmt: skip
    reports: dict[str, list[dict]] = {}
    if inj.height:
        rs = pl.col("report_status").replace({"Probable": None, "Note": None})
        pr = (pl.col("practice_status") if "practice_status" in inj.columns
              else pl.lit(None, pl.Utf8))  # fmt: skip
        rows = inj.select(
            pl.col("gsis_id").alias("key"), "week", rs.alias("status"), pr.alias("practice")
        ).unique(["key", "week"], keep="last")
        for r in rows.iter_rows(named=True):
            if r["status"] or r["practice"]:
                reports.setdefault(r["key"], []).append(r)
    ro = rosters.filter((pl.col("season") == season) & (pl.col("week") <= week)
                        & pl.col("gsis_id").is_in(list(want)))  # fmt: skip
    reserve: dict[str, list[int]] = {}
    latest: dict[str, str] = {}
    for k, w, s in ro.sort("week").select("gsis_id", "week", "status").iter_rows():
        latest[k] = s
        if s == "RES":
            reserve.setdefault(k, []).append(int(w))
    played = (pl.col("played").fill_null(False) if "played" in player_games.columns
              else pl.col("snaps").fill_null(0) > 0)  # fmt: skip
    pg = player_games.filter(played & pl.col("key").is_in(list(want)))
    games = dict(pg.filter((pl.col("season") == season) & (pl.col("week") < week))
                 .group_by("key").len().iter_rows())  # fmt: skip
    prev = {r[0]: (r[1], r[2]) for r in pg.filter(pl.col("season") == season - 1)
            .group_by("key").agg(pl.len(), pl.col("snap_pct").fill_null(0).mean())
            .iter_rows()}  # fmt: skip
    cur = {r["key"]: r for r in current.iter_rows(named=True)} if current.height else {}
    out = []
    for k in keys:
        pg_, ps = prev.get(k, (None, None))
        line, ret = story(week=week, reports=reports.get(k, []), reserve_weeks=reserve.get(k, []),
                          latest_roster=latest.get(k), games=games.get(k, 0), prev_games=pg_,
                          prev_snap=ps, current=cur.get(k, {}))  # fmt: skip
        out.append({"key": k, "injury_context": line, "returning": ret})
    return pl.DataFrame(out, schema={"key": pl.Utf8, "injury_context": pl.Utf8,
                                     "returning": pl.Boolean})  # fmt: skip
