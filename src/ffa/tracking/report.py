"""Score a week from frozen data and write the weekly report (markdown)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ffa.tracking import benchmarks, snapshots
from ffa.tracking import metrics as m

RESULTS = "eval_results"
ORDER = ["QB", "RB", "WR", "TE", "K", "DEF"]


def score_week(root: str | Path, actuals: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """eval_results rows for one week, or empty if it has no snapshot before kickoff."""
    snap = snapshots.scoring_snapshot(snapshots.all_snapshots(root, season, week))
    if snap.height == 0:
        return pl.DataFrame()
    bench = snapshots.scoring_snapshot(benchmarks.all_benchmarks(root, season, week))
    rows = m.week_rows(snap, actuals, week, bench)
    versions = snap["model_version"].unique().sort().to_list()
    return m.metrics(rows).with_columns(
        pl.lit(season).cast(pl.Int32).alias("season"),
        pl.lit(week).cast(pl.Int32).alias("week"),
        pl.lit(",".join(versions)).alias("model_version"),
        pl.lit(snap.height).alias("players_scored"),
    )


def _fmt(v) -> str:
    return "–" if v is None else f"{v:.2f}"


def _table(p: pl.DataFrame, league: str, scope: str) -> list[str]:
    sub = p.filter((pl.col("league") == league) & (pl.col("scope") == scope))
    if sub.height == 0:
        return ["_No scored players._", ""]

    def get(pos, source, metric):
        r = sub.filter((pl.col("position") == pos) & (pl.col("source") == source)
                       & (pl.col("metric") == metric))  # fmt: skip
        return (r["value"][0], r["n"][0]) if r.height else (None, 0)

    lines = [
        "| Pos | n | Model MAE | vs last-4: model / baseline | vs season avg: model / baseline "
        "| Bias | p10–p90 cover | p25–p75 cover | MAE incl. zeros |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pos in ORDER:
        mae, n = get(pos, "model", "mae_if_active")
        if not n:
            continue
        cells = [pos, str(n), _fmt(mae)]
        for base in ("last4", "season_avg"):
            mv, _ = get(pos, f"model_vs_{base}", "mae_if_active")
            bv, bn = get(pos, base, "mae_if_active")
            win = "" if mv is None or bv is None else (" ✓" if mv < bv else " ✗")
            cells.append(f"{_fmt(mv)} / {_fmt(bv)}{win} (n={bn})")
        cells += [_fmt(get(pos, "model", "bias")[0]),
                  f"{(get(pos, 'model', 'coverage_80')[0] or 0) * 100:.0f}%",
                  f"{(get(pos, 'model', 'coverage_50')[0] or 0) * 100:.0f}%",
                  _fmt(get(pos, "model", "mae_expected")[0])]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def _sleeper_table(p: pl.DataFrame, league: str, scope: str) -> list[str]:
    sub = p.filter((pl.col("league") == league) & (pl.col("scope") == scope))

    def get(pos, source, metric):
        r = sub.filter((pl.col("position") == pos) & (pl.col("source") == source)
                       & (pl.col("metric") == metric))  # fmt: skip
        return (r["value"][0], r["n"][0]) if r.height else (None, 0)

    lines = ["| Pos | MAE incl. zeros: model / Sleeper | MAE if active: model / Sleeper "
             "| Rank corr.: model / Sleeper | Start/sit (close pairs): model / Sleeper |",
             "| --- | --- | --- | --- | --- |"]  # fmt: skip
    any_row = False
    for pos in ORDER:
        cells = [pos]
        for metric, lower in (("mae_expected", True), ("mae_if_active", True),
                              ("spearman", False), ("start_sit", False)):  # fmt: skip
            mv, n = get(pos, "model_vs_sleeper", metric)
            sv, _ = get(pos, "sleeper", metric)
            if mv is None or sv is None:
                cells.append("–")
                continue
            any_row = True
            win = " ✓" if (mv < sv if lower else mv > sv) else " ✗"
            pct = metric == "start_sit"
            f = (lambda v: f"{v * 100:.1f}%") if pct else _fmt
            cells.append(f"{f(mv)} / {f(sv)}{win} (n={n})")
        lines.append("| " + " | ".join(cells) + " |")
    if not any_row:
        return ["_No Sleeper benchmark captured before kickoff for these weeks._", ""]
    return lines + [""]


def _misses_table(p: pl.DataFrame, league: str) -> list[str]:
    sub = p.filter((pl.col("league") == league) & (pl.col("scope") == "all")
                   & (pl.col("source") == "misses"))  # fmt: skip
    if sub.height == 0:
        return []
    lines = ["| Pos | n | Availability misses | Early exits | Model (scored) "
             "| Sat (as projected, or no role) |",
             "| --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for pos in ORDER:
        r = {x["metric"]: x for x in sub.filter(pl.col("position") == pos).iter_rows(named=True)}
        if not r:
            continue
        n = next(iter(r.values()))["n"]
        cell = lambda k, r=r: f"{(r[k]['value'] or 0) * 100:.1f}%" if k in r else "–"  # noqa: E731
        lines.append(f"| {pos} | {n} | {cell('availability_miss')} | {cell('early_exit')} | "
                     f"{cell('model')} | {cell('not_expected')} |")  # fmt: skip
    return lines + [""]


def _availability_table(p: pl.DataFrame, league: str) -> list[str]:
    sub = p.filter((pl.col("league") == league)
                   & pl.col("source").str.starts_with("availability:"))  # fmt: skip
    if sub.height == 0:
        return []
    lines = [
        "| Status | n | Said (avg chance to play) | Happened (played) | Brier |",
        "| --- | --- | --- | --- | --- |",
    ]
    order = ["None", "Questionable", "Doubtful", "Out"]
    stats = sorted(sub["source"].unique().to_list(),
                   key=lambda x: (order.index(x.split(":", 1)[1])
                                  if x.split(":", 1)[1] in order else 9, x))  # fmt: skip
    for src in stats:
        r = {x["metric"]: x for x in sub.filter(pl.col("source") == src).iter_rows(named=True)}
        label = src.split(":", 1)[1].replace("None", "No designation")
        n = r["brier"]["n"] if "brier" in r else 0
        lines.append(
            f"| {label} | {n} | {r['said']['value'] * 100:.1f}% | "
            f"{r['happened']['value'] * 100:.1f}% | {r['brier']['value']:.3f} |"
        )
    return lines + [""]


def markdown(results: pl.DataFrame, season: int, week: int) -> str:
    """Report for `week`: that week, rolling 4 weeks, season to date."""
    out = [f"# Projection accuracy — {season} week {week}", "",
           "Scored against projections frozen before each player's kickoff. MAE is on players "
           "who took a snap (projection if active); 'MAE incl. zeros' scores the expected "
           "points the lineup uses, inactive players counted at 0. Baselines use only games "
           "before the week. Targets: model beats both baselines at every position; p10–p90 "
           "coverage 75–85%, p25–p75 about 50%. K and DEF are reported without targets. "
           "MAE if active, bias and coverage exclude availability misses (given 50%+ to play, "
           "didn't) and early exits (under 25% of usual snaps), which are reported "
           "separately. Sleeper's projections were frozen at the same moments as ours; "
           "start/sit pairs are same-position players within 3 points by season average.",
           ""]  # fmt: skip
    versions = sorted(set(",".join(results.filter(pl.col("week") == week)["model_version"]
                                   .to_list()).split(",")) - {""})  # fmt: skip
    out += [f"Model version(s): {', '.join(versions) or 'n/a'}", ""]
    windows = [("This week", [week]), ("Last 4 weeks", list(range(week - 3, week + 1))),
               ("Season to date", list(range(1, week + 1)))]  # fmt: skip
    for league in sorted(results["league"].unique().to_list()):
        out += [f"## {'Standard half-PPR' if league == 'default' else league}", ""]
        for title, weeks in windows:
            p = m.pooled(results.filter(pl.col("week").is_in(weeks)))
            got = sorted(results.filter(pl.col("week").is_in(weeks))["week"].unique().to_list())
            out += [f"### {title} (weeks scored: {', '.join(map(str, got)) or 'none'})", ""]
            scopes = (
                ("all", "All projected players"),
                ("relevant", "Fantasy-relevant (top 12 QB/TE, 36 RB/WR)"),
            )
            for scope, label in scopes:
                out += [f"**{label}**", ""] + _table(p, league, scope)
                out += [f"*Against Sleeper ({label.lower()}), same players, same scoring:*",
                        ""] + _sleeper_table(p, league, scope)  # fmt: skip
            out += [
                "**Where the misses came from** (all projected players; only 'Model' "
                "rows are scored above)",
                "",
            ] + _misses_table(p, league)
            out += [
                "**Chance to play: said vs happened** (players projected 3+ points if active)",
                "",
            ] + _availability_table(p, league)
    return "\n".join(out)
