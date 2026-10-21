"""Score a week from frozen data and write the weekly report (markdown)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ffa.tracking import metrics as m
from ffa.tracking import snapshots

RESULTS = "eval_results"
ORDER = ["QB", "RB", "WR", "TE", "K", "DEF"]


def score_week(root: str | Path, actuals: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """eval_results rows for one week, or empty if it has no snapshot before kickoff."""
    snap = snapshots.scoring_snapshot(snapshots.all_snapshots(root, season, week))
    if snap.height == 0:
        return pl.DataFrame()
    rows = m.week_rows(snap, actuals, week)
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


def markdown(results: pl.DataFrame, season: int, week: int) -> str:
    """Report for `week`: that week, rolling 4 weeks, season to date."""
    out = [f"# Projection accuracy — {season} week {week}", "",
           "Scored against projections frozen before each player's kickoff. MAE is on players "
           "who took a snap (projection if active); 'MAE incl. zeros' scores the expected "
           "points the lineup uses, inactive players counted at 0. Baselines use only games "
           "before the week. Targets: model beats both baselines at every position; p10–p90 "
           "coverage 75–85%, p25–p75 about 50%. K and DEF are reported without targets.",
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
    return "\n".join(out)
