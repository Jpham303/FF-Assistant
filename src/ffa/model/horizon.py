"""Chance to play in each coming game, from where a player stands today.

This week's chance to play comes from the injury report (ffa.model.sim_params). For the
games after it, the question is different: how often does a player in his spot today
(healthy, Questionable, Doubtful, Out, inactive, or on a reserve list such as IR, PUP or
NFI) take a snap in his team's k-th game from now? Byes don't count as games. Answered
empirically, by status, role and k, from seasons of rosters, injury reports and snaps.
It covers recovery (an Out player is likelier to be back in three games than in one),
reserve-list minimum stays, and the injuries healthy players will pick up along the way.

Role: "top" if he is a top producer at his position so far (top 12 QB/TE, top 36 RB/WR
by fantasy points per game, 2+ games); else by his average share of offensive snaps in
his earlier games that season: starter (70%+), role (30-70%), depth (under 30%). Top
producers and starters miss far fewer games than part-timers, who also lose snaps to
benchings and depth-chart changes.
"""

from __future__ import annotations

import polars as pl

from ffa.model.sim_params import STARTER_SNAP_PCT

FULL_TIME = 0.70  # average snap share of a starter
TOP_N = {"QB": 12, "TE": 12, "RB": 36, "WR": 36}
# Standard half-PPR points, only to rank producers (independent of any league's rules).
_PTS = {"pass_yd": 0.04, "pass_td": 4, "pass_int": -1, "rush_yd": 0.1, "rush_td": 6,
        "rec": 0.5, "rec_yd": 0.1, "rec_td": 6, "fum_lost": -2}  # fmt: skip


def _ppg(player_games: pl.DataFrame) -> pl.DataFrame:
    """key, season, week -> points per game in his earlier played games that season."""
    cols = [c for c in _PTS if c in player_games.columns]
    played = pl.col("played").fill_null(False) if "played" in player_games.columns else (
        pl.col("snaps").fill_null(0) > 0)  # fmt: skip
    pts = sum((pl.col(c).fill_null(0) * w for c, w in _PTS.items() if c in cols), pl.lit(0.0))
    g = player_games.filter(played).sort("season", "week").with_columns(pts.alias("_pts"))
    return g.with_columns(
        pl.col("_pts").cum_sum().over("key", "season").alias("_cum"),
        pl.int_range(1, pl.len() + 1).over("key", "season").alias("_gp"),
    ).select("key", "season", pl.col("week").cast(pl.Int32), "_cum", "_gp")


MAX_K = 8  # games ahead; further out uses this row
MIN_N = 30
STATUSES = ("healthy", "questionable", "doubtful", "out", "inactive", "reserve")
POSITIONS = ["QB", "RB", "WR", "TE"]


def category(roster_status: str | None, report_status: str | None) -> str:
    """Where a player stands today, from his roster status and this week's report."""
    if roster_status == "RES":
        return "reserve"
    if report_status in ("Out", "Doubtful", "Questionable"):
        return report_status.lower()
    if roster_status == "INA":
        return "inactive"
    return "healthy"


def _standing(rosters: pl.DataFrame, injuries: pl.DataFrame,
              player_games: pl.DataFrame) -> pl.DataFrame:  # fmt: skip
    """One row per (player, season, week): team, category and role at that week."""
    r = rosters.filter(
        pl.col("position").replace({"FB": "RB"}).is_in(POSITIONS)
        & pl.col("status").is_in(["ACT", "INA", "RES"]) & pl.col("gsis_id").is_not_null()
    ).select(
        pl.col("gsis_id").alias("key"), "season", pl.col("week").cast(pl.Int32), "team",
        pl.col("position").replace({"FB": "RB"}), pl.col("status").alias("roster_status"),
    ).unique(["key", "season", "week"])  # fmt: skip
    inj = injuries.filter(pl.col("report_status").is_in(["Out", "Doubtful", "Questionable"]))
    inj = inj.select(
        pl.col("gsis_id").alias("key"), "season", pl.col("week").cast(pl.Int32),
        "report_status",
    ).unique(["key", "season", "week"])  # fmt: skip
    role = (
        player_games.sort("season", "week")
        .with_columns(
            (pl.col("snap_pct").fill_null(0).cum_sum() / pl.col("snap_pct").cum_count())
            .over("key", "season").alias("avg_snap")
        )
        .select("key", "season", pl.col("week").cast(pl.Int32), "avg_snap")
    )  # fmt: skip
    s = r.join(inj, on=["key", "season", "week"], how="left").sort("week")
    s = s.join_asof(role.sort("week"), on="week", by=["key", "season"], strategy="backward",
                    allow_exact_matches=False, check_sortedness=False)  # fmt: skip
    if any(c in player_games.columns for c in _PTS):
        s = s.join_asof(_ppg(player_games).sort("week"), on="week", by=["key", "season"],
                        strategy="backward", allow_exact_matches=False,
                        check_sortedness=False)  # fmt: skip
        ppg = pl.when(pl.col("_gp") >= 2).then(pl.col("_cum") / pl.col("_gp"))
        s = s.with_columns(
            (ppg.rank("ordinal", descending=True).over("season", "week", "position")
             <= pl.col("position").replace_strict(TOP_N, default=0)).fill_null(False).alias("_top")
        ).drop("_cum", "_gp")  # fmt: skip
    else:
        s = s.with_columns(pl.lit(False).alias("_top"))
    return s.with_columns(
        pl.struct("roster_status", "report_status").map_elements(
            lambda x: category(x["roster_status"], x["report_status"]), return_dtype=pl.Utf8
        ).alias("cat"),
        pl.when(pl.col("_top")).then(pl.lit("top"))
        .when(pl.col("avg_snap").fill_null(0) >= FULL_TIME).then(pl.lit("starter"))
        .when(pl.col("avg_snap").fill_null(0) >= STARTER_SNAP_PCT).then(pl.lit("role"))
        .otherwise(pl.lit("depth")).alias("role"),
    )  # fmt: skip


def outcomes(rosters: pl.DataFrame, injuries: pl.DataFrame, player_games: pl.DataFrame,
             team_games: pl.DataFrame) -> pl.DataFrame:  # fmt: skip
    """(cat, role, k, played) for every player-week and each of his team's next MAX_K
    games (regular season, same season)."""
    st = _standing(rosters, injuries, player_games)
    games = (
        team_games.filter(pl.col("game_type") == "REG")
        .select("season", pl.col("week").cast(pl.Int32).alias("g_week"), "team")
        .unique()
    )
    nxt = st.join(games, on=["season", "team"], how="inner").filter(
        pl.col("g_week") > pl.col("week")
    )
    nxt = nxt.with_columns(
        pl.col("g_week").rank("ordinal").over("key", "season", "week").cast(pl.Int32).alias("k")
    ).filter(pl.col("k") <= MAX_K)
    played = (
        player_games.filter(pl.col("played").fill_null(False) | (pl.col("snaps").fill_null(0) > 0))
        .select("key", "season", pl.col("week").cast(pl.Int32).alias("g_week"))
        .unique().with_columns(pl.lit(True).alias("played"))
    )  # fmt: skip
    return (
        nxt.join(played, on=["key", "season", "g_week"], how="left")
        .with_columns(pl.col("played").fill_null(False))
        .select("key", "season", "week", "position", "cat", "role", "k", "played")
    )


def fit(rosters: pl.DataFrame, injuries: pl.DataFrame, player_games: pl.DataFrame,
        team_games: pl.DataFrame) -> dict[str, float]:  # fmt: skip
    """P(plays) by "pos|cat|role|k", pooled as "any|cat|role|k" and "any|cat|any|k".
    Cells under MIN_N rows are left out (lookups fall back to the pooled ones)."""
    o = outcomes(rosters, injuries, player_games, team_games)
    out: dict[str, float] = {}
    for keys in (["position", "cat", "role", "k"], ["cat", "role", "k"], ["cat", "k"]):
        g = o.group_by(keys).agg(pl.len().alias("n"), pl.col("played").mean().alias("p"))
        for r in g.filter(pl.col("n") >= MIN_N).iter_rows(named=True):
            key = f"{r.get('position', 'any')}|{r['cat']}|{r.get('role', 'any')}|{r['k']}"
            out[key] = round(float(r["p"]), 4)
    return out


def p_play(table: dict[str, float], position: str, cat: str, role: str, k: int) -> float:
    """P(plays in his team's k-th game from now), k >= 1."""
    k = min(max(k, 1), MAX_K)
    for key in (f"{position}|{cat}|{role}|{k}", f"any|{cat}|{role}|{k}", f"any|{cat}|any|{k}",
                f"any|healthy|{role}|{k}"):  # fmt: skip
        if key in table:
            return table[key]
    return 1.0
