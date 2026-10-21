"""Command line: `uv run ffa --help`.

ffa leagues sleeper jpham03            list your Sleeper leagues and their IDs
ffa sync sleeper <league_id>           fetch, resolve and store a league; print the report
ffa sync sleeper <league_id> --dry-run same, without writing to the database
ffa check-scoring sleeper <league_id>  our points vs Sleeper's, every completed week
ffa evaluate --season 2025 --scoring s.json   holdout accuracy vs baselines
ffa rebuild-projections --dry-run      stored weeks whose files are in an old format
ffa report [--week N]                  projection accuracy vs baselines, from frozen data
"""

from __future__ import annotations

import argparse
import sys

from ffa.config import settings


def _leagues(args: argparse.Namespace) -> int:
    from ffa.leagues.sleeper import SleeperAdapter

    refs = SleeperAdapter().find_leagues(args.user, args.season)
    if not refs:
        print(f"No {args.season} leagues for {args.user}")
        return 1
    for r in refs:
        print(f"{r.league_id}  {r.name}  ({r.num_teams} teams)")
    return 0


def _sync(args: argparse.Namespace) -> int:
    from ffa.ingest.ids import Resolver
    from ffa.leagues import sync
    from ffa.leagues.sleeper import SleeperAdapter
    from ffa.warehouse import duck

    resolver = Resolver(duck.connect(settings.warehouse_dir).sql("select * from player_ids").pl())
    adapter = SleeperAdapter(
        me=args.me or settings.sleeper_username or None,
        players=sync.load_sleeper_players(settings.warehouse_dir),
    )
    league = sync.resolve(adapter.fetch_league(args.league_id), resolver)
    if not args.dry_run:
        from ffa.db.session import session

        with session() as db:
            sync.store(league, db)
    rep = sync.report(league)

    mine = league.my_team()
    print(f"{league.name} ({league.season}) — {rep.teams} teams, {rep.players} players")
    print(
        f"Lineup: {' '.join(league.lineup_slots)} + {league.bench_slots} BN, {league.ir_slots} IR"
    )
    print(f"Your team: {mine.name if mine else 'not identified (pass --me)'}")
    print(f"Matched {rep.match_rate:.1%}: {rep.by_method}")
    for u in rep.unmatched:
        print(
            f"  UNMATCHED {u.team}: {u.platform_id} {u.name or '?'} {u.position or ''} ({u.reason})"
        )
    if rep.unsupported_slots:
        print(f"Unsupported lineup slots (ignored): {rep.unsupported_slots}")
    print("dry run: nothing written" if args.dry_run else "Saved to database.")
    return 0


def _check_scoring(args: argparse.Namespace) -> int:
    import polars as pl

    from ffa.ingest.ids import Resolver
    from ffa.leagues import sync
    from ffa.leagues.base import Platform
    from ffa.leagues.sleeper import SleeperAdapter
    from ffa.scoring import from_sleeper, stat_lines
    from ffa.scoring.check import awarded_frame, compare, explain
    from ffa.scoring.defense import load_pbp
    from ffa.warehouse import duck

    adapter = SleeperAdapter(players=sync.load_sleeper_players(settings.warehouse_dir))
    league = adapter.fetch_league(args.league_id)
    weeks = args.weeks or list(range(1, adapter.current_week()))
    con = duck.connect(settings.warehouse_dir)

    def q(t: str) -> pl.DataFrame:
        return con.sql(f"select * from {t} where season = {league.season}").pl()

    lines = stat_lines(q("player_stats"), load_pbp(con, league.season), q("schedules"))
    rules = from_sleeper(league.scoring)
    xw = con.sql("select * from player_ids").pl()
    awarded = awarded_frame({w: adapter.fetch_week(args.league_id, w) for w in weeks})
    out, rep = compare(awarded, lines, rules, Resolver(xw), Platform.SLEEPER)

    print(f"{league.name}, weeks {weeks[0]}-{weeks[-1]}")
    print(
        f"Matched {rep.matched}/{rep.compared} player-weeks ({rep.match_rate:.1%}); "
        f"{rep.unresolved} unresolved"
    )
    for key, why in {**rules.unsupported, **rules.approximate}.items():
        print(f"  note: {key} — {why}")
    names = dict(xw.select("gsis_id", "name").iter_rows())
    bad = out.filter(pl.col("status") == "mismatch")
    terms = explain(bad, lines, rules)
    for r in bad.iter_rows(named=True):
        who = names.get(r["key"], r["key"])
        print(f"  MISMATCH wk{r['week']} {who}: Sleeper {r['awarded']}, ours {r['ours']}")
        mine = terms.filter((pl.col("key") == r["key"]) & (pl.col("week") == r["week"]))
        for term, pts in mine.select("term", "points").iter_rows():
            print(f"      {term:>18} {pts:+.2f}")
    for r in out.filter(pl.col("status") == "unresolved").iter_rows(named=True):
        print(f"  UNRESOLVED wk{r['week']} Sleeper id {r['platform_id']}: {r['awarded']} pts")
    return 0 if rep.mismatched == 0 else 1


def _evaluate(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    import polars as pl

    from ffa import evaluate
    from ffa.model.inputs import Models
    from ffa.model.matchup import MatchupModel
    from ffa.model.player import PlayerModel
    from ffa.model.sim_params import SimParams
    from ffa.model.team import TeamModel
    from ffa.scoring import from_sleeper
    from ffa.warehouse import duck

    m = Path(settings.models_dir)
    models = Models(
        TeamModel.load(m / "team_volume.json"),
        PlayerModel.load(m / "player.json"),
        MatchupModel.load(m / "matchup.json"),
        SimParams.load(m / "sim_params.json"),
    )
    rules = from_sleeper(json.loads(Path(args.scoring).read_text()))
    lo, hi = (int(x) for x in args.weeks.split("-"))
    con = duck.connect(settings.warehouse_dir)
    h = evaluate.run(con, args.season, list(range(lo, hi + 1)), models, rules, n_sims=args.sims)
    pl.Config.set_tbl_cols(14)
    pl.Config.set_tbl_width_chars(200)
    print(f"Season {args.season}, weeks {lo}-{hi}: {h.rows.height:,} player-weeks")
    print("\nAccuracy (MAE on games played; edge = baseline MAE - model MAE)")
    print(evaluate.accuracy(h).with_columns(pl.col(pl.Float64).round(3)))
    print("\nShare of outcomes inside the 10th-90th range (projection >= 5)")
    print(evaluate.coverage(h, 5).sort("position").with_columns(pl.col(pl.Float64).round(3)))
    acc, calib = evaluate.start_sit(h)
    print("\nStart/sit: picks the higher scorer")
    print(acc.with_columns(pl.col("accuracy").round(3)))
    print("\nP(A outscores B): said vs happened")
    print(calib.with_columns(pl.col("said", "happened").round(3)))
    return 0


def _rebuild_projections(args: argparse.Namespace) -> int:
    """Regenerate stored weeks whose projection files are in an older format. Rewrites
    projection_sims / _inputs / _teams for those weeks; projection_history (what was
    projected at the time) is left untouched."""
    from pathlib import Path

    import polars as pl

    from ffa.service import projections as proj
    from ffa.warehouse import duck

    root = Path(settings.warehouse_dir) / "projection_inputs" / f"season={args.season}"
    stored = sorted(int(d.name.split("=")[1]) for d in root.glob("week=*") if d.is_dir())
    stale = []
    for w in stored:
        files = sorted((root / f"week={w:02d}").glob("*.parquet"))
        try:
            proj.check_inputs(pl.read_parquet(files[-1], n_rows=0))
        except proj.SchemaMismatch as e:
            stale.append((w, str(e)))
    weeks = stored if args.all else [w for w, _ in stale]
    print(f"{args.season}: stored weeks {stored}; old format: {[w for w, _ in stale] or 'none'}")
    for w, why in stale:
        print(f"  week {w}: {why}")
    if args.dry_run or not weeks:
        return 0
    models = proj.load_models(settings.models_dir)
    for w in weeks:
        con = duck.connect(settings.warehouse_dir)
        r = proj.run_week(con, settings.warehouse_dir, models, args.season, w,
                          record_history=False)  # fmt: skip
        print(f"  rebuilt week {w}: {r.players} players")
    return 0


def _report(args: argparse.Namespace) -> int:
    """Print the accuracy report for a week from stored eval_results."""
    from ffa.tracking import report as rep
    from ffa.warehouse import duck

    con = duck.connect(settings.warehouse_dir, datasets=[rep.RESULTS])
    try:
        res = con.sql(f"select * from {rep.RESULTS} where season = {args.season}").pl()
    except Exception:
        print("No scored weeks yet (eval_results runs Tuesday and Wednesday mornings).")
        return 1
    week = args.week or int(res["week"].max())
    print(rep.markdown(res.filter(res["week"] <= week), args.season, week))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ffa")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("leagues", help="list a user's leagues")
    p.add_argument("platform", choices=["sleeper"])
    p.add_argument("user")
    p.add_argument("--season", type=int, default=settings.current_season)
    p.set_defaults(func=_leagues)

    p = sub.add_parser("sync", help="fetch and store a league")
    p.add_argument("platform", choices=["sleeper"])
    p.add_argument("league_id")
    p.add_argument("--me", help="your username, to mark your team")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_sync)

    p = sub.add_parser("check-scoring", help="compare our points with what the platform awarded")
    p.add_argument("platform", choices=["sleeper"])
    p.add_argument("league_id")
    p.add_argument("--weeks", type=int, nargs="+", help="default: every completed week")
    p.set_defaults(func=_check_scoring)

    p = sub.add_parser("evaluate", help="out-of-sample projection evaluation vs baselines")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--weeks", default="1-18", help="e.g. 1-18")
    p.add_argument("--scoring", required=True, help="league scoring settings JSON (Sleeper)")
    p.add_argument("--sims", type=int, default=1000)
    p.set_defaults(func=_evaluate)

    p = sub.add_parser(
        "rebuild-projections",
        help="regenerate stored weeks whose projection files are in an old format",
    )
    p.add_argument("--season", type=int, default=settings.current_season)
    p.add_argument("--all", action="store_true", help="rebuild every stored week")
    p.add_argument("--dry-run", action="store_true", help="only list what would be rebuilt")
    p.set_defaults(func=_rebuild_projections)

    p = sub.add_parser("report", help="projection accuracy report for a scored week")
    p.add_argument("--season", type=int, default=settings.current_season)
    p.add_argument("--week", type=int, help="default: latest scored week")
    p.set_defaults(func=_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
