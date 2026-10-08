"""Command line: `uv run ffa --help`.

ffa leagues sleeper jpham03            list your Sleeper leagues and their IDs
ffa sync sleeper <league_id>           fetch, resolve and store a league; print the report
ffa sync sleeper <league_id> --dry-run same, without writing to the database
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
