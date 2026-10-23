# FF Assistant

A fantasy football assistant built on a backtested statistical engine, not just a chatbot.

- **Projections:** probabilistic weekly projections of raw stats (targets, carries, yards, TDs), converted to points under any league's scoring rules
- **Matchups:** opponent-adjusted run and pass defense, game script from Vegas lines, pace
- **Start/sit:** P(player A outscores player B) with the drivers behind it
- **Trades:** roster-aware rest-of-season value and trade target search
- **Interfaces:** FastAPI backend, Streamlit app, and an MCP server so AI agents can query the same API

Leagues: Sleeper and Yahoo, with custom scoring read from each platform.

## Architecture

| Layer | Tech |
| --- | --- |
| Pipeline | Dagster |
| Analytics warehouse | Parquet + DuckDB |
| App database | Postgres |
| API | FastAPI |
| Frontend | Streamlit |
| AI interface | MCP server over the API |

Data comes from [nflverse](https://github.com/nflverse) and the official Sleeper and Yahoo APIs. No scraping.

## Run locally

```bash
cp .env.example .env        # fill in values
docker compose up --build
```

- API: http://localhost:8000/health
- Dagster: http://localhost:3300
- Postgres (for a SQL client): localhost:5433

Ports are bound to this computer only, not your network. If one is already in use, change
`API_PORT`, `DAGSTER_PORT` or `POSTGRES_HOST_PORT` in `.env`.

## Load data

The first time, backfill every season (about 30 seconds, ~100 MB):

- Dagster UI → **Assets** → group `nflverse` → **Materialize all** → select all partitions

After that, two schedules keep the current season fresh (turn them on in the UI):

| Schedule | When | What |
| --- | --- | --- |
| `nightly_current_season` | 2:15am Central | all datasets for the current season |
| `hourly_injuries` | :05 every hour | injury reports and rosters, kept as timestamped snapshots |
| `nightly_dims` | 2:45am Central | player ID crosswalk (nflverse ↔ Sleeper ↔ Yahoo) |

### Player IDs

Every player is keyed by nflverse's `gsis_id`; team defenses use `DST_<TEAM>`. The
`player_ids` table maps Sleeper and Yahoo IDs onto it from three sources (Sleeper's player
DB, nflverse rosters, DynastyProcess), recording which source supplied each ID and flagging
disagreements. Players with no platform ID — usually recent rookies on Yahoo — are matched
at import by normalized name and position, then team. Ambiguous names are never guessed.

Data lands in `data/warehouse/{dataset}/season=YYYY/week=WW/`. Query it with
`ffa.warehouse.duck.connect()`; use `duck.as_of()` to see data as it stood at a past moment.

## Connect your league

```bash
uv run ffa leagues sleeper <your_username>          # find your league ID
uv run ffa sync sleeper <league_id> --dry-run       # preview the import and match report
```

Then set `SLEEPER_USERNAME` and `SLEEPER_LEAGUE_ID` in `.env`. The `sleeper_league` asset
syncs settings, teams and rosters into Postgres hourly (`hourly_leagues` schedule), and
logs any player it could not match.

| Table | Contents |
| --- | --- |
| `leagues` | lineup slots, bench/IR/taxi sizes, raw scoring settings |
| `teams` | owner, record, points for, which team is yours |
| `roster_entries` | every rostered player: canonical key, how it was matched, starter/bench/IR slot |

Database schema changes go through Alembic (`migrations/`); the `migrate` service applies
them on `docker compose up`.

## Scoring

The engine projects and scores **raw stats**, then applies each league's rules, so any
scoring system works without retraining anything.

```python
from ffa.scoring import stat_lines, score, breakdown, from_sleeper

rules = from_sleeper(league.scoring)  # platform settings -> canonical rules
pts = score(stat_lines(player_stats, pbp, schedules), rules)
why = breakdown(...)  # per-term points, for explanations
```

- **Canonical stats** follow Sleeper's vocabulary (`rec`, `pass_yd`, `fgm_yds`, `pts_allow`…).
  Players come from nflverse weekly stats; team defenses are built play by play, because
  the platforms score things team totals blur together: special-teams vs defensive
  fumbles, uncredited strip-sacks, blocked extra points, and points allowed that excludes
  return touchdowns and safeties the defense was not on the field for.
- **Rules** are linear terms, bands (points-allowed tiers, yardage bonuses) and
  position-specific terms (TE premium). Platform settings we can't compute are listed in
  `rules.unsupported` or `rules.approximate`, never silently dropped.

### Verification

| Check | Result |
| --- | --- |
| Our points vs **Sleeper's awarded points**, a real 12-team league, weeks 1–4 of 2026 | **683 / 683 player-weeks exact** |
| Offense vs nflverse's own fantasy points, every player-week since 2015 | 204,077 / 204,077 (return fumbles aside, which nflverse omits) |
| Play-by-play scoring events vs final scores, every game since 2015 | 6,184 / 6,184 team-games exact |

Re-run the Sleeper comparison any week; it exits non-zero on any mismatch and prints our
per-term breakdown for it:

```bash
uv run ffa check-scoring sleeper <league_id>
```

## Projections

A simulation model, not a single-number guess. Each week, every game is simulated 2,000
times from data available before kickoff:

1. **Team volume** from the betting lines plus team tendencies: plays, pass rate, TDs, FGs
2. **Player shares** (targets, carries, dropbacks, red-zone looks) and **efficiency**,
   recency-weighted and shrunk toward priors fitted from data
3. **Matchups**: where each defense gives up production relative to expectation
4. **Availability** from injury status, final practice and role
5. Players' stat lines from their team's simulated game (the QB's passing is the sum of
   his receivers'; a defense's points allowed is the opponent's simulated score), scored
   with the league's rules

**2025 holdout** (model frozen before it was run; details in
[docs/evaluation_2025.md](docs/evaluation_2025.md)):

| | Model | Season average | Opportunity average |
| --- | --- | --- | --- |
| Start/sit, all startable pairs | **62.3%** | 58.9% | 61.1% |
| Start/sit, close calls | **58.9%** | 53.6% | 56.0% |
| Projection error | lowest at every position | | |

## Injuries and game status

Availability is the input that matters most on a given Sunday, and the slowest to arrive:
teams give game designations (Out, Doubtful, Questionable) only with the final injury
report, usually Friday.

- **Two sources.** nflverse's injury file (hourly) and Sleeper's player DB (once a day, as
  Sleeper requests: `daily_status` at 5:05 PM CT, `sunday_status` at 10:45 AM CT after
  inactives for the noon games; each rebuilds projections right after). Schedules are on
  by default, and the projection run fetches Sleeper itself if its data is missing or
  over a day old. The more severe
  designation wins: a missed ruling costs far more than a cautious one. The practice note
  comes from the official report when it has one, even when Sleeper supplies the
  designation.
- **Before the team's first report of the week** (usually until Wednesday), each player's
  designation from the team's previous report is carried forward, at the rate players
  in that spot historically played the next game (Out ~30%, Doubtful ~45%, Questionable
  ~70%, by role; fitted 2016-2023, matching 2024-2025). Absence from a report counts as
  healthy only once the team's new report exists.
- **Before the final report**, a player with no designation who missed or was limited in
  practice is not assumed healthy. His chance to play comes from history: every report row
  with that practice level, split by role and whether he was Out the week before (a starter
  who was Out and hasn't practiced played about 1% of the time). Rest days don't count.
  Whether a team's final report is in is judged from the data (final reports designate
  someone; 99.7% of team-weeks since 2015), not the clock, because feeds lag.
- **Every injured player carries his source and time**, e.g. "Injury: OUT (hamstring), did
  not practice [Sleeper, updated Fri Oct 16 4:12 PM CDT; fetched 5:05 PM CDT]". Ruled-out
  players are never recommended and never put in a lineup.
- **Limit:** inactives announced after the last fetch (late games, Sunday-morning
  scratches) are not reflected. `/week` says when each source was last fetched, and every
  player who may not play carries `game_day_check`: when inactives are announced for his
  game (about 90 minutes before kickoff), so you know when to confirm before lineups lock.
  Sleeper's once-a-day limit rules out refreshing before each kickoff window.

## API

The weekly projections job (`weekly_projections`, hourly at :25 after the injury snapshot)
simulates the upcoming week and stores league-neutral stat lines, so any league's scoring
is applied at request time. Interactive docs at http://localhost:8000/docs.

| Endpoint | What it returns |
| --- | --- |
| `GET /week` | current season, upcoming week, when projections ran, when each injury source was last fetched, data sources |
| `GET /players/search?q=` | player keys by name, typos allowed; `match` (1.0 = exact) and `projected` (false on a bye). A bye-week player is reported as not playing, never swapped for a similar name |
| `GET /projections?keys=` | **`proj_if_active`** (points if he plays) with its range, `play_probability` and injury status beside it, `expected_points` (= play probability × if-active; what lineups rank by), expected stats if active, points per scoring rule, and plain-language drivers each carrying its number. `mean`/`median`/`p10`/`p90`/`expected_stats` are deprecated copies for one release |
| `POST /start-sit` | 2–6 players: recommendation (key and name, by expected points), `p_a_beats_b` (chance of sitting counted), `p_a_beats_b_if_both_active` (talent alone) and `median_gap` from simulations that share each game; `decision_drivers`: the three biggest point differences between the pick and the runner-up by stat group (receptions, rushing yards, TDs...) plus availability, each with both players' expected stats. With a `league` where your team is known, `matchup`: your opponent this week (from Sleeper, cached a minute), his lineup (as set, else his best), and your chance of winning with each candidate starting (your best lineup around him; finished games use Sleeper's awarded points). On a close call (below 55%) the candidate with the higher chance of winning, by at least 1 point, becomes the pick, with a note saying why (floor when you're favored, ceiling when you're the underdog, or game correlation). Close-call and injury notes |
| `GET /leagues/{id}/my-team` | best lineup for your roster under the league's slots and scoring; bench; who isn't playing |

A player with a low chance to play (fewer than 500 of 2,000 simulations active) gets his
if-active numbers from an extra batch with him forced active, teammates as usual
(`forced_active_batch`). Players not playing have no if-active projection and 0 expected
points.

Pass `league` (e.g. `sleeper:<league_id>`) to score with that league's settings; without
it, `SLEEPER_LEAGUE_ID` or standard half-PPR is used.

The API reads only the requested week's projection files (`duck.week`), so weeks stored in
an older format can't affect it. Each projection run checks its inputs against the format
the API expects (`projections.INPUT_SCHEMA`) and fails the run on a mismatch. After a
format change, regenerate stored weeks: `uv run ffa rebuild-projections --dry-run` lists
them, without `--dry-run` rebuilds them (projection_history, the record of what was
projected at the time, is left as it was).

## Live accuracy tracking

Every week is scored against projections frozen before kickoff (spec: *FF Assistant —
Projection Evaluation Spec*, Phases 1 and 2).

| Piece | What it does |
| --- | --- |
| `projection_snapshots` | Written by every projection run, never overwritten (a new snapshot only when the forecast changed). Per player and league scoring: `expected_points`, `proj_if_active`, p10/p25/p50/p75/p90 and 21 quantiles, play probability, injury status with source and time, kickoff, model version (hash of the fitted files), git commit, and the scoring rules used |
| `pre_kickoff_snapshot` sensor | Fresh projections, a snapshot and Sleeper's benchmark ~45 minutes before each kickoff window (Thursday, London, Sunday, Monday) |
| `sleeper_benchmark` | Sleeper's own projections (unofficial endpoint, one request per window), raw response kept in `benchmark_raw/`, scored in each league's scoring (Sleeper stat keys × league settings, as Sleeper does) into `benchmark_snapshots`; write-once, skipped when unchanged. A failure is logged and never blocks projections |
| `actuals` | Stat lines, snaps and an active flag per player-week; Tuesday and Wednesday 5:30 AM CT (after stat corrections) |
| `eval_results` | Each player scored against the last snapshot taken before his kickoff: MAE if active, MAE on expected points, bias, p10–p90 and p25–p75 coverage, by position, against last-4 and season-average baselines on the same players. Against Sleeper (last capture before kickoff, same players): MAE, rank correlation, start/sit accuracy on close pairs (season averages within 3 points). Misses split into availability misses (50%+ to play and projected 3+ if active, no snap), early exits (under 25% of usual snaps) and model misses; only model misses feed MAE if active, bias and coverage. Chance to play said vs happened by injury status. Writes `data/reports/accuracy_<season>_week<NN>.md` |

`uv run ffa report [--week N]` prints the report. To record the git commit in snapshots,
build with `FFA_GIT_SHA` set (PowerShell: `$env:FFA_GIT_SHA = git rev-parse --short HEAD`).

## Use it from Claude (MCP)

`ffa-mcp` is an MCP server that gives AI assistants six read-only tools over the API:
`current_week`, `search_players`, `start_sit`, `player_projections`, `list_leagues`,
`my_team`. Tools accept player names; a name is used only on an exact match for a player
who is playing this week, otherwise the tool reports the problem instead of guessing.
The server's instructions tell the assistant to cite only numbers the tools return, to
treat matchups as a small factor, and to name the data sources and timestamps.

1. Start the stack: `docker compose up` (the API must be reachable at `FFA_API_URL`,
   default `http://localhost:8000`).
2. Add to Claude Desktop's config (`~/Library/Application Support/Claude/claude_desktop_config.json`
   on macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows), then restart it:

```json
{
  "mcpServers": {
    "ff-assistant": {
      "command": "uv",
      "args": ["--directory", "/path/to/FF-Assistant", "run", "ffa-mcp"],
      "env": { "FFA_API_URL": "http://localhost:8000" }
    }
  }
}
```

If Claude Desktop can't find `uv`, use its full path (`which uv`).

## Develop

```bash
uv sync --all-groups
uv run pre-commit install
uv run pytest
```

## Status

Phase 0 (data, leagues, scoring) done. Phase 1 (projections, start/sit API, MCP server) in progress.
