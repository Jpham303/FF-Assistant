"""Game logs the projection model is built from.

`team_games` and `player_games` hold one row per team-game and per player-game: the betting
lines, the opportunities, and the outcomes. Rows describe what happened in that game; the
model reads only rows from earlier games when it predicts a later one (see ffa.model).
"""

from ffa.features.games import player_game_log, team_game_log

__all__ = ["player_game_log", "team_game_log"]
