"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_sleeper_matchups(monkeypatch):
    """Start/sit fetches the week's fantasy matchups from Sleeper; tests never hit the
    network (a test that needs matchups patches in its own)."""
    from ffa.service import fantasy_matchup

    def offline(*_a, **_k):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(fantasy_matchup, "fetch_matchups", offline)
