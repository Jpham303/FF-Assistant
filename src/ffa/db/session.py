from __future__ import annotations

from functools import cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ffa.config import settings


@cache
def engine(url: str | None = None) -> Engine:
    return create_engine(url or settings.database_url, pool_pre_ping=True)


def session(url: str | None = None) -> Session:
    return sessionmaker(engine(url), expire_on_commit=False)()
