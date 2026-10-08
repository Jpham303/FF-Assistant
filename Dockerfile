FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --no-dev --no-install-project

COPY src ./src
COPY dagster_defs ./dagster_defs
COPY dagster.yaml workspace.yaml alembic.ini ./
COPY migrations ./migrations
RUN uv sync --no-dev

ENV PATH="/app/.venv/bin:$PATH" DAGSTER_HOME=/app/.dagster_home
RUN mkdir -p $DAGSTER_HOME && cp dagster.yaml $DAGSTER_HOME/
