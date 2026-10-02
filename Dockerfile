FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY app ./app

RUN uv run python -m spacy download en_core_web_lg

CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]
