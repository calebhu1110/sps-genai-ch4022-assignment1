FROM python:3.13-slim-bookworm

# Use the image's Python 3.13; never let uv download a newer Python
# (packages like httptools have no prebuilt wheels for it yet).
ENV UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY app ./app

RUN uv run python -m spacy download en_core_web_lg

CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]
