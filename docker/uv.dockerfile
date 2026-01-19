# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY enlight/ enlight/
COPY data/ data/
COPY config/ config/
COPY main.py main.py
    
WORKDIR /
RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT ["uv", "run", "./main.py"]

# docker build -f ./Docker/uv.dockerfile . -t enlight_test:latest