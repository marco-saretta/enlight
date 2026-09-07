ARG GUROBI_VERSION=13.0.1
ARG PYTHON_VERSION=3.12
FROM gurobi/python:${GUROBI_VERSION}_${PYTHON_VERSION}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first (layer cache: only invalidated when lock file changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Install the project itself
COPY src/ src/
COPY main.py ./
RUN uv sync --locked --no-dev

# License is mounted at runtime — never bake it into the image
ENV GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic

ENTRYPOINT ["uv", "run", "python", "main.py"]
