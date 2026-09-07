# Running with Docker (Gurobi)

Some Gurobi licenses restrict usage to machines with 8 or fewer CPU cores. If your machine has more, the solver will refuse to run. The Docker setup in `dockerfiles/` works around this by running the solver inside a container capped at 7 CPUs, which satisfies the license check regardless of the host hardware.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A valid `gurobi.lic` license file

## One-time setup

**1. Place your license file** somewhere stable on your machine, for example:

```
~/.gurobi/gurobi.lic
```

**2. Set the environment variable** so Docker knows where to find it:

```bash
# Linux / macOS
export GRB_LICENSE_FILE=~/.gurobi/gurobi.lic

# Windows (PowerShell)
$env:GRB_LICENSE_FILE = "$HOME\.gurobi\gurobi.lic"
```

You can add this line to your shell profile (`~/.bashrc`, `~/.zshrc`) to avoid repeating it.

**3. Build the image** from the repo root:

```bash
docker compose -f dockerfiles/docker-compose.gurobi.yml build
```

This installs all Python dependencies and the Gurobi solver inside the image. The license file is never copied into the image.

## Running a simulation

```bash
docker compose -f dockerfiles/docker-compose.gurobi.yml run gurobi
```

Pass Hydra overrides as you normally would:

```bash
docker compose -f dockerfiles/docker-compose.gurobi.yml run gurobi simulations=sim_1
docker compose -f dockerfiles/docker-compose.gurobi.yml run gurobi --multirun simulations=sim_1,sim_2
```

Results are written to `outputs/` in your local repo, exactly as when running without Docker.

## Workflow

You edit code and configs in your local repo as usual. The container is only invoked for the solve step. Git works normally throughout.

```
edit code / configs locally
        |
        v
docker compose ... run gurobi simulations=sim_1
        |
        v
outputs/ appears in your local repo
        |
        v
git add / commit / push as usual
```

## Rebuilding the image

Rebuild only when `pyproject.toml` or `uv.lock` changes (i.e. when dependencies change). Config and data changes do not require a rebuild because `configs/` and `data/` are mounted as volumes.

```bash
docker compose -f dockerfiles/docker-compose.gurobi.yml build --no-cache
```

## Gurobi version

The image uses `gurobi/python:13.0.1_3.12` by default. To use a different version, edit the `GUROBI_VERSION` build arg in [dockerfiles/docker-compose.gurobi.yml](../../dockerfiles/docker-compose.gurobi.yml) or pass it at build time:

```bash
docker compose -f dockerfiles/docker-compose.gurobi.yml build --build-arg GUROBI_VERSION=12.0.2
```

Available tags are listed on the [Gurobi Docker Hub page](https://hub.docker.com/r/gurobi/python/tags).
