# Installation

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- A Gurobi license (optional - HiGHS is the open-source default)

## Install

```bash
git clone https://github.com/marco-saretta-DTU/enlight.git
cd enlight
uv sync            # creates .venv and installs all dependencies
uv run main.py     # runs the default scenario
```

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Build these docs locally

```bash
uv run --with "mkdocs<2" --with mkdocs-material --with mkdocstrings-python mkdocs serve -f docs/mkdocs.yaml
```
