# Installation

## Requirements

- Python >= 3.12
- A valid Gurobi license (optional — HiGHS is the open-source alternative)
- An ENTSO-E API key (optional — required for live data fetching)

## Clone the Repository

```bash
git clone https://github.com/marco-saretta-DTU/enlight.git
cd enlight
```

## Install with uv (recommended)

```bash
uv sync          # creates .venv and installs all dependencies
uv run main.py   # run the model
```

If `uv` is not installed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Install with conda

```bash
conda env create -f envs/environment.yaml
conda activate enlight-env
python main.py
```

## ENTSO-E API Key

Place your key in `configs/entsoe_api_key/placeholder.yaml`:

```yaml
entsoe_api_key: "YOUR-KEY-GOES-HERE"
```