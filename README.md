# ENLIGHT

European Network for Long-term Insights on Grid prices, Hedging & Trends

## Overview
Multi-year electricity market forecast model simulating market clearing across all European bidding zones using solver-based optimization.

- Models all European bidding zones with multi-year foresight  
- Reproduces market clearing using optimization (Gurobi/HiGHS)  
- Easy scenario configuration via Excel and YAML
- Flexible yearly or weekly simulation modes
- Comprehensive output: prices, dispatch, flows, curtailment

## Installation

Clone the repository via:
```bash
git clone https://github.com/marco-saretta/enlight.git
```

Then proceed to create the virtual environment ro run enlight.
* **Installing with conda**: 
```bash
cd <enlight_directory>/enlight        # Set the enlight directory
conda env create -f environment.yaml  # Create the environment via .yaml file
conda activate enlight-env            # Activate the environment
```

* **Installing with uv**:
```bash
cd <enlight_directory>/enlight    # Set the enlight directory
uv sync                           # Create the virtual environment with uv from .toml file
uv run main.py                    # Run main.py script
```
*Note: id uv is not installed, please run the following in powershell. Please refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).*
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"    # Optional: install uv
```

## Quick Start

**1. Configure scenarios** in `config/scenarios_config.xlsx`  
Define technology capacities, costs, and constraints for each scenario.

**2. Set simulation parameters** in `config/config.yaml`

```yaml
scenario_list:
  - scenario_1:
      run_mode: "yearly"      # Full year optimization
      year: 2030
      plant_aggregation: false

  - scenario_2:
      run_mode: "weekly"      # Week-by-week optimization
      year: 2040
      plant_aggregation: true
      week_range:
        start_week: 10
        end_week: 20

solver_name: "gurobi"         # Options: "gurobi", "highs"
bidding_zones:                # Select zones to model
  - AT
  - DELU
  - FR
  # ...
```

**3. Run simulations**

```bash
python main.py
```

```python
# main.py
from enlight.runner import EnlightRunner
from pathlib import Path

if __name__ == "__main__":
    root_path = Path(__file__).parent.resolve()
    r = EnlightRunner(root_path=root_path)
    
    r.run_scenario('scenario_1')
    r.run_scenario('scenario_2')
```

## How It Works: The Four-Step Pipeline

Each scenario execution follows an automated pipeline:

### 1. Data Preprocessing (`DataProcessor`)
Transforms raw data into model-ready formats. Processes generation profiles, demand, network topology, and validates consistency. Saves to `simulations/<scenario_name>/data/`.

### 2. Data Loading (`DataLoader`)
Loads preprocessed data into memory as structured objects (DataFrames, arrays). Organizes by time period (yearly: 8760h, weekly: 168h). Accessible for debugging.

### 3. Model Execution (`EnlightModel`)
Solves market clearing optimization (maximize social welfare subject to power balance, generation limits, transmission capacity, storage dynamics). Outputs dispatch schedules, prices, and flows.

### 4. Results Export (`DataExporter`)
Extracts solutions, calculates metrics, saves results to `simulations/<scenario_name>/results/`. For weekly mode, combines individual weeks into annual summaries.

## Yearly vs Weekly Mode

**Yearly:** Single optimization over 8760 hours. Captures seasonal patterns and long-term storage. Higher memory requirement.

**Weekly:** Sequential optimization of individual weeks. Lower memory footprint, suitable for large systems. May not capture multi-week storage strategies.

## Output Structure

```
simulations/<scenario_name>/
├── data/                          # Preprocessed inputs
└── results/                       # Model outputs
    ├── electricity_prices.csv
    ├── generation_schedules.csv
    ├── lineflows.csv
    ├── demand_served.csv
    ├── curtailment.csv
    └── marginal_generator.csv
```

## Logging

Execution logs with timing saved to `logs/enlight.log`:

```
2024-11-24 10:30:00 - enlight - INFO - Starting: Data preprocessing
2024-11-24 10:30:05 - enlight - INFO - Completed: Data preprocessing in 5.23 seconds
2024-11-24 10:30:05 - enlight - INFO - Starting: Model execution
2024-11-24 10:45:12 - enlight - INFO - Completed: Model execution in 904.56 seconds
```

## Code Structure

```
.
├── main.py                   # Entry point
├── config/
│   ├── config.yaml           # Simulation settings
│   └── scenarios_config.xlsx # Scenario parameters
├── data/                     # Base input datasets
├── enlight/
│   ├── data_ops/             # Preprocessing, loading, export
│   ├── model/                # Optimization model
│   ├── runner/               # Pipeline orchestration
│   └── utils/                # Logging, timing, helpers
├── simulations/              # Scenario outputs
└── logs/                     # Execution logs
```

## Documentation

See `docs/` for detailed guides on configuration, architecture, API reference, and troubleshooting.

If your Gurobi license restricts the number of CPU cores and you need to run inside Docker, see the [Docker guide](docs/source/docker.md).

## Testing

```bash
pytest tests/
```

## License

GPL-3.0 license.

## Authors

**Marco Saretta** | Collaborator: Viktor Johnsen

## Citation

```bibtex
@software{enlight2024,
  author = {Saretta, Marco and Johnsen, Viktor},
  title = {ENLIGHT: European Network for Long-term Insights on Grid prices, Hedging \& Trends},
  year = {2024},
  url = {https://github.com/marco-saretta/enlight}
}
```