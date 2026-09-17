# Running simulations

## From the command line

```bash
uv run main.py                                        # the default scenario
uv run main.py simulations=demo1                      # another scenario file
uv run main.py 'steps=[preprocess]'                   # only part of the pipeline
uv run main.py simulations.run.mode=rolling_horizon   # override any config value
```

Keep the quotes around `[...]` and `glob(...)`: otherwise shells such as zsh expand them before Hydra sees them.

## Several scenarios

Hydra's `--multirun` runs one job per value, one after another:

```bash
uv run main.py --multirun simulations=demo1,demo2       # two scenarios
uv run main.py --multirun 'simulations=glob(*)'         # every file in config/simulations/
uv run main.py --multirun simulations=demo1,demo2 simulations.run.solver=highs,gurobi   # all 4 combinations
```

## From Python or a notebook

`load_config` composes the same configuration without `main.py`; the steps can then be called one at a time and their outputs inspected.

```python
from enlight.runner import EnlightRunner, load_config

cfg = load_config("demo1", overrides=["simulations.run.mode=yearly"])
runner = EnlightRunner(cfg)

runner.preprocess()
runner.load()
runner.build()
runner.solve()
runner.export()

runner.data.thermal.marginal_cost   # loaded inputs
runner.model.model                  # the linopy model
```

In rolling horizon mode, pass the week: `runner.load(week=1)` … `runner.export(week=1)`.

## Output

```text
simulations/<label>/
├── data/                        # preprocessed inputs (inspect before solving)
└── results/
    ├── electricity_prices.csv   # EUR/MWh, hour x zone
    ├── dispatch.csv             # MW, hour x (technology, zone); + injection, - withdrawal
    ├── energy_balance.csv       # TWh, technology x zone
    ├── capture_prices.csv       # EUR/MWh, technology x zone, plus baseload
    └── solver.log
```

The console shows one line per event, tagged with its step (`[preprocess]`, `[solve]`, …); `logs/enlight.log` keeps every line, including each week's details in rolling horizon mode.
