# ENLIGHT

## European Network for Long-term Insights on Grid prices, Hedging & Trends

ENLIGHT is a multi-year electricity market forecast model that simulates market clearing across European bidding zones using solver-based optimization.

## Key Features

- Models all European bidding zones with multi-year foresight
- Reproduces market clearing via optimization (Gurobi or HiGHS)
- Scenario configuration via YAML files
- Rolling-horizon or full-year simulation modes
- Outputs: prices, dispatch, line flows, curtailment

## How It Works

Each simulation run follows a four-step pipeline:

1. **Data Preprocessing** — transforms raw input data into model-ready formats and saves to `simulations/<name>/data/`
2. **Data Loading** — loads preprocessed data into structured objects (DataFrames, arrays)
3. **Model Execution** — solves the market clearing optimization (maximise social welfare subject to power balance, generation limits, transmission capacity, storage dynamics)
4. **Results Export** — extracts solutions, calculates metrics, saves to `simulations/<name>/results/`

## Output Structure

```text
simulations/<name>/
├── data/        # preprocessed inputs
└── results/
    ├── electricity_prices.csv
    ├── generation_schedules.csv
    ├── lineflows.csv
    ├── demand_served.csv
    └── curtailment.csv
```

## License

GPL-3.0. See [LICENSE](https://github.com/marco-saretta-DTU/enlight/blob/main/LICENSE).
