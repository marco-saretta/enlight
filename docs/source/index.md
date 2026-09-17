# ENLIGHT

**European Network for Long-term Insights on Grid prices, Hedging & Trends**

ENLIGHT forecasts hourly electricity prices across European bidding zones by clearing the market as a linear program: supply offers and demand bids meet in every zone and hour, linked by transmission lines, and the zonal prices are the duals of the power balance.

!!! note "Work in progress"
    These pages are a placeholder while the model is being rebuilt. They describe the current state of the code, not a finished product.

## Pipeline

A run goes through five steps, each a method of `EnlightRunner`:

| Step | Reads | Writes |
|---|---|---|
| 1. `preprocess` | raw data in `data/` | model inputs in `simulations/<label>/data/` |
| 2. `load` | `simulations/<label>/data/` | xarray objects (`runner.data`) |
| 3. `build` | loaded data | linopy model (`runner.model`) |
| 4. `solve` | model | solution and prices |
| 5. `export` | solved model | results in `simulations/<label>/results/` |

The run is either **yearly** (one model for 8760 h) or **rolling horizon** (one model per week, results joined afterwards).

## Pages

- [Installation](installation.md)
- [Running simulations](running.md)
- [Configuration](configuration.md)
- [Model](model.md)
- [Files](files.md)
- [Docker (Gurobi)](docker.md)

## License

GPL-3.0. See [LICENSE](https://github.com/marco-saretta-DTU/enlight/blob/main/LICENSE).
