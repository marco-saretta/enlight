# Files

```text
enlight/
├── main.py                        # entry point: python main.py
├── config/                        # Hydra configuration (see Configuration)
├── data/                          # raw input data, never modified by a run
├── simulations/<label>/           # created by a run
│   ├── data/                      # preprocessed inputs
│   └── results/                   # outputs
├── logs/enlight.log               # full log of every run
├── docs/                          # this documentation
└── src/enlight/
    ├── runner/runner.py           # EnlightRunner: the five pipeline steps, load_config
    ├── data_ops/
    │   ├── data_preprocessor.py   # step 1: data/ -> simulations/<label>/data/
    │   ├── data_loader.py         # step 2: CSVs -> xarray, one object per technology
    │   └── data_exporter.py       # step 5: results CSVs
    ├── model/
    │   ├── energy_model.py        # step 3-4: assembles and solves the model
    │   └── build_<tech>.py        # one script per technology
    └── utils/                     # logging, validation, helpers
```

## Raw data (`data/`)

| Folder | Content |
|---|---|
| `wind_onshore/`, `wind_offshore/`, `solar_pv/`, `hydro_ror/` | `capacity_projections/<file>.csv` (MW per year and zone) and `weather_data/<source>/<tech>_<source>_wy_<year>.csv` (hourly per-unit profiles) |
| `demand_inflexible_classic/`, `demand_inflexible_ev/` | `demand_projection/` (annual energy) and `profile_years/` (hourly profiles) |
| `thermal_plants/units/` | one row per unit: zone, technology, fuel, fuel type, capacity |
| `technology_data/`, `emissions/`, `fuel_price_projections/<dataset>/` | efficiencies and O&M, CO2 intensities, fuel and CO2 prices per year |
| `lines/<dataset>/` | hourly transmission capacity in both directions |

Most files have one column per bidding zone; only the zones listed in the scenario are used.
