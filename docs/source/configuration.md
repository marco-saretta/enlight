# Configuration

ENLIGHT uses [Hydra](https://hydra.cc/) for configuration. Settings are split into two layers:

| File | Purpose |
|---|---|
| `configs/default_config.yaml` | Global settings: solver, bidding zones, paths |
| `configs/simulations/<name>.yaml` | Per-simulation settings: run mode, years, technology parameters |

---

## Global Config (`default_config.yaml`)

### Solver

```yaml
solver_name: highs   # highs (open-source) | gurobi (requires license)
```

### Bidding Zones

Uncomment zones to include them. Each active zone must have corresponding input data.

```yaml
bidding_zones:
  - AT   # Austria
  - FR   # France
  # - DE  # (commented out = excluded)
```

### Multi-run Mode

Uncomment the `hydra` block to enable sweeps across multiple simulation configs:

```yaml
hydra:
  mode: MULTIRUN
  sweeper:
    params:
      simulations: glob(*)
```

---

## Simulation Config (`configs/simulations/<name>.yaml`)

### Run Control

```yaml
run:
  mode:              rolling_horizon   # rolling_horizon | yearly
  prediction_year:   2040              # capacity/demand projection year
  plant_aggregation: true              # aggregate units by zone+fuel

rolling_horizon:
  start_week: 1    # [1-52]
  end_week:   52   # [1-52]
```

### Production Technologies

Each technology block follows the same pattern:

```yaml
wind_onshore:
  weather_year:  2020                      # historical profile year
  capacity_file: TYNDP_2024_National_Trends
  bid_price:     0.01                      # EUR/MWh
```

Technologies: `wind_onshore`, `wind_offshore`, `solar_pv`, `hydro_ror`, `hydro_res`, `hydro_ps`, `thermal`.

### Demand

Demand is split into **inflexible** (price-inelastic, bid at `voll`) and **flexible** (price-elastic, bid at `wtp`) categories.

Categories within each: `classical`, `industrial`, `household`, `public`, `ev`.

```yaml
demand_inflexible:
  classical:
    profile_year: 2020
    amount_file:  TYNDP_2024_National_Trends
    voll:         5000   # EUR/MWh — value of lost load
```

### Storage

```yaml
bess:
  units_file:           bess_units
  initial_soc:          0.5    # fraction of capacity at t=0
  roundtrip_efficiency: 0.85
```