# Configuration

ENLIGHT is configured with [Hydra](https://hydra.cc/). Everything lives in `config/`:

| File | Content |
|---|---|
| `config/config.yaml` | Which scenario runs by default, and the pipeline `steps` |
| `config/paths/default.yaml` | Where raw data, simulations and logs are |
| `config/simulations/default.yaml` | The full scenario: every setting, with its default value |
| `config/simulations/<name>.yaml` | Other scenarios |

## Scenario settings

`default.yaml` is organised in five sections:

| Section | Main settings |
|---|---|
| `label`, `run` | Output folder name; `mode` (`yearly` \| `rolling_horizon`), `prediction_year`, `solver` (`highs` \| `gurobi`) |
| `rolling_horizon` | `start_week`, `end_week`, `keep_weekly_results` |
| `supply_curve` | One block per technology: bid prices, capacity projection files, weather data, thermal unit file and marginal cost datasets, `plant_aggregation` |
| `demand_curve` | Inflexible (`*_inflex`, bid at `voll`) and flexible (`*_flex`, bid at `wtp`) demand categories |
| `lines`, `bidding_zones` | Transmission capacity dataset; the zones in the model |

Every setting is documented with a comment in `default.yaml` itself.

## Creating a scenario

A scenario file lists only what differs from `default.yaml`; Hydra loads `default.yaml` first and applies the file on top:

```yaml
# config/simulations/demo1.yaml
defaults:
  - default
  - _self_

label: demo1          # must be unique: it names simulations/<label>/

run:
  mode: rolling_horizon

supply_curve:
  thermal:
    plant_aggregation: true
```

Nested blocks are merged key by key, but lists are replaced as a whole:

```yaml
bidding_zones: [AT, BE, FR, IT, CH, DELU, NL, ES]   # the complete list, not an addition
```

Run it with `uv run main.py simulations=demo1`.

## Validation

At start-up the composed scenario is checked against the schema in `src/enlight/utils/validation.py` (types, allowed values, required settings); a run with an invalid configuration stops before preprocessing.
