# Quick Start

## 1. Pick or create a simulation config

Simulation configs live in `configs/simulations/`. Copy the template and rename it:

```bash
cp configs/simulations/_template.yaml configs/simulations/my_sim.yaml
```

Edit `my_sim.yaml` to set the run mode, prediction year, and any parameters you want to change.

## 2. Set the active simulation

Open `configs/default_config.yaml` and point the `simulations` default at your file:

```yaml
defaults:
  - simulations: my_sim   # ← your filename without .yaml
```

## 3. Run

```bash
# Single simulation
uv run main.py

# Or explicitly select a config from the CLI
uv run main.py simulations=my_sim
```

## 4. Multi-run (sweep over all configs)

```bash
uv run main.py --multirun simulations='glob(*)'

# Or select specific configs
uv run main.py --multirun simulations=sim_1,sim_2
```

## 5. Find the results

```text
simulations/<label>/
├── data/      # preprocessed inputs
└── results/   # prices, dispatch, flows, curtailment
```

The `label` field in your simulation config controls the output folder name.