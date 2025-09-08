# Usage

## Basic Usage

```python
import energyhubx as egx

scenario = 'sample_scenario'
year = 2025
timesteps = 8760  # Example: Number of hourly timesteps in a year

runner = egx.EnergyHubRunner(scenario, year, timesteps)
runner.execute()
```

## 
[Return to documentation index](./index.md)