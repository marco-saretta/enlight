# Model

ENLIGHT solves one linear program per run (or per week in rolling horizon mode) with [linopy](https://linopy.readthedocs.io/): it maximises social welfare (consumer benefit minus production cost) subject to a power balance in every zone and hour.

## Sets

| Symbol | Dimension | Meaning |
|---|---|---|
| $t \in T$ | `T` | hours |
| $z \in Z$ | `Z` | bidding zones |
| $g \in G$ | `G` | thermal units (or zone-technology plants when aggregated) |
| $l \in L$ | `L` | transmission lines between active zones |

## Variables

Every market participant bids a volume at a price. In the build scripts each quantity carries its technology name: `<tech>_potential` or `<tech>_capacity` (input: upper limit), `<tech>_bid_price` (input: price), `<tech>_bid_volume` (decision: accepted volume).

| Variable | Dims | Bounds | Built in |
|---|---|---|---|
| `wind_onshore_bid_volume`, `wind_offshore_bid_volume`, `solar_pv_bid_volume`, `hydro_ror_bid_volume` | T × Z | $0 \le p \le$ potential | `build_<tech>.py` |
| `hydro_res_bid_volume` | T × G | $0 \le p \le$ capacity, weekly energy budget per zone | `build_hydro_res.py` |
| `thermal_bid_volume` | T × G | $0 \le p \le$ capacity | `build_thermal.py` |
| `classical_inflex_bid_volume` | T × Z | $0 \le d \le$ load | `build_demand_inflexible.py` |
| `lines_flow` | T × L | $-\bar{F}^{\,to \to from} \le f \le \bar{F}^{\,from \to to}$ | `build_lines.py` |

## Objective

$$
\min \; \sum_{t} \Big( \sum_{r,z} c_r \, p_{r,z,t} + \sum_{g} c_g \, p_{g,t} - \sum_{z} \text{VOLL} \cdot d_{z,t} \Big)
$$

with $c_r$ the renewable bid prices, $c_g$ the thermal marginal costs and VOLL the value of lost load. Demand that is not served is load shedding.

## Power balance

For every zone $z$ and hour $t$:

$$
\sum_{r} p_{r,z,t} + \sum_{g \in z} p_{g,t} - d_{z,t} + \sum_{l:\,to(l)=z} f_{l,t} - \sum_{l:\,from(l)=z} f_{l,t} = 0
$$

The **electricity price** of zone $z$ in hour $t$ is the dual of this constraint. Units and lines are mapped to zones by grouping on their zone labels, so no incidence matrix is built.

## Thermal marginal cost

Computed per unit in preprocessing, for the prediction year, and written with every component to `simulations/<label>/data/thermal_units.csv`:

$$
MC = \underbrace{\frac{\text{fuel price}}{\eta}}_{\text{fuel cost}}
   + \underbrace{\frac{\text{CO}_2\text{ price} \cdot \text{CO}_2\text{ intensity} \cdot (1 - \text{capture rate})}{\eta}}_{\text{CO}_2\text{ cost}}
   + \text{VOM}
$$

with $\eta$ the electric efficiency from the technology catalogue. Units lacking any input are dropped with a warning.

## Rolling horizon

Each week (168 h; week 52 takes the remaining 192 h) is solved as a separate model and the results are joined. The model has no constraints linking hours yet, so this reproduces the yearly solution; once storage is added, each week will have to start from the state of charge the previous one ended with.

## Adding a technology

1. Write `src/enlight/model/build_<tech>.py`: add the variables, then call `em.add_to_power_balance("<tech>", expr)` and `em.add_to_objective(expr)`.
2. Add the function to `BUILD_STEPS` in `energy_model.py`.

The exporter picks up the new term automatically from its label.
