# ENLIGHT Model Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Sets and Indices](#sets-and-indices)
3. [Parameters](#parameters)
4. [Variables](#variables)
5. [Constraints](#constraints)
6. [Objective Function](#objective-function)
7. [Model Assumptions](#model-assumptions)



## Introduction

ENLIGHT implements a **day-ahead electricity market clearing** optimization model that replicates the European market structure. The model solves a **social welfare maximization** problem subject to physical and operational constraints, determining:

- Optimal dispatch schedules for all generation units
- Market clearing prices for each bidding zone
- Cross-border electricity flows
- Storage operation schedules

### Market Design Principles

The model follows European day-ahead market conventions:

- **Zonal pricing**: Single price per bidding zone per hour
- **DC power flow**: Linear approximation of transmission constraints
- **Perfect competition**: All agents are price-takers
- **Merit order dispatch**: Units bid at long-run marginal cost (LRMC)
- **Market clearing**: Supply and demand cleared simultaneously across all zones

### Mathematical Formulation

The optimization problem can be expressed as:

$$
\max_{x} \quad SW(x) = \text{Consumer Surplus} - \text{Production Cost}
$$

Subject to:
- Power balance in each zone and time period
- Generation capacity limits
- Transmission capacity limits
- Storage dynamics
- Hydro reservoir energy availability
- Flexible demand limits

The dual variables of the power balance constraints represent the **market clearing prices** (electricity prices) in each zone.



## Sets and Indices

| Symbol | Description | Cardinality |
|--------|-------------|-------------|
| $t \in \mathcal{T}$ | Time periods (hours) | $T$ = 168 (weekly) or 8760 (yearly) |
| $z \in \mathcal{Z}$ | Bidding zones | $Z$ = number of zones |
| $w \in \mathcal{W}$ | Weeks (yearly mode only) | $W$ = 52 |
| $g \in \mathcal{G}$ | Conventional generation units | $G$ |
| $h \in \mathcal{H}^{res}$ | Hydro reservoir units | $G_{hydro}^{res}$ |
| $h \in \mathcal{H}^{ps}$ | Pumped hydro storage units | $G_{hydro}^{ps}$ |
| $b \in \mathcal{B}$ | Battery energy storage systems (BESS) | $G_{bess}$ |
| $p \in \mathcal{P}$ | Power-to-X units | $L_{PtX}$ |
| $d \in \mathcal{D}$ | District heating units | $L_{DH}$ |
| $l \in \mathcal{L}$ | Transmission lines | $L$ |

### Incidence Matrices

Binary mapping matrices connect units to zones:

- $\mathbf{M}^{G,Z} \in \{0,1\}^{G \times Z}$: Conventional units to zones
- $\mathbf{M}^{H^{res},Z} \in \{0,1\}^{G_{hydro}^{res} \times Z}$: Hydro reservoir to zones
- $\mathbf{M}^{H^{ps},Z} \in \{0,1\}^{G_{hydro}^{ps} \times Z}$: Pumped hydro to zones
- $\mathbf{M}^{B,Z} \in \{0,1\}^{G_{bess} \times Z}$: BESS to zones
- $\mathbf{M}^{P,Z} \in \{0,1\}^{L_{PtX} \times Z}$: PtX to zones
- $\mathbf{M}^{D,Z} \in \{0,1\}^{L_{DH} \times Z}$: District heating to zones
- $\mathbf{M}^{L,Z} \in \{-1,0,1\}^{L \times Z}$: Lines to zones (±1 for direction, 0 otherwise)


## Parameters

### Renewable Generation Profiles

| Parameter | Unit | Description |
|-----------|------|-------------|
| $\overline{P}_{t,z}^{wind,on}$ | MW | Onshore wind availability |
| $\overline{P}_{t,z}^{wind,off}$ | MW | Offshore wind availability |
| $\overline{P}_{t,z}^{solar}$ | MW | Solar PV availability |
| $\overline{P}_{t,z}^{hydro,ror}$ | MW | Run-of-river hydro availability |

### Demand Parameters

| Parameter | Unit | Description |
|-----------|------|-------------|
| $D_{t,z}^{inflex}$ | MW | Inflexible demand |
| $\overline{D}_{t,z}^{flex}$ | MW | Flexible demand capacity |
| $E_z^{flex}$ | MWh | Total flexible demand energy (weekly) |
| $VOLL_{classic}$ | €/MWh | Value of lost load (inflexible demand) |
| $WTP_{classic}$ | €/MWh | Willingness to pay (flexible demand) |

### Generation Unit Parameters

**Conventional units:**
- $\overline{P}_g^{conv}$: Capacity [MW]
- $c_g^{conv}$: Marginal cost [€/MWh]

**Hydro reservoir:**
- $\overline{P}_h^{res}$: Generation capacity [MW]
- $E_{w,z}^{hydro,res}$: Weekly energy availability [MWh]
- $c_h^{res}$: Marginal cost [€/MWh]

**Pumped hydro storage:**
- $\overline{P}_h^{ps,charge}$: Charging capacity [MW]
- $\overline{P}_h^{ps,discharge}$: Discharging capacity [MW]
- $\overline{E}_h^{ps}$: Storage capacity [MWh]
- $\eta_h^{ps,charge}$: Charging efficiency [%]
- $\eta_h^{ps,discharge}$: Discharging efficiency [%]
- $SOC_h^{ps,init}$: Initial state of charge [%]
- $c_h^{ps,charge}$: Charging bid price [€/MWh]
- $c_h^{ps,discharge}$: Discharging offer price [€/MWh]

**Battery energy storage (BESS):**
- $\overline{P}_b^{bess,charge}$: Charging capacity [MW]
- $\overline{P}_b^{bess,discharge}$: Discharging capacity [MW]
- $\overline{E}_b^{bess}$: Storage capacity [MWh]
- $\eta_b^{bess,charge}$: Charging efficiency [%]
- $\eta_b^{bess,discharge}$: Discharging efficiency [%]
- $SOC_b^{bess,init}$: Initial state of charge [%]
- $c_b^{bess,charge}$: Charging bid price [€/MWh]
- $c_b^{bess,discharge}$: Discharging offer price [€/MWh]

### Renewable Bid Prices

| Parameter | Unit | Description |
|-----------|------|-------------|
| $c^{wind,on}$ | €/MWh | Onshore wind bid price |
| $c^{wind,off}$ | €/MWh | Offshore wind bid price |
| $c^{solar}$ | €/MWh | Solar PV bid price |
| $c^{hydro,ror}$ | €/MWh | Run-of-river bid price |

### Power-to-X and District Heating

| Parameter | Unit | Description |
|-----------|------|-------------|
| $\overline{P}_p^{ptx}$ | MW | PtX unit capacity |
| $c_p^{ptx}$ | €/MWh | PtX bid price (willingness to pay) |
| $\overline{P}_d^{dh}$ | MW | District heating unit capacity |
| $c_d^{dh}$ | €/MWh | District heating bid price |

### Transmission Network

| Parameter | Unit | Description |
|-----------|------|-------------|
| $\overline{F}_l^{a \to b}$ | MW | Line capacity A to B |
| $\overline{F}_l^{b \to a}$ | MW | Line capacity B to A |



## Variables

### Generation and Load Variables

| Variable | Domain | Unit | Description |
|----------|--------|------|-------------|
| $p_{t,z}^{wind,on}$ | $[0, \overline{P}_{t,z}^{wind,on}]$ | MW | Onshore wind production |
| $p_{t,z}^{wind,off}$ | $[0, \overline{P}_{t,z}^{wind,off}]$ | MW | Offshore wind production |
| $p_{t,z}^{solar}$ | $[0, \overline{P}_{t,z}^{solar}]$ | MW | Solar PV production |
| $p_{t,z}^{hydro,ror}$ | $[0, \overline{P}_{t,z}^{hydro,ror}]$ | MW | Run-of-river hydro production |
| $d_{t,z}^{inflex}$ | $[0, D_{t,z}^{inflex}]$ | MW | Inflexible demand served |
| $d_{t,z}^{flex}$ | $[0, \overline{D}_{t,z}^{flex}]$ | MW | Flexible demand served |

### Conventional and Hydro Reservoir Generation

| Variable | Domain | Unit | Description |
|----------|--------|------|-------------|
| $p_{t,g}^{conv}$ | $[0, \overline{P}_g^{conv}]$ | MW | Conventional unit production |
| $p_{t,h}^{res}$ | $[0, \overline{P}_h^{res}]$ | MW | Hydro reservoir production |

### Storage Variables (Pumped Hydro)

| Variable | Domain | Unit | Description |
|----------|--------|------|-------------|
| $p_{t,h}^{ps,charge}$ | $[0, \overline{P}_h^{ps,charge}]$ | MW | Pumped hydro charging |
| $p_{t,h}^{ps,discharge}$ | $[0, \overline{P}_h^{ps,discharge}]$ | MW | Pumped hydro discharging |
| $e_{t,h}^{ps}$ | $[0, \overline{E}_h^{ps}]$ | MWh | Pumped hydro state of charge |

### Storage Variables (BESS)

| Variable | Domain | Unit | Description |
|----------|--------|------|-------------|
| $p_{t,b}^{bess,charge}$ | $[0, \overline{P}_b^{bess,charge}]$ | MW | BESS charging |
| $p_{t,b}^{bess,discharge}$ | $[0, \overline{P}_b^{bess,discharge}]$ | MW | BESS discharging |
| $e_{t,b}^{bess}$ | $[0, \overline{E}_b^{bess}]$ | MWh | BESS state of charge |

### Sector Coupling Variables

| Variable | Domain | Unit | Description |
|----------|--------|------|-------------|
| $p_{t,p}^{ptx}$ | $[0, \overline{P}_p^{ptx}]$ | MW | Power-to-X consumption |
| $p_{t,d}^{dh}$ | $[0, \overline{P}_d^{dh}]$ | MW | District heating consumption |

### Transmission Variables

| Variable | Domain | Unit | Description |
|----------|--------|------|-------------|
| $f_{t,l}$ | $[-\overline{F}_l^{b \to a}, \overline{F}_l^{a \to b}]$ | MW | Line flow (positive: A→B) |
| $ex_{t,z}$ | $\mathbb{R}$ | MW | Net export from zone |



## Constraints

### 1. Power Balance

**Aggregated plants mode:**

$$
\begin{align}
&p_{t,z}^{wind,on} + p_{t,z}^{wind,off} + p_{t,z}^{solar} + p_{t,z}^{hydro,ror} \\
&+ \sum_{g \in \mathcal{G}} p_{t,g}^{conv} \cdot M_{g,z}^{G,Z} \\
&+ \sum_{h \in \mathcal{H}^{res}} p_{t,h}^{res} \cdot M_{h,z}^{H^{res},Z} \\
&+ p_{t,z}^{ps,discharge} + p_{t,z}^{bess,discharge} \\
&= d_{t,z}^{inflex} + d_{t,z}^{flex} + ex_{t,z} \\
&+ p_{t,z}^{ps,charge} + p_{t,z}^{bess,charge} \\
&+ \sum_{p \in \mathcal{P}} p_{t,p}^{ptx} \cdot M_{p,z}^{P,Z} \\
&+ \sum_{d \in \mathcal{D}} p_{t,d}^{dh} \cdot M_{d,z}^{D,Z}
\end{align}
\quad \forall t \in \mathcal{T}, z \in \mathcal{Z}
$$

**Disaggregated plants mode:** Similar structure but storage variables indexed by individual units.

**Interpretation:** Total generation in each zone equals total consumption plus net exports at every hour.



### 2. Electricity Exports (Network Flow)

$$
\sum_{l \in \mathcal{L}} f_{t,l} \cdot M_{l,z}^{L,Z} = ex_{t,z} \quad \forall t \in \mathcal{T}, z \in \mathcal{Z}
$$

**Interpretation:** Net exports equal the algebraic sum of flows on connected lines. $M_{l,z}^{L,Z} = +1$ if line $l$ exports from zone $z$, $-1$ if importing, $0$ otherwise.



### 3. Hydro Reservoir Energy Availability

**Yearly mode:**

$$
\sum_{t \in \mathcal{T}_w} \sum_{h \in \mathcal{H}^{res}} p_{t,h}^{res} \cdot M_{h,z}^{H^{res},Z} \leq E_{w,z}^{hydro,res} \quad \forall w \in \mathcal{W}, z \in \mathcal{Z}
$$

where $\mathcal{T}_w$ is the set of hours in week $w$.

**Weekly mode:**

$$
\sum_{t \in \mathcal{T}} \sum_{h \in \mathcal{H}^{res}} p_{t,h}^{res} \cdot M_{h,z}^{H^{res},Z} \leq E_z^{hydro,res} \quad \forall z \in \mathcal{Z}
$$

**Interpretation:** Total hydro reservoir energy produced in each week cannot exceed available water energy budget.



### 4. Flexible Demand Energy Limit

$$
\sum_{t \in \mathcal{T}_w} d_{t,z}^{flex} \leq E_z^{flex} \quad \forall w \in \mathcal{W}, z \in \mathcal{Z}
$$

**Interpretation:** Total flexible demand served per week cannot exceed the contracted energy amount.



### 5. Pumped Hydro Storage Dynamics

$$
e_{t,h}^{ps} - e_{t-1,h}^{ps} - \mathbb{1}_{t=1} \cdot SOC_h^{ps,init} \cdot \overline{E}_h^{ps} = \eta_h^{ps,charge} \cdot p_{t,h}^{ps,charge} - \frac{p_{t,h}^{ps,discharge}}{\eta_h^{ps,discharge}}
$$

$$
\forall t \in \mathcal{T}, h \in \mathcal{H}^{ps}
$$

where $\mathbb{1}_{t=1}$ equals 1 when $t=1$ and 0 otherwise.

**Interpretation:** Change in stored energy equals charging (with losses) minus discharging (with losses). Initial condition added at $t=1$.


### 6. BESS Storage Dynamics

$$
e_{t,b}^{bess} - e_{t-1,b}^{bess} - \mathbb{1}_{t=1} \cdot SOC_b^{bess,init} \cdot \overline{E}_b^{bess} = \eta_b^{bess,charge} \cdot p_{t,b}^{bess,charge} - \frac{p_{t,b}^{bess,discharge}}{\eta_b^{bess,discharge}}
$$

$$
\forall t \in \mathcal{T}, b \in \mathcal{B}
$$

**Interpretation:** Identical to pumped hydro storage dynamics but for batteries.



## Objective Function

The model **minimizes negative social welfare**, which is equivalent to **maximizing social welfare**:

$$
\min \quad -SW = \sum_{t \in \mathcal{T}} \left[ \text{Production Cost}_t - \text{Consumer Surplus}_t \right]
$$

### Expanded Formulation

$$
\begin{align}
\min \quad & \sum_{t,z} \Bigg[ 
    c^{wind,on} \cdot p_{t,z}^{wind,on} + c^{wind,off} \cdot p_{t,z}^{wind,off} \\
    &+ c^{solar} \cdot p_{t,z}^{solar} + c^{hydro,ror} \cdot p_{t,z}^{hydro,ror} \Bigg] \\
    %
    &+ \sum_{t,g} c_g^{conv} \cdot p_{t,g}^{conv} \\
    %
    &+ \sum_{t,h \in \mathcal{H}^{res}} c_h^{res} \cdot p_{t,h}^{res} \\
    %
    &+ \sum_{t,h \in \mathcal{H}^{ps}} \left[ c_h^{ps,discharge} \cdot p_{t,h}^{ps,discharge} - c_h^{ps,charge} \cdot p_{t,h}^{ps,charge} \right] \\
    %
    &+ \sum_{t,b} \left[ c_b^{bess,discharge} \cdot p_{t,b}^{bess,discharge} - c_b^{bess,charge} \cdot p_{t,b}^{bess,charge} \right] \\
    %
    &- \sum_{t,z} \left[ VOLL_{classic} \cdot d_{t,z}^{inflex} + WTP_{classic} \cdot d_{t,z}^{flex} \right] \\
    %
    &- \sum_{t,p} c_p^{ptx} \cdot p_{t,p}^{ptx} \\
    %
    &- \sum_{t,d} c_d^{dh} \cdot p_{t,d}^{dh}
\end{align}
$$

### Component Interpretation

**Production costs** (positive terms):
- Renewable generation at bid prices (typically near-zero)
- Conventional generation at marginal costs (fuel + CO₂)
- Hydro reservoir at opportunity cost
- Storage discharge at offer prices

**Consumer surplus** (negative terms):
- Inflexible demand valued at VOLL (very high willingness to pay)
- Flexible demand valued at WTP (lower, contract-dependent)
- Storage charging as "negative generation" (flexible demand)
- PtX and district heating as price-elastic consumption

**Market clearing prices** are the dual variables (shadow prices) of the power balance constraints, representing the marginal value of electricity in each zone and hour.



## Model Assumptions

### Economic Assumptions
1. **Perfect competition**: All market participants are price-takers
2. **Cost-based bidding**: Generators bid at long-run marginal cost (LRMC)
3. **No strategic behavior**: No gaming or market power
4. **Simultaneous market clearing**: All zones cleared simultaneously

### Technical Assumptions
5. **DC power flow**: Linearized approximation of AC power flow
6. **Lossless transmission**: No resistive losses on lines
7. **No unit commitment**: No minimum generation levels, startup costs, or ramping constraints
8. **Perfect foresight**: Full knowledge of renewable generation and demand profiles
9. **Weekly/yearly decomposition**: No inter-temporal coupling beyond weekly hydro budgets

### Operational Assumptions
10. **Fixed renewable profiles**: Exogenous wind, solar, and hydro ROR availability
11. **Inflexible demand profile**: Classical demand follows fixed pattern, only magnitude is optimized
12. **Storage initial conditions**: Hydro and BESS start at specified state of charge
13. **No must-run constraints**: All generation is economically dispatched

### Simplifications
14. **Aggregation options**: Plants can be aggregated to zones for computational efficiency
15. **No ancillary services**: No reserves, frequency response, or voltage support
16. **No curtailment costs**: Renewable curtailment has zero explicit cost
17. **No demand response beyond flexible demand**: No price-responsive demand curves



## Model Properties

### Linearity
The model is a **linear program (LP)** with:
- Linear objective function
- Linear constraints
- Continuous variables

This ensures:
- Unique optimal solution (if feasible and bounded)
- Dual variables represent exact marginal prices
- Solvable with efficient LP algorithms (simplex, interior point)

### Convexity
The feasible region is a convex polytope, guaranteeing:
- Local optimum = global optimum
- Convex dual problem
- Strong duality holds (primal = dual objective at optimum)

### Duality and Pricing
From the **KKT conditions**, dual variables of power balance constraints equal market clearing prices:

$$
\lambda_{t,z} = \frac{\partial SW}{\partial d_{t,z}} = \text{marginal value of electricity}
$$

This is the fundamental economic interpretation: **electricity price equals marginal generation cost** at the optimal dispatch.


## References

1. Schweppe, F. C., et al. (1988). *Spot Pricing of Electricity*. Springer.
2. Stoft, S. (2002). *Power System Economics: Designing Markets for Electricity*. Wiley-IEEE Press.
3. Kirschen, D. S., & Strbac, G. (2018). *Fundamentals of Power System Economics*. Wiley.
4. ENTSO-E (2024). *Day-Ahead Market Coupling*.


---

[Back to Table of Contents](./index.md)