import linopy as lp
import pandas as pd

# --- Parameters ---
units = ["A", "B"]
mc = pd.Series({"A": 20, "B": 60})        # marginal cost €/MWh
pmax = pd.Series({"A": 100, "B": 100})    # MW
pmin = pd.Series({"A": 0, "B": 0})        # MW

demand = 100                              # MW energy demand
reserve_req = 10                          # MW upward reserve
voll = 5000
# --- Model ---
m = lp.Model()

# Decision variables: energy (E) and upward reserve (Ru)
E = m.add_variables(name="E", lower=pmin, upper=pmax)
Ru = m.add_variables(name="Ru", lower=0, upper=pmax)

spot_demand = m.add_variables(name="spot_demand", lower=0, upper=demand)
fcr_demand = m.add_variables(name="fcr_demand", lower=reserve_req, upper=reserve_req)


# Capacity constraint: E_i + Ru_i <= Pmax_i
m.add_constraints(E + Ru <= pmax)
m.add_constraints(E - Ru >= pmin, name="pmin")

# Energy balance
m.add_constraints(E.sum() == spot_demand, name='spot_balance')

# Reserve requirement
m.add_constraints(Ru.sum() == fcr_demand, name='reserve_balance')

# Objective: Minimize cost (energy MC, assume zero physical reserve cost)
m.add_objective( - (voll * spot_demand 
                    -(E * mc).sum()
                    ),
                sense="min")

# Solve
m.solve()

# --- Results ---
print("\nDispatch:")
print(pd.DataFrame({"E": E.solution, "Ru": Ru.solution}))

# --- Prices ---
# Linopy stores duals in constraint attributes
spot_price = m.constraints['spot_balance'].dual  # Energy balance constraint dual
reserve_price = m.constraints['reserve_balance'].dual  # Reserve requirement dual

print(f"\nSpot price: {spot_price:.2f} €/MWh")
print(f"Reserve price: {reserve_price:.2f} €/MW·h")
