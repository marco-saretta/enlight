import linopy
import numpy as np

# Create model
model = linopy.Model()

# Define variables x and y ≥ 0 with shape (200, 10)
x = model.add_variables(lower=0, name="x", coords=[np.arange(1000), np.arange(100)])
y = model.add_variables(lower=0, name="y", coords=[np.arange(1000), np.arange(100)])

# Objective: minimize the total sum of x + y
model.add_objective((x + y).sum())

# Constraint 2: x + 2*y ≥ 4 elementwise
model.add_constraints(x + 2 * y >= 4, name="constraint")

# Solve the model
model.solve(solver_name='gurobi')

# Print solution for first few entries (optional)
model.solution