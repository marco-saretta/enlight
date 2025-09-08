# Architecture

## High-Level Architecture

![Architecture Diagram](../images/architecture.png)

## Components

- **EnergyHubRunner**: Manages the execution of energy system simulations for a specific scenario and year.
- **DataLoader**: Loads scenario-specific data for the simulation.
- **EnergySystemModel**: Initializes and configures the energy system model.
- **Energy Models**: Includes various models like MethanolProduction, MethaneProduction, AmmoniaProduction, etc.
- **Optimization**: Configures the optimization objective, finalizes the model, and runs the solver to optimize the energy system.

## Data Flow

1. Load configuration from YAML files.
2. Initialize the `EnergyHubRunner`.
3. Load scenario data using `DataLoader`.
4. Configure and initialize the `EnergySystemModel`.
5. Add energy system components and models.
6. Run the optimization process.
7. Collect and return results.

### Detailed Component Interactions

- **Scenario Loading**: Retrieves configuration settings from YAML files.
- **Model Initialization**: Sets up the energy system model with necessary parameters and variables.
- **Optimization**: Runs the optimization solver to determine optimal values for energy system variables.

## 
[Return to documentation index](./index.md)