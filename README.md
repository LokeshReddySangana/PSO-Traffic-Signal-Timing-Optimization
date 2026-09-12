# PSO Traffic Signal Timing Optimization

Particle Swarm Optimization (PSO) project for optimizing traffic signal timing across multi-junction networks.

## Features

- Dynamic low, medium, and high traffic phases
- Emergency vehicle priority override
- Two-junction traffic flow simulation
- Dataset-based arrival rates
- Per-lane queue management
- Waiting-time and fuel-waste estimates
- PSO optimization of green times
- Interactive HTML dashboard

## Files

- `PSO_Traffic_Signal_Timing_Optimization.ipynb`: Complete simulation, optimization, analysis, and visualizations.
- `pso_traffic_dashboard.html`: Interactive traffic signal dashboard.
- `pso_traffic_dataset.csv`: Traffic arrival-rate dataset.
- `traffic_pso_dataset.csv`: Additional traffic dataset.

## Running the Notebook

1. Open the notebook in VS Code or Jupyter.
2. Select a Python kernel with NumPy, pandas, and Matplotlib installed.
3. Run the cells in order.
4. The final cell opens the local interactive dashboard.

## Dashboard

Open `pso_traffic_dashboard.html` in a browser to explore the simulation interface, traffic presets, signal controls, emergency priority, and analytics.

## Note

The fuel and delay values are approximations intended for simulation and comparison, not traffic-control deployment.
