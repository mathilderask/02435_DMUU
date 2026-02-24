#%% Import packages

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
import Data.SystemCharacteristics as SC


from gurobipy import GRB
from Functions import solve_day_milp, plot_results




#%% Load data

price_data = pd.read_csv("Data/PriceData.csv")
occ_room1 = pd.read_csv("Data/OccupancyRoom1.csv")
occ_room2 = pd.read_csv("Data/OccupancyRoom2.csv")

price = price_data.to_numpy()
occ1  = occ_room1.to_numpy()
occ2  = occ_room2.to_numpy()

num_days, H = price.shape
assert occ1.shape == (num_days, H)
assert occ2.shape == (num_days, H)




#%% Define parameters

fixed = SC.get_fixed_data()

# Check horizon matches
assert fixed["num_timeslots"] == H, "Mismatch: data hours vs SystemCharacteristics num_timeslots"

params = {}

# Power limits (same max power for both rooms in the provided file)
Pmax = float(fixed["heating_max_power"])
params["P_heater"] = {1: Pmax, 2: Pmax}

# Ventilation power (kW)
params["P_vent"] = float(fixed["ventilation_power"])

# Temperature coefficients
params["z_exch"] = float(fixed["heat_exchange_coeff"])
params["z_loss"] = float(fixed["thermal_loss_coeff"])
params["z_conv"] = float(fixed["heating_efficiency_coeff"])
params["z_cool"] = float(fixed["heat_vent_coeff"])          # cooling when ventilation ON
params["z_occ"]  = float(fixed["heat_occupancy_coeff"])

# Humidity coefficients
params["eta_occ"]  = float(fixed["humidity_occupancy_coeff"])
params["eta_vent"] = float(fixed["humidity_vent_coeff"])    # humidity removed when vent ON

# Thresholds
params["T_low"]  = float(fixed["temp_min_comfort_threshold"])
params["T_ok"]   = float(fixed["temp_OK_threshold"])
params["T_high"] = float(fixed["temp_max_comfort_threshold"])
params["H_high"] = float(fixed["humidity_threshold"])

# Outdoor temperature time series
params["T_out"] = np.array(fixed["outdoor_temperature"], dtype=float)

# Initial conditions (same for both rooms in the fixed file)
T0 = float(fixed["initial_temperature"])
params["T_init"] = {1: T0, 2: T0}
params["H_init"] = float(fixed["initial_humidity"])

# Ventilation minimum up-time (in hours)
params["vent_min_up_time"] = int(fixed["vent_min_up_time"])




#%% Calculate Average Cost

daily_costs = []
solutions = {}

example_days = [30, 72]  # pick any two days for plotting later

for d in range(num_days):
    sol = solve_day_milp(
        prices=price[d, :],
        occ1_day=occ1[d, :],
        occ2_day=occ2[d, :],
        params=params,
        output_flag=0
    )
    daily_costs.append(sol["obj"])

    if d in example_days:
        solutions[d] = sol

avg_cost = float(np.mean(daily_costs))
print("Average daily electricity cost over", num_days, "days:", avg_cost)




#%% Plot Results

for d in example_days:
    plot_results(solutions[d], price, occ1, occ2, d)