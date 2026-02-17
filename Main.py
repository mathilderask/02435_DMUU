
#%% Load package

import pandas as pd
from SystemCharacteristics import get_fixed_data
from PlotsRestaurant import plot_HVAC_results


#%% Load and define data

OccRoom1 = pd.read_csv("OccupancyRoom1.csv")
OccRoom2 = pd.read_csv("OccupancyRoom2.csv")
Price = pd.read_csv("PriceData.csv")

T = 100

#%%