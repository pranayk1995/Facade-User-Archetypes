import pandas as pd
import numpy as np

df_heat = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                        f"02_Working/Export/20230419_113656/00_heating.xlsx")

df_cool = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                        f"02_Working/Export/20230419_113656/00_cooling.xlsx")

df_udi = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                        f"02_Working/Export/20230419_113656/00_UDI_100_3000_clean.xlsx")

df_dgp = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                        f"02_Working/Export/20230419_113656/00_DGP.xlsx")

udi_low = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                        f"02_Working/Export/20230419_113656/00_UDI_LOW.csv")

df_pmv = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                       f"02_Working/Export/20230419_113656/00_pmv.xlsx")

df_ppd = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                       f"02_Working/Export/20230419_113656/00_ppd.xlsx")

kpi = pd.DataFrame()

# Energy
# Summing of heating and cooling loads
heating_load = df_heat.sum(axis=0).values / (3600000 * 30)
cooling_load = df_cool.sum(axis=0).values / (3600000 * 30)

# Understanding lighting loads due to inadequate illuminance levels.
output = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                     f"02_Working/Export/20230419_113656/edit_output_0_0.csv")
num_occupied = (output["ROOM_453_5DC66EC0_SPACE PEOPLE_ROOM1:People Occupant Count [](Hourly)"] > 0).sum()
udi_low_hours = (udi_low["udilow_shade_5"] * (num_occupied / 100)).values
lighting_load = udi_low_hours * (5 / 1000)

# Thermal
# get the index of all non-zero values in column
index_occupied = np.nonzero(output['ROOM_453_5DC66EC0_SPACE PEOPLE_ROOM1:People Occupant Count [](Hourly)'].values)[0]
print(index_occupied)

print("1")

# Filter rows in df1 and df2 based on the indices
pmv_occupied = (df_pmv.loc[index_occupied]).abs()
ppd_occupied = df_ppd.loc[index_occupied]

print(pmv_occupied)
print(ppd_occupied)

# calculate sum of absolute differences from 0
pmv_diff = pmv_occupied.sum().values
ppd_diff = ppd_occupied.sum().values

df_thermal_weighted = pmv_occupied * (ppd_occupied/100)
df_thermal_weighted = df_thermal_weighted.sum().values

print(df_thermal_weighted)

# Visual / Daylight
# Get UDI Values and DGP Values into the dataframe.

udi_1m = df_udi['pt3']
udi_3m = df_udi['pt4']
udi_5m = df_udi['pt5']
dgp_1m = df_dgp['someshade_1m']
dgp_3m = df_dgp['someshade_3m']
dgp_5m = df_dgp['someshade_5m']

# New df to add the KPI's
kpi = pd.DataFrame([heating_load, cooling_load, lighting_load, pmv_diff, ppd_diff, udi_1m, udi_3m, udi_5m, dgp_1m,
                    dgp_3m, dgp_5m, df_thermal_weighted])

kpi.to_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                "02_Working/Export/20230419_113656/00_random.xlsx", header=True, index=False)



