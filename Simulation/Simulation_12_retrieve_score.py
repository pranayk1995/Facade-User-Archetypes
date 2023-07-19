import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

df_energy = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                          "02_Working/Export/20230419_113656/00_OG_total.xlsx", index_col=0)

df_heat = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                          "02_Working/Export/20230419_113656/00_OG_heat.xlsx", index_col=0)

df_cool = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                          "02_Working/Export/20230419_113656/00_OG_cool.xlsx", index_col=0)

df_light = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                          "02_Working/Export/20230419_113656/00_OG_light.xlsx", index_col=0)

df_thermal = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                           "02_Working/Export/20230419_113656/00_OG_PMV_weighted.xlsx", index_col=0)

df_pmv = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                           "02_Working/Export/20230419_113656/00_OG_PMV.xlsx", index_col=0)

df_ppd = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                           "02_Working/Export/20230419_113656/00_OG_PPD.xlsx", index_col=0)

df_udi_1 = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                         "02_Working/Export/20230419_113656/00_OG_UDI_1.xlsx", index_col=0)

df_udi_3 = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                         "02_Working/Export/20230419_113656/00_OG_UDI_3.xlsx", index_col=0)

df_udi_5 = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                         "02_Working/Export/20230419_113656/00_OG_UDI_5.xlsx", index_col=0)

df_dgp_1 = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                         "02_Working/Export/20230419_113656/00_OG_DGP_1.xlsx", index_col=0)

df_dgp_3 = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                         "02_Working/Export/20230419_113656/00_OG_DGP_3.xlsx", index_col=0)

df_dgp_5 = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                         "02_Working/Export/20230419_113656/00_OG_DGP_5.xlsx", index_col=0)

output = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                     f"02_Working/Export/20230419_113656/edit_output_0_0.csv")

num_occupied = (output["ROOM_453_5DC66EC0_SPACE PEOPLE_ROOM1:People Occupant Count [](Hourly)"] > 0).sum()

print(num_occupied)

color_1_base = '#6D909C'
color_1_base_L1 = '#7f9da8'
color_1_base_L2 = '#95aeb7'
color_1_base_dark = '#5B7C86'
color_1_base_light = '#C2D1D6'
color_2_base = '#AB8788'
color_2_base_L1 = '#b79899'
color_2_base_L2 = '#bfa4a5'
color_3_base = '#8fb3b0'
color_3_base_L1 = '#9abbb8'
color_3_base_L2 = '#a1c0bd'
color_4_base = '#d6b680'
color_4_base_L1 = '#ddc397'
color_4_base_L2 = '#e4cfab'
color_0_background = '#f5f5f5'
color_map_2_wtob = mcolors.LinearSegmentedColormap.from_list('my_color_map', ['#336371', '#FAF9F6'])
color_map_2_btow = mcolors.LinearSegmentedColormap.from_list('my_color_map', ['#FAF9F6', '#336371'])
fs = 20

color_map = {'RB_05_L': color_1_base, 'RB_05_M': color_1_base, 'RB_05_D': color_1_base,
             'RB_10_L': color_2_base, 'RB_10_M': color_2_base, 'RB_10_D': color_2_base,
             'VB_25_L': color_3_base, 'VB_25_M': color_3_base, 'VB_25_D': color_3_base,
             'VB_50_L': color_4_base, 'VB_50_M': color_4_base, 'VB_50_D': color_4_base}

thickness_map = {'RB_05_L': 1.0, 'RB_05_M': 2.0, 'RB_05_D': 3.0,
                 'RB_10_L': 1.0, 'RB_10_M': 2.0, 'RB_10_D': 3.0,
                 'VB_25_L': 1.0, 'VB_25_M': 2.0, 'VB_25_D': 3.0,
                 'VB_50_L': 1.0, 'VB_50_M': 2.0, 'VB_50_D': 3.0}

linetype_map = {'RB_05_L': '-', 'RB_05_M': '--', 'RB_05_D': '-.',
                'RB_10_L': '-', 'RB_10_M': '--', 'RB_10_D': '-.',
                'VB_25_L': '-', 'VB_25_M': '--', 'VB_25_D': '-.',
                'VB_50_L': '-', 'VB_50_M': '--', 'VB_50_D': '-.'}

dataframes = [df_energy, df_heat, df_cool, df_light, df_thermal, df_pmv, df_ppd, df_udi_1, df_udi_3, df_udi_5, df_dgp_1, df_dgp_3, df_dgp_5]

row_index = 'East'  # Specify the row index you want to access
column_name = 'RB_10_L'  # Specify the column name you want to access

for df in dataframes:
    value = df.loc[row_index, column_name]
    print(value)