import pandas as pd

df_kpi = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                        "02_Working/Export/20230419_113656/00_random.xlsx")

print(df_kpi)

kpi_nos = df_kpi.shape[0]
shade_nos = int(df_kpi.shape[1]/8)
orientations = int(df_kpi.shape[1]/12)

# Making an empty df to load the ranking of various KPI's as shade versus orientation

print(kpi_nos, shade_nos, orientations)

# Making an empty df to load the ranking of various KPI's as shade versus orientation
df_rating = pd.DataFrame(columns=['RB_05_L', 'RB_05_M', 'RB_05_D',
                                  'RB_10_L', 'RB_10_M', 'RB_10_D',
                                  'VB_25_L', 'VB_25_M', 'VB_25_D',
                                  'VB_50_L', 'VB_50_M', 'VB_50_D' ])

for i in range(kpi_nos):
    for j in range(orientations):
        # Select the ith row and the 8 columns
        shade_sel = [((x * 8)+j) for x in range(12)]
        row = df_kpi.iloc[i, shade_sel]
        print(row)

        # Transform each value in row
        min_val = row.min()
        max_val = row.max()
        row = (max_val - row) / (max_val - min_val)
        row = row.fillna(1) # replace NaN with 1

        print(row)

        # Add the rating values as a new row to df_rating
        row_label = f"Orientation {j+1}"
        df_rating.loc[row_label] = row.values

    if i == 0:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_Heating.xlsx', header=True, index=False)
    elif i == 1:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_Cooling.xlsx', header=True, index=False)
    elif i == 2:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_Lighting.xlsx', header=True, index=False)
    elif i == 3:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_PMV.xlsx', header=True, index=False)
    elif i == 4:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_PPD.xlsx', header=True, index=False)
    elif i == 5:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_UDI_1.xlsx', header=True, index=False)
    elif i == 6:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_UDI_3.xlsx', header=True, index=False)
    elif i == 7:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_UDI_5.xlsx', header=True, index=False)
    elif i == 8:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_DGP_1.xlsx', header=True, index=False)
    elif i == 9:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_DGP_3.xlsx', header=True, index=False)
    else:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_DGP_5.xlsx', header=True, index=False)

for i in range(5, 8):
    for j in range(orientations):
        # Select the ith row and the 8 columns
        shade_sel = [((x * 8) + j) for x in range(12)]
        row = df_kpi.iloc[i, shade_sel]
        print(row)

        # Calculate the range of values in the row
        value_range = row.max() - row.min()

        # Transform each value in row
        if value_range == 0:
            row = row.apply(lambda x: 0)
        else:
            row = row.apply(lambda x: (row.max() - x) / value_range)

        row = row.fillna(1) # replace NaN with 1

        print(row)

        # Add the rating values as a new row to df_rating
        row_label = f"Orientation {j+1}"
        df_rating.loc[row_label] = row.values

    if i == 5:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_UDI_1.xlsx', header=True, index=False)
    elif i == 6:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_UDI_3.xlsx', header=True, index=False)
    else:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_UDI_5.xlsx', header=True, index=False)


import pandas as pd

df_kpi = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                        "02_Working/Export/20230419_113656/00_random.xlsx")

print(df_kpi)

kpi_nos = df_kpi.shape[0]
shade_nos = int(df_kpi.shape[1]/8)
orientations = int(df_kpi.shape[1]/12)

# Making an empty df to load the ranking of various KPI's as shade versus orientation

print(kpi_nos, shade_nos, orientations)

# Making an empty df to load the ranking of various KPI's as shade versus orientation
df_rating = pd.DataFrame(columns=['RB_05_L', 'RB_05_M', 'RB_05_D',
                                  'RB_10_L', 'RB_10_M', 'RB_10_D',
                                  'VB_25_L', 'VB_25_M', 'VB_25_D',
                                  'VB_50_L', 'VB_50_M', 'VB_50_D' ])

for i in range(1):
    for j in range(orientations):
        # Select the ith row and the 8 columns
        shade_sel = [((x * 8)+j) for x in range(12)]
        row_heat = df_kpi.iloc[i, shade_sel]
        row_cool = df_kpi.iloc[(i+1), shade_sel]
        row_light = df_kpi.iloc[(i+2), shade_sel]
        row_total = row_heat + row_cool + row_light
        print(row_total)

        # Calculate the range of values in the row
        value_range = row_total.max() - row_total.min()

        # Transform each value in row
        if value_range == 0:
            row_total = row_total.apply(lambda x: 0)
        else:
            row_total = row_total.apply(lambda x: (row_total.max() - x) / value_range)

        row_total = row_total.fillna(1)  # replace NaN with 1

        print(row_total)

        # Add the rating values as a new row to df_rating
        row_label = f"Orientation {j + 1}"
        df_rating.loc[row_label] = row_total.values

    df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                       f'02_Working/Export/20230419_113656/00_Rating_11_Energy.xlsx', header=True, index=False)