import pandas as pd

df_kpi = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                        "02_Working/Export/20230419_113656/00_random.xlsx")

df_udi_raw = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                            "02_Working/Export/20230419_113656/00_UDI_100_3000_clean.xlsx")

print(df_udi_raw)

df_udi_noshade = df_udi_raw.iloc[0:8, 5:8]

print(df_udi_noshade)

kpi_nos = df_kpi.shape[0]
shade_nos = int(df_kpi.shape[1]/8)
orientations = int(df_kpi.shape[1]/12)
row_names = ['South', 'South West', 'West', 'North West', 'North', 'North East', 'East', 'South East']

# Making an empty df to load the ranking of various KPI's as shade versus orientation

print(kpi_nos, shade_nos, orientations)

# Making an empty df to load the ranking of various KPI's as shade versus orientation
df_rating = pd.DataFrame(columns=['RB_05_L', 'RB_05_M', 'RB_05_D',
                                  'RB_10_L', 'RB_10_M', 'RB_10_D',
                                  'VB_25_L', 'VB_25_M', 'VB_25_D',
                                  'VB_50_L', 'VB_50_M', 'VB_50_D' ])

df_scores_1 = pd.DataFrame(columns=['RB_05_L', 'RB_05_M', 'RB_05_D',
                                    'RB_10_L', 'RB_10_M', 'RB_10_D',
                                    'VB_25_L', 'VB_25_M', 'VB_25_D',
                                    'VB_50_L', 'VB_50_M', 'VB_50_D' ])

df_scores_2 = pd.DataFrame(columns=['RB_05_L', 'RB_05_M', 'RB_05_D',
                                    'RB_10_L', 'RB_10_M', 'RB_10_D',
                                    'VB_25_L', 'VB_25_M', 'VB_25_D',
                                    'VB_50_L', 'VB_50_M', 'VB_50_D' ])

df_scores_3 = pd.DataFrame(columns=['RB_05_L', 'RB_05_M', 'RB_05_D',
                                    'RB_10_L', 'RB_10_M', 'RB_10_D',
                                    'VB_25_L', 'VB_25_M', 'VB_25_D',
                                    'VB_50_L', 'VB_50_M', 'VB_50_D' ])

df_scores_4 = pd.DataFrame(columns=['RB_05_L', 'RB_05_M', 'RB_05_D',
                                    'RB_10_L', 'RB_10_M', 'RB_10_D',
                                    'VB_25_L', 'VB_25_M', 'VB_25_D',
                                    'VB_50_L', 'VB_50_M', 'VB_50_D' ])

for i in range(1):
    for j in range(orientations):
        # Select the ith row and the 8 columns
        shade_sel = [((x * 8) + j) for x in range(12)]
        row_heat = df_kpi.iloc[i, shade_sel]
        row_cool = df_kpi.iloc[(i + 1), shade_sel]
        row_light = df_kpi.iloc[(i + 2), shade_sel]
        row_total = row_heat + row_cool + row_light
        print(row_total)

        min_val = row_total.min()
        max_val = row_total.max()

        # Calculate the ratings based on min_val and max_val
        scaled_values = (max_val - row_total) / (max_val - min_val) * 0.5 + 0.5
        print(scaled_values)

        # Add the rating values as a new row to df_rating
        row_label = row_names[j]
        df_rating.loc[row_label] = scaled_values.values
        df_scores_1.loc[row_label] = row_heat.values
        df_scores_2.loc[row_label] = row_cool.values
        df_scores_3.loc[row_label] = row_light.values
        df_scores_4.loc[row_label] = row_total.values

    df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                       f'02_Working/Export/20230419_113656/00_Rating_11_Energy.xlsx', header=True, index=True)
    df_scores_1.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                       f'02_Working/Export/20230419_113656/00_OG_heat.xlsx', header=True, index=True)
    df_scores_2.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                       f'02_Working/Export/20230419_113656/00_OG_cool.xlsx', header=True, index=True)
    df_scores_3.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                       f'02_Working/Export/20230419_113656/00_OG_light.xlsx', header=True, index=True)
    df_scores_4.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                       f'02_Working/Export/20230419_113656/00_OG_total.xlsx', header=True, index=True)
#
for i in range(3,4):
    for j in range(orientations):
        # Select the ith row and the 8 columns
        shade_sel = [((x * 8) + j) for x in range(12)]
        row_total = df_kpi.iloc[i, shade_sel]
        print(row_total)

        min_val = row_total.min()
        max_val = row_total.max()

        # Calculate the ratings based on min_val and max_val
        scaled_values = (max_val - row_total) / (max_val - min_val) * 0.5 + 0.5
        print(scaled_values)

        # Add the rating values as a new row to df_rating
        row_label = row_names[j]
        df_rating.loc[row_label] = scaled_values.values
        df_rating.fillna(0.5, inplace=True)
        df_scores_1.loc[row_label] = row_total.values

    print(df_rating)

    df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                       f'02_Working/Export/20230419_113656/00_Rating_3_PMV.xlsx', header=True, index=True)
    df_scores_1.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                         f'02_Working/Export/20230419_113656/00_OG_PMV.xlsx', header=True, index=True)

for i in range(11,12):
    for j in range(orientations):
        # Select the ith row and the 8 columns
        shade_sel = [((x * 8) + j) for x in range(12)]
        row_total = df_kpi.iloc[i, shade_sel]
        print(row_total)

        min_val = row_total.min()
        max_val = row_total.max()

        # Calculate the ratings based on min_val and max_val
        scaled_values = (max_val - row_total) / (max_val - min_val) * 0.5 + 0.5
        print(scaled_values)

        # Add the rating values as a new row to df_rating
        row_label = row_names[j]
        df_rating.loc[row_label] = scaled_values.values
        df_rating.fillna(0.5, inplace=True)
        df_scores_1.loc[row_label] = row_total.values

    print(df_rating)

    df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                       f'02_Working/Export/20230419_113656/00_Rating_3_PMV_weighted.xlsx', header=True, index=True)
    df_scores_1.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                         f'02_Working/Export/20230419_113656/00_OG_PMV_weighted.xlsx', header=True, index=True)

df_scores_1 = pd.DataFrame(columns=['RB_05_L', 'RB_05_M', 'RB_05_D',
                                    'RB_10_L', 'RB_10_M', 'RB_10_D',
                                    'VB_25_L', 'VB_25_M', 'VB_25_D',
                                    'VB_50_L', 'VB_50_M', 'VB_50_D' ])

for i in range(4,5):
    for j in range(orientations):
        # Select the ith row and the 8 columns
        shade_sel = [((x * 8) + j) for x in range(12)]
        row_total = df_kpi.iloc[i, shade_sel]
        print(row_total)

        min_val = row_total.min()
        max_val = row_total.max()

        # Calculate the ratings based on min_val and max_val
        scaled_values = (max_val - row_total) / (max_val - min_val) * 0.5 + 0.5
        print(scaled_values)

        # Add the rating values as a new row to df_rating
        row_label = row_names[j]
        df_rating.loc[row_label] = scaled_values.values
        df_rating.fillna(0.5, inplace=True)
        df_scores_1.loc[row_label] = row_total.values

    print(df_rating)

    df_scores_1.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                         f'02_Working/Export/20230419_113656/00_OG_PPD.xlsx', header=True, index=True)


for i in range(5, 8):
    for j in range(orientations):
        # Select the ith row and the 8 columns
        shade_sel = [((x * 8) + j) for x in range(12)]
        row = df_kpi.iloc[i, shade_sel]
        print(row)

        if i == 5:
            k = 0
        elif i == 6:
            k = 1
        else:
            k = 2

        base_udi_val = df_udi_noshade.iloc[j, k]

        print(base_udi_val)

        normalised_udi = row / base_udi_val

        print(normalised_udi)

        # Add the rating values as a new row to df_rating
        row_label = row_names[j]
        df_rating.loc[row_label] = normalised_udi.values
        df_scores_1.loc[row_label] = row.values


    if i == 5:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_UDI_1.xlsx', header=True, index=True)
        df_scores_1.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                             f'02_Working/Export/20230419_113656/00_OG_UDI_1.xlsx', header=True, index=True)
    elif i == 6:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_UDI_3.xlsx', header=True, index=True)
        df_scores_1.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                             f'02_Working/Export/20230419_113656/00_OG_UDI_3.xlsx', header=True, index=True)
    else:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_UDI_5.xlsx', header=True, index=True)
        df_scores_1.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                             f'02_Working/Export/20230419_113656/00_OG_UDI_5.xlsx', header=True, index=True)

for i in range(8, 11):
    for j in range(orientations):
        # Select the ith row and the 8 columns
        shade_sel = [((x * 8) + j) for x in range(12)]
        row = df_kpi.iloc[i, shade_sel]
        row = row + 1
        print(row)

        min_val = row.min()
        max_val = row.max()

        shifted_values = row - min_val
        print(shifted_values)

        realigned_values = min_val - (shifted_values)
        print(realigned_values)

        max_realigned = realigned_values.max()
        normalised_values = 1 - ((max_realigned - realigned_values)/ max_realigned)

        print(normalised_values)

        # Add the rating values as a new row to df_rating
        row_label = row_names[j]
        df_rating.loc[row_label] = normalised_values.values
        df_scores_1.loc[row_label] = row.values

    if i == 8:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_DGP_1.xlsx', header=True, index=True)
        df_scores_1.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                             f'02_Working/Export/20230419_113656/00_OG_DGP_1.xlsx', header=True, index=True)
    elif i == 9:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_DGP_3.xlsx', header=True, index=True)
        df_scores_1.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                             f'02_Working/Export/20230419_113656/00_OG_DGP_3.xlsx', header=True, index=True)
    else:
        df_rating.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                           f'02_Working/Export/20230419_113656/00_Rating_{i}_DGP_5.xlsx', header=True, index=True)
        df_scores_1.to_excel(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                             f'02_Working/Export/20230419_113656/00_OG_DGP_5.xlsx', header=True, index=True)
