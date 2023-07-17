import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

df_energy = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                          "02_Working/Export/20230419_113656/00_Rating_11_Energy.xlsx", index_col=0)

df_thermal = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                           "02_Working/Export/20230419_113656/00_Rating_3_PMV_weighted.xlsx", index_col=0)

df_udi_1 = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                         "02_Working/Export/20230419_113656/00_Rating_5_UDI_1.xlsx", index_col=0)

df_udi_3 = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                         "02_Working/Export/20230419_113656/00_Rating_6_UDI_3.xlsx", index_col=0)

df_udi_5 = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                         "02_Working/Export/20230419_113656/00_Rating_7_UDI_5.xlsx", index_col=0)

df_dgp_1 = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                         "02_Working/Export/20230419_113656/00_Rating_8_DGP_1.xlsx", index_col=0)

df_dgp_3 = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                         "02_Working/Export/20230419_113656/00_Rating_9_DGP_3.xlsx", index_col=0)

df_dgp_5 = pd.read_excel("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                         "02_Working/Export/20230419_113656/00_Rating_10_DGP_5.xlsx", index_col=0)

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

def sim_heatmap(dataframe, label, fs):

    fig, ax = plt.subplots(figsize=(20, 10))  # Set the figure size

    # Round off the values in the dataframe to one decimal point
    dataframe = dataframe.round(2)

    # Plot the correlation matrix as a heatmap
    heatmap = sns.heatmap(data=dataframe, cmap=color_map_2_wtob, annot=True, fmt='.2f', cbar=False,
                linewidths=1, linecolor='white', square=True, annot_kws={"size": fs})

    # Set font size for all elements in the plot
    plt.rcParams.update({'font.size': 20})

    # Add legend to the right
    cbar = heatmap.figure.colorbar(heatmap.collections[0])

    # Set font size of x-axis and y-axis tick labels
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=20, rotation=45, ha='right')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=20, rotation=0)
    heatmap.set_xlabel('Shade', fontsize=20)
    heatmap.set_ylabel('Orientation', fontsize=20)

    plt.tight_layout()
    plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                f"02_Working/Jpeg/Survey/{label}.png", dpi=300, bbox_inches='tight')
    plt.show()


df_kpi = {}

df_kpi_1 = df_energy + df_thermal + df_udi_1 + df_dgp_1
df_kpi_3 = df_energy + df_thermal + df_udi_3 + df_dgp_3
df_kpi_5 = df_energy + df_thermal + df_udi_5 + df_dgp_5
#
# sim_heatmap(df_kpi_1, 'rate_1', fs)
# sim_heatmap(df_kpi_3, 'rate_3', fs)
# sim_heatmap(df_kpi_5, 'rate_5', fs)

plt.rcParams.update({'figure.autolayout': True})
sns.set(rc={'figure.figsize': (16, 12)})
sns.set_style("darkgrid", {"axes.facecolor": color_0_background})

for i in range (8):
    for j in range (3):

        if i == 2 and j == 0:

            if j == 0:
                df_udi_select = df_udi_1
                df_dgp_select = df_dgp_1
            elif j == 1:
                df_udi_select = df_udi_3
                df_dgp_select = df_dgp_3
            else:
                df_udi_select = df_udi_5
                df_dgp_select = df_dgp_5

            energy_select = df_energy.iloc[i]
            thermal_select = df_thermal.iloc[i]
            udi_select = df_udi_select.iloc[i]
            dgp_select = df_dgp_select.iloc[i]

            labels = []
            labelsy = []
# 'Energy', 'Thermal', 'UDI', 'DGP', 'Energy'
            # Create the data DataFrame
            data = {
                'Energy': energy_select,
                'Thermal': thermal_select,
                'UDI': udi_select,
                'DGP': dgp_select
            }
            index = energy_select.index
            df_values = pd.DataFrame(data, index=index)

            num_columns = len(df_values.columns) + 1
            theta = np.linspace(0, 2 * np.pi, num=num_columns)

            # Update the mapping based on shade name in the index
            for shade in df_values.index:
                color = color_map.get(shade, 'black')
                thickness = thickness_map.get(shade, 1.0)
                linetype = linetype_map.get(shade, '-')
                ax = plt.subplot(111, polar=True)

                # Plot the values with the updated mapping
                values = df_values.loc[shade].values
                values = np.append(values, values[0])  # Append the first value to close the polygon
                ax.plot(theta, values, color=color, linewidth=thickness, linestyle=linetype, label=shade)

            ax.set_xticks(theta)
            ax.set_xticklabels(labels, fontsize=20)
            ax.set_yticklabels(labelsy, fontsize=20)
            ax.grid(color='gray')
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                        f"02_Working/Jpeg/Survey/orientation{i + 1}_distance{j + 1}.png", dpi=300, bbox_inches='tight')
            plt.show()

