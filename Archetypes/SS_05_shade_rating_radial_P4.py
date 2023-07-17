import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# Define your custom colors
color_0_background = '#f7f7f7'
custom_colors_RGB = ['#336371', '#8E5F60']
custom_colors_R = ['#E3C8C8', '#6D4848']
color_1_base = '#6D909C'
color_1_base_dark = '#5B7C86'
color_1_base_light = '#C2D1D6'
group_1_blues = sns.color_palette([color_1_base_dark, color_1_base, color_1_base_light])
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
color_map_2_wtob = mcolors.LinearSegmentedColormap.from_list('my_color_map', ['#336371', '#FAF9F6'])
color_map_2_btow = mcolors.LinearSegmentedColormap.from_list('my_color_map', ['#FAF9F6', '#336371'])
fs = 20

# Setting definition for plot name and label length
def update_plot(plot, fontsize):
    xtick_labels = plot.get_xticklabels()
    for label in xtick_labels:
        label.set_fontsize(fontsize)
    plot.set_xticklabels(xtick_labels)
    ytick_labels = plot.get_yticklabels()
    for label in ytick_labels:
        label.set_fontsize(fontsize)
    plot.set_yticklabels(ytick_labels)

df = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_clustered_ss_directions.csv')

pmv = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                     "02_Working/Export/20230419_113656/00_Rating_3_PMV.xlsx", index_col=0)

dgp_1 = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                       "02_Working/Export/20230419_113656/00_Rating_8_DGP_1.xlsx", index_col=0)

dgp_3 = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                       "02_Working/Export/20230419_113656/00_Rating_9_DGP_3.xlsx", index_col=0)

dgp_5 = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                       "02_Working/Export/20230419_113656/00_Rating_10_DGP_5.xlsx", index_col=0)

udi_1 = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                       "02_Working/Export/20230419_113656/00_Rating_5_UDI_1.xlsx", index_col=0)

udi_3 = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                       "02_Working/Export/20230419_113656/00_Rating_6_UDI_3.xlsx", index_col=0)

udi_5 = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                       "02_Working/Export/20230419_113656/00_Rating_7_UDI_5.xlsx", index_col=0)

energy = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                       "02_Working/Export/20230419_113656/00_Rating_11_Energy.xlsx", index_col=0)

# cluster_weight = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                               "02_Working/Excel/02_Qualtrics Data/Archetype_ss.csv")

cluster_weight = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                              "02_Working/Excel/02_Qualtrics Data/median_table_for_rating - Copy.csv", index_col=0)

# Count the occurrences of each category
category_counts = df['archetype'].value_counts()
print(category_counts)

# Create a list of column names for the 12 new empty columns
new_column_names = ['RB_05_L', 'RB_05_M', 'RB_05_D', 'RB_10_L', 'RB_10_M', 'RB_10_D',
                    'VB_25_L', 'VB_25_M', 'VB_25_D', 'VB_50_L', 'VB_50_M', 'VB_50_D',
                    'RB_05_L_Adj', 'RB_05_M_Adj', 'RB_05_D_Adj', 'RB_10_L_Adj', 'RB_10_M_Adj', 'RB_10_D_Adj',
                    'VB_25_L_Adj', 'VB_25_M_Adj', 'VB_25_D_Adj', 'VB_50_L_Adj', 'VB_50_M_Adj', 'VB_50_D_Adj']

# Create a new DataFrame with 12 empty columns and the specified column names Concatenate the two DataFrames (axis=1)
empty_columns = pd.DataFrame(columns=new_column_names)
df = pd.concat([df, empty_columns], axis=1)

count = 0
count_ml = 0

color_map = {'RB_05_L': color_1_base, 'RB_05_M': color_1_base, 'RB_05_D': color_1_base,
             'RB_10_L': color_2_base, 'RB_10_M': color_2_base, 'RB_10_D': color_2_base,
             'VB_25_L': color_3_base, 'VB_25_M': color_3_base, 'VB_25_D': color_3_base,
             'VB_50_L': color_4_base, 'VB_50_M': color_4_base, 'VB_50_D': color_4_base}

thickness_map = {'RB_05_L': 1.5, 'RB_05_M': 2.25, 'RB_05_D': 3.0,
                 'RB_10_L': 1.5, 'RB_10_M': 2.25, 'RB_10_D': 3.0,
                 'VB_25_L': 1.5, 'VB_25_M': 2.25, 'VB_25_D': 3.0,
                 'VB_50_L': 1.5, 'VB_50_M': 2.25, 'VB_50_D': 3.0}

linetype_map = {'RB_05_L': '-', 'RB_05_M': '--', 'RB_05_D': '-.',
                'RB_10_L': '-', 'RB_10_M': '--', 'RB_10_D': '-.',
                'VB_25_L': '-', 'VB_25_M': '--', 'VB_25_D': '-.',
                'VB_50_L': '-', 'VB_50_M': '--', 'VB_50_D': '-.'}

output_df = pd.DataFrame(columns=['column1_obj', 'column2_obj', 'column3_obj',
                                  'column1_ind', 'column2_ind', 'column3_ind',
                                  'column1_clu', 'column2_clu', 'column3_clu'])

success_rate_df = pd.DataFrame(columns=['obj_success_1', 'clu_success_1',
                                        'obj_success_2', 'clu_success_2',
                                        'obj_success_3', 'clu_success_3',
                                        'archetype'])
plt.rcParams.update({'figure.autolayout': True})
sns.set(rc={'figure.figsize': (16, 16)})
sns.set_style("darkgrid", {"axes.facecolor": color_0_background})

count_obj_1 = 0
count_obj_2 = 0
count_obj_3 = 0
count_clu_1 = 0
count_clu_2 = 0
count_clu_3 = 0

# Window distance can be 1, 3 or 5
window_dist = 1

# Orientation can be 0 to 7
orientation = 6

if window_dist == 1:
    df_udi_select = udi_1
    df_dgp_select = dgp_1
elif window_dist == 3:
    df_udi_select = udi_3
    df_dgp_select = dgp_3
else:
    df_udi_select = udi_5
    df_dgp_select = dgp_5

# Running an initial ranking based on objective parameters
energy_select = energy.iloc[orientation]
thermal_select = pmv.iloc[orientation]
udi_select = df_udi_select.iloc[orientation]
dgp_select = df_dgp_select.iloc[orientation]

labels = ['Energy', 'Thermal', 'UDI', 'DGP', 'Energy']
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
ax.set_xticklabels(['', '', '', '', ''], fontsize=20)
ax.grid(color='gray')
# Add legend
# plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
            f"02_Working/Jpeg/Survey/radial_orientation{orientation}_distance{window_dist}.png", dpi=300, bbox_inches='tight')
plt.show()

# Generate the variable names for the dataframes to be added using string formatting
dgp_var = "dgp_{}".format(window_dist)
udi_var = "udi_{}".format(window_dist)

for index, row in cluster_weight.iterrows():

    occ_clu = index
    archetype = occ_clu + 1
    print("this is archetype:", archetype)

    rowz = ['South', 'South West', 'West', 'North West', 'North', 'North East', 'East', 'South East']

    vqi_clu = pd.DataFrame(index=rowz, columns=['RB_05_L', 'RB_05_M', 'RB_05_D', 'RB_10_L', 'RB_10_M', 'RB_10_D',
                                                'VB_25_L', 'VB_25_M', 'VB_25_D', 'VB_50_L', 'VB_50_M', 'VB_50_D'])

    # Assign scores based on OF and slat size
    RB_05 = (cluster_weight.loc[occ_clu, 'rb_05']) + 0.001
    RB_10 = (cluster_weight.loc[occ_clu, 'rb_10']) + 0.001
    VB_25 = (cluster_weight.loc[occ_clu, 'vb_25']) + 0.001
    VB_50 = (cluster_weight.loc[occ_clu, 'vb_50']) + 0.001

    RB_L = cluster_weight.loc[occ_clu, 'sr_view1_rb_05_L'] + 0.001
    RB_M = cluster_weight.loc[occ_clu, 'sr_view1_rb_05_M'] + 0.001
    RB_D = cluster_weight.loc[occ_clu, 'sr_view1_rb_05_D'] + 0.001

    VB_L = cluster_weight.loc[occ_clu, 'sr_view1_vb_25_L'] + 0.001
    VB_D = cluster_weight.loc[occ_clu, 'sr_view1_vb_25_D'] + 0.001

    RB_05_L_rate = ((RB_05 + RB_L)) / 2
    RB_05_M_rate = ((RB_05 + RB_M)) / 2
    RB_05_D_rate = ((RB_05 + RB_D)) / 2
    RB_10_L_rate = ((RB_10 + RB_L)) / 2
    RB_10_M_rate = ((RB_10 + RB_M)) / 2
    RB_10_D_rate = ((RB_10 + RB_D)) / 2
    VB_25_L_rate = ((VB_25 + VB_L)) / 2
    VB_25_M_rate = (VB_25)
    VB_25_D_rate = ((VB_25 + VB_D)) / 2
    VB_50_L_rate = ((VB_50 + VB_L)) / 2
    VB_50_M_rate = (VB_50)
    VB_50_D_rate = ((VB_50 + VB_D)) / 2

    # Define additional scoring for each column
    scores = {'RB_05_L': RB_05_L_rate, 'RB_05_M': RB_05_M_rate, 'RB_05_D': RB_05_D_rate,
              'RB_10_L': RB_10_L_rate, 'RB_10_M': RB_10_M_rate, 'RB_10_D': RB_10_D_rate,
              'VB_25_L': VB_25_L_rate, 'VB_25_M': VB_25_M_rate, 'VB_25_D': VB_25_D_rate,
              'VB_50_L': VB_50_L_rate, 'VB_50_M': VB_50_M_rate, 'VB_50_D': VB_50_D_rate}

    # Repeat each value 8 times
    values = np.tile(list(scores.values()), 8)

    # Assign the values to the DataFrame
    vqi_clu.iloc[:, :] = values.reshape((8, 12))

    # Rating of interior properties
    int_clu = pd.DataFrame(index=rowz, columns=['RB_05_L', 'RB_05_M', 'RB_05_D', 'RB_10_L', 'RB_10_M', 'RB_10_D',
                                                'VB_25_L', 'VB_25_M', 'VB_25_D', 'VB_50_L', 'VB_50_M', 'VB_50_D'])

    RB_05_L_interiors = ((cluster_weight.loc[occ_clu, 'sr_view4_rb_05'])) + 0.001
    RB_05_M_interiors = ((cluster_weight.loc[occ_clu, 'sr_view4_rb_05'])) + 0.001
    RB_05_D_interiors = ((cluster_weight.loc[occ_clu, 'sr_view4_rb_05'])) + 0.001
    RB_10_L_interiors = ((cluster_weight.loc[occ_clu, 'sr_view4_rb_10'])) + 0.001
    RB_10_M_interiors = ((cluster_weight.loc[occ_clu, 'sr_view4_rb_10'])) + 0.001
    RB_10_D_interiors = ((cluster_weight.loc[occ_clu, 'sr_view4_rb_10'])) + 0.001
    VB_25_L_interiors = ((cluster_weight.loc[occ_clu, 'sr_view4_vb_25'])) + 0.001
    VB_25_M_interiors = ((cluster_weight.loc[occ_clu, 'sr_view4_vb_25'])) + 0.001
    VB_25_D_interiors = ((cluster_weight.loc[occ_clu, 'sr_view4_vb_25'])) + 0.001
    VB_50_L_interiors = ((cluster_weight.loc[occ_clu, 'sr_view4_vb_50'])) + 0.001
    VB_50_M_interiors = ((cluster_weight.loc[occ_clu, 'sr_view4_vb_50'])) + 0.001
    VB_50_D_interiors = ((cluster_weight.loc[occ_clu, 'sr_view4_vb_50'])) + 0.001

    interior_scores = {'RB_05_L': RB_05_L_interiors, 'RB_05_M': RB_05_M_interiors, 'RB_05_D': RB_05_D_interiors,
                       'RB_10_L': RB_10_L_interiors, 'RB_10_M': RB_10_M_interiors, 'RB_10_D': RB_10_D_interiors,
                       'VB_25_L': VB_25_L_interiors, 'VB_25_M': VB_25_M_interiors, 'VB_25_D': VB_25_D_interiors,
                       'VB_50_L': VB_50_L_interiors, 'VB_50_M': VB_50_M_interiors, 'VB_50_D': VB_50_D_interiors}

    # Repeat each value 8 times
    interior_values = np.tile(list(interior_scores.values()), 8)

    # Assign the values to the DataFrame
    int_clu.iloc[:, :] = interior_values.reshape((8, 12))
    print(int_clu)

    vqi_select = vqi_clu.iloc[orientation]
    int_select = int_clu.iloc[orientation]

    # Start plotting radial grid including visual preferences
    labels = ['Energy', 'Thermal', 'UDI', 'DGP', 'VQI', 'Interior', 'Energy']
    data = {
        'Energy': energy_select,
        'Thermal': thermal_select,
        'UDI': udi_select,
        'DGP': dgp_select,
        'VQI': vqi_select,
        'Interior': int_select
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
    ax.set_xticklabels(['', '', '', '', '', '', ''], fontsize=20)
    ax.grid(color='gray')
    # Add legend
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                f"02_Working/Jpeg/Survey/radial_pref_orientation{orientation}_distance{window_dist}_archetyp{archetype}.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    factor_thermal_cluster = (cluster_weight.loc[occ_clu, 'weight_temperature'])
    factor_energy_cluster = (cluster_weight.loc[occ_clu, 'weight_energy'])
    factor_udi_cluster = (cluster_weight.loc[occ_clu, 'weight_daylight'])
    factor_dgp_cluster = (cluster_weight.loc[occ_clu, 'weight_glare'])
    factor_vqi_cluster = (cluster_weight.loc[occ_clu, 'weight_view'])
    factor_int_cluster = (cluster_weight.loc[occ_clu, 'weight_interiors'])

    data = {
        'Energy': (energy_select * factor_energy_cluster),
        'Thermal': (thermal_select * factor_thermal_cluster),
        'UDI': (udi_select * factor_udi_cluster),
        'DGP': (dgp_select * factor_dgp_cluster),
        'VQI': (vqi_select * factor_vqi_cluster),
        'Interior': (int_select * factor_int_cluster)
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
    ax.set_xticklabels(['', '', '', '', '', '', ''], fontsize=20)
    ax.grid(color='gray')
    # Add legend
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                f"02_Working/Jpeg/Survey/radial_phinal_orientation{orientation}_distance{window_dist}_archetyp{archetype}.png",
                dpi=300, bbox_inches='tight')
    plt.show()