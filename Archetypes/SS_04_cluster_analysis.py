import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.colors as mcolors

# Load CSV as dataframe
df = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_clustered_ss.csv')

ar = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/Archetype_ss.csv')

median_stf = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                         '02_Working/Excel/02_Qualtrics Data/median_std.csv',  index_col=0)

# print(df)
#
# # Count the occurrences of each category
# category_counts = df['archetype'].value_counts()
# print(category_counts)
#
# # Plot the barplot
# plt.figure(figsize=(8, 6))
# category_counts.plot(kind='bar')
# plt.title("Distribution of 'archetype' Categories")
# plt.xlabel("Categories")
# plt.ylabel("Count")
# plt.xticks(rotation=45)
# plt.show()

# archetype_comp = pd.read_excel('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
#                                '02_Working/Excel/02_Qualtrics Data/comparison.xlsx',
#                                skiprows=1, index_col=0)

# archetype_comp = archetype_comp.astype(float)
# archetype_comp = archetype_comp.round(2)
#
# latex_table = archetype_comp.to_latex(index=True, escape=False)
# print(latex_table)

# fs = 12
# pad = 12
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
color_1_background = '#adadad'
color_map_2_wtob = mcolors.LinearSegmentedColormap.from_list('my_color_map', ['#336371', '#FAF9F6'])
color_map_2_btow = mcolors.LinearSegmentedColormap.from_list('my_color_map', ['#FAF9F6', '#336371'])



df['rb_05'] = df[['sr_view1_rb_05', 'sr_view2_rb_05', 'sr_view3_rb_05']].mean(axis=1)
df['rb_10'] = df[['sr_view1_rb_10', 'sr_view2_rb_10', 'sr_view3_rb_10']].mean(axis=1)
df['vb_25'] = df[['sr_view1_vb_25', 'sr_view2_vb_25', 'sr_view3_vb_25']].mean(axis=1)
df['vb_50'] = df[['sr_view1_vb_50', 'sr_view2_vb_50', 'sr_view3_vb_50']].mean(axis=1)
df['weight_temperature'] = df[['ef_importance_temperature', 'ef_productivity_temperature']].mean(axis=1)
df['weight_energy'] = df[['psy_e_consciously_sustainable',
                                    'psy_e_change_sustainable',
                                    'psy_e_compromise_comfort_sustainabile']].mean(axis=1)
df['weight_daylight'] = df[['ef_importance_daylight', 'ef_productivity_daylight']].mean(axis=1)
df['weight_glare'] = df[['ef_importance_glare', 'ef_productivity_glare']].mean(axis=1)
df['weight_view'] = df[['ef_importance_view', 'ef_productivity_view']].mean(axis=1)
df['weight_interiors'] = df[['psy_c_importance_interior_features']]

# List of columns to plot
beliefs = ['weight_temperature', 'weight_energy', 'weight_daylight',
           'weight_view', 'weight_interiors', 'weight_glare']

preferences = ['sr_view4_rb_05', 'sr_view4_rb_10', 'sr_view4_vb_25', 'sr_view4_vb_50',
               'rb_05', 'rb_10', 'vb_25', 'vb_50', 'sr_view1_rb_05_L',
               'sr_view1_rb_05_M', 'sr_view1_rb_05_D', 'sr_view1_vb_25_L', 'sr_view1_vb_25_D']

variables = ['weight_temperature', 'weight_energy', 'weight_daylight',
             'weight_view', 'weight_interiors', 'weight_glare',
             'sr_view4_rb_05', 'sr_view4_rb_10', 'sr_view4_vb_25', 'sr_view4_vb_50',
             'rb_05', 'rb_10', 'vb_25', 'vb_50', 'sr_view1_rb_05_L',
             'sr_view1_rb_05_M', 'sr_view1_rb_05_D', 'sr_view1_vb_25_L', 'sr_view1_vb_25_D']

colors = ['#8fa9b2', '#AB8788', '#89b8a6', '#d6b680']
# Define the mapping of values
archetype_mapping = {0: 1, 1: 2, 2: 3, 3: 4}

# Create a new column with modified archetype values
df['archetype_modified'] = df['archetype'].map(archetype_mapping)

# Update the x-tick labels accordingly
xtick_labels = ['1', '2', '3', '4']

plt.rcParams.update({'figure.autolayout': True})
sns.set(rc={'figure.figsize': (10, 10)})
sns.set_style("whitegrid", {"axes.facecolor": color_0_background, 'grid.color': color_1_background})

for variable in variables:
    plt.figure(figsize=(10, 16))
    ax = sns.boxplot(x='archetype_modified', y=variable, data=df, palette=colors, width=0.6, linewidth=2.5)

    # Add a line at the median
    medians = df.groupby('archetype_modified')[variable].median()
    for xtick, median in enumerate(medians):
        x = xtick
        y = median
        ax.plot([x, x], [y, y], color='black', linewidth=2)

    plt.xlabel('Archetype', fontsize=20)
    plt.ylabel('Values', fontsize=20)
    plt.ylim(-0.3, 1.3)
    plt.xlim(-0.5, 3.5)
    plt.xticks(np.arange(4), xtick_labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('')
    plt.tight_layout()
    plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                f"02_Working/Jpeg/Survey/Archetype_item_{variable}.png", dpi=300, bbox_inches='tight')
    plt.show()


# # Create separate distribution tables for median and standard deviation
# median_table = pd.DataFrame()
# std_table = pd.DataFrame()
# mean_table = pd.DataFrame()
#
# # Iterate over each variable
# for variable in variables:
#     # Group the data by cluster label and calculate the distribution and median for the current variable
#     variable_distribution = df.groupby('archetype')[variable].agg(['mean', 'std', 'median'])
#
#     # Add the median to the median table with a separate column for each cluster
#     median_table[f'median_{variable}'] = variable_distribution['median']
#
#     # Add the standard deviation to the standard deviation table with a separate column for each cluster
#     std_table[f'std_{variable}'] = variable_distribution['std']
#
#     # Add the standard deviation to the standard deviation table with a separate column for each cluster
#     mean_table[f'std_{variable}'] = variable_distribution['mean']
#
# # Set the index of the median and standard deviation tables to the cluster labels
# median_table.index.name = 'Cluster'
# std_table.index.name = 'Cluster'
# mean_table.index.name = 'Cluster'
#
# # Transpose the median and standard deviation tables
# transposed_median_table = median_table.transpose()
# transposed_std_table = std_table.transpose()
# transposed_mean_table = mean_table.transpose()
#
# # Print the transposed median table
# print("Median Table:")
# print(transposed_median_table)
# median_table.to_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
#                     '02_Working/Excel/02_Qualtrics Data/median_table_for_rating.csv')
#
# # Print the transposed standard deviation table
# print("\nStandard Deviation Table:")
# print(transposed_std_table)





#
# combined_variables = pd.DataFrame()
#
# for i in range(4):
#
#     median_column_name = f"{i}_median"
#     std_column_name = f"{i}_std"
#
#     combined_variables[median_column_name] = transposed_median_table[i]
#
#     combined_variables[std_column_name] = transposed_std_table[i]
#
#     combined_variables[str(i)] = i
#
# print(combined_variables)
#
# combined_variables.to_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
#                         '02_Working/Excel/02_Qualtrics Data/archetype_distribution.csv')
#
# transposed_median_table.to_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
#                         '02_Working/Excel/02_Qualtrics Data/median.csv')
#
# transposed_std_table.to_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
#                         '02_Working/Excel/02_Qualtrics Data/std.csv')
#
# transposed_mean_table.to_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
#                         '02_Working/Excel/02_Qualtrics Data/mean.csv')


# # Save the transposed tables to CSV files
# transposed_median_table.to_csv('median_distribution.csv')
# transposed_std_table.to_csv('std_distribution.csv')
#
# # Save the dataframe used for clustering (non-scaled)
# transposed_table.to_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
#                         '02_Working/Excel/02_Qualtrics Data/archetype_distribution.csv')

# median_stf.rename(columns={"median_1": "Archetype 1"}, inplace=True)
# median_stf.rename(columns={"median_2": "Archetype 2"}, inplace=True)
# median_stf.rename(columns={"median_3": "Archetype 3"}, inplace=True)
# median_stf.rename(columns={"median_4": "Archetype 4"}, inplace=True)
#
# color_map = {'Archetype 1': color_1_base, 'Archetype 2': color_2_base, 'Archetype 3': color_3_base,
#              'Archetype 4': color_4_base}
#
# thickness_map = {'Archetype 1': 3.0, 'Archetype 2': 3.0, 'Archetype 3': 3.0, 'Archetype 4': 3.0}
#
# linetype_map = {'Archetype 1': '-', 'Archetype 2': '-', 'Archetype 3': '-', 'Archetype 4': '-'}
#
# plt.rcParams.update({'figure.autolayout': True})
# sns.set(rc={'figure.figsize': (16, 12)})
# sns.set_style("darkgrid", {"axes.facecolor": color_0_background})
#
# fig = plt.figure()
# ax = fig.add_subplot(111, polar=True)

# for i in range(1, 5):
#     col_select = median_stf.filter(regex=f"Archetype {i}$").iloc[:6]
#     print(col_select)
#     num_elements = col_select.shape[0]  # Use the number of rows
#     theta = np.linspace(0, 2 * np.pi, num=num_elements, endpoint=False)
#
#     column = col_select.columns[0]
#     color = color_map.get(column, 'black')
#     thickness = thickness_map.get(column, 1.0)
#     linetype = linetype_map.get(column, '-')
#
#     # Plot the values in polar coordinates
#     values = col_select.values.flatten()
#     values = np.append(values, values[0])  # Append the first value to create a closed loop
#     theta = np.append(theta, theta[0])  # Append the first angle to create a closed loop
#     theta_ticks = np.linspace(0, 2 * np.pi, num=num_elements, endpoint=False)
#     # theta_labels = ['Thermal', 'Energy', 'Daylight', 'View', 'Interiors', 'Glare']
#     theta_labels = ['', '', '', '', '', '']
#     ax.plot(theta, values, label=column, color=color, linewidth=thickness,
#             linestyle=linetype, marker='o', markersize=10)
#
# # Customize the plot properties
# ax.legend()
# ax.set_xticks(theta_ticks)  # Set the x-ticks to align with theta values
# ax.set_xticklabels(theta_labels, fontsize=20)  # Set the x-tick labels
# ax.grid(color='gray', linestyle='--')  # Set grid lines with gray color and dashed linestyle
# ax.tick_params(axis='x', pad=40)
# # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=15)
# ax.legend().remove()
# plt.tight_layout()
# plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#             f"02_Working/Jpeg/Survey/Archetype_belief_NL.png", dpi=300, bbox_inches='tight')
# # Display the plot
# plt.show()

# for i in range(1, 5):
#     col_select = median_stf.filter(regex=f"Archetype {i}$").iloc[6:10]
#     print(col_select)
#     num_elements = col_select.shape[0]  # Use the number of rows
#     theta = np.linspace(0, 2 * np.pi, num=num_elements, endpoint=False)
#
#     column = col_select.columns[0]
#     color = color_map.get(column, 'black')
#     thickness = thickness_map.get(column, 1.0)
#     linetype = linetype_map.get(column, '-')
#
#     # Plot the values in polar coordinates
#     values = col_select.values.flatten()
#     values = np.append(values, values[0])  # Append the first value to create a closed loop
#     theta = np.append(theta, theta[0])  # Append the first angle to create a closed loop
#     theta_ticks = np.linspace(0, 2 * np.pi, num=num_elements, endpoint=False)
#     theta_labels = ['RB_05', 'RB_10', 'VB_25', 'VB_50']
#     theta_labels = ['', '', '', '']
#     ax.plot(theta, values, label=column, color=color, linewidth=thickness,
#             linestyle=linetype, marker='o', markersize=10)
#
# # Customize the plot properties
# ax.legend()
# ax.set_xticks(theta_ticks)  # Set the x-ticks to align with theta values
# ax.set_xticklabels(theta_labels, fontsize=20)  # Set the x-tick labels
# ax.grid(color='gray', linestyle='--')  # Set grid lines with gray color and dashed linestyle
# ax.tick_params(axis='x', pad=25)
# plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=15)
# ax.legend().remove()
# plt.tight_layout()
# plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#             f"02_Working/Jpeg/Survey/Archetype_interior_NL.png", dpi=300, bbox_inches='tight')
# # Display the plot
# plt.show()

# for i in range(1, 5):
#     col_select = median_stf.filter(regex=f"Archetype {i}$").iloc[10:14]
#     print(col_select)
#     num_elements = col_select.shape[0]  # Use the number of rows
#     theta = np.linspace(0, 2 * np.pi, num=num_elements, endpoint=False)
#
#     column = col_select.columns[0]
#     color = color_map.get(column, 'black')
#     thickness = thickness_map.get(column, 1.0)
#     linetype = linetype_map.get(column, '-')
#
#     # Plot the values in polar coordinates
#     values = col_select.values.flatten()
#     values = np.append(values, values[0])  # Append the first value to create a closed loop
#     theta = np.append(theta, theta[0])  # Append the first angle to create a closed loop
#     theta_ticks = np.linspace(0, 2 * np.pi, num=num_elements, endpoint=False)
#     theta_labels = ['RB_05', 'RB_10', 'VB_25', 'VB_50']
#     theta_labels = ['', '', '', '']
#     ax.plot(theta, values, label=column, color=color, linewidth=thickness,
#             linestyle=linetype, marker='o', markersize=10)
#
# # Customize the plot properties
# ax.legend()
# ax.set_xticks(theta_ticks)  # Set the x-ticks to align with theta values
# ax.set_xticklabels(theta_labels, fontsize=20)  # Set the x-tick labels
# ax.grid(color='gray', linestyle='--')  # Set grid lines with gray color and dashed linestyle
# ax.tick_params(axis='x', pad=25)
# # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=15)
# ax.legend().remove()
# plt.tight_layout()
# plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#             f"02_Working/Jpeg/Survey/Archetype_clarity_NL.png", dpi=300, bbox_inches='tight')
# # Display the plot
# plt.show()

# for i in range(1, 5):
#     col_select = median_stf.filter(regex=f"Archetype {i}$").iloc[14:19]
#     print(col_select)
#     num_elements = col_select.shape[0]  # Use the number of rows
#     theta = np.linspace(0, 2 * np.pi, num=num_elements, endpoint=False)
#
#     column = col_select.columns[0]
#     color = color_map.get(column, 'black')
#     thickness = thickness_map.get(column, 1.0)
#     linetype = linetype_map.get(column, '-')
#
#     # Plot the values in polar coordinates
#     values = col_select.values.flatten()
#     values = np.append(values, values[0])  # Append the first value to create a closed loop
#     theta = np.append(theta, theta[0])  # Append the first angle to create a closed loop
#     theta_ticks = np.linspace(0, 2 * np.pi, num=num_elements, endpoint=False)
#     theta_labels = ['RB_Light', 'RB_Mid', 'RB_Dark', 'VB_Light', 'VB_Dark']
#     theta_labels = ['', '', '', '', '']
#     ax.plot(theta, values, label=column, color=color, linewidth=thickness,
#             linestyle=linetype, marker='o', markersize=10)
#
# # Customize the plot properties
# ax.legend()
# ax.set_xticks(theta_ticks)  # Set the x-ticks to align with theta values
# ax.set_xticklabels(theta_labels, fontsize=20)  # Set the x-tick labels
# ax.grid(color='gray', linestyle='--')  # Set grid lines with gray color and dashed linestyle
# ax.tick_params(axis='x', pad=25)
# # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=15)
# ax.legend().remove()
# plt.tight_layout()
# plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#             f"02_Working/Jpeg/Survey/Archetype_color_NL.png", dpi=300, bbox_inches='tight')
# # Display the plot
# plt.show()