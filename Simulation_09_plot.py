import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

df_kpi = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                        "02_Working/Export/20230419_113656/00_random.xlsx")

# Add the first three rows and create a new row at the end
sum_row = df_kpi.iloc[0:3].sum()
df_kpi = df_kpi.append(sum_row, ignore_index=True)

print(df_kpi)

# Define your custom colors
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

kpi_nos = df_kpi.shape[0]
sns.set_style("darkgrid", {"axes.facecolor": color_0_background})

color_map_gradient = {'RB_05_L': color_1_base_L2, 'RB_05_M': color_1_base_L1, 'RB_05_D': color_1_base,
                      'RB_10_L': color_2_base_L2, 'RB_10_M': color_2_base_L1, 'RB_10_D': color_2_base,
                      'VB_25_L': color_3_base_L2, 'VB_25_M': color_3_base_L1, 'VB_25_D': color_3_base,
                      'VB_50_L': color_4_base_L2, 'VB_50_M': color_4_base_L1, 'VB_50_D': color_4_base}

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

def parallel(pivoted_df, plot_label, plotname):

    plt.figure(figsize=(20, 8))
    for shade in pivoted_df['shade']:
        plt.plot(pivoted_df.columns[1:],
                 pivoted_df.loc[pivoted_df['shade'] == shade].values[0][1:],
                 color=color_map[shade], linewidth=thickness_map[shade], linestyle=linetype_map[shade], label=shade)
    plt.xlabel('Orientation', fontsize=20)
    plt.ylabel(plot_label, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(bbox_to_anchor=(1.14, 1.01), loc='upper right', fontsize=15)

    # Adjust the plot to fit within the viewframe
    plt.tight_layout()
    plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                f"02_Working/Jpeg/Survey/parallel_{plotname}.png", dpi=300)
    plt.show()


def sim_heatmap(dataframe, label, fs):

    fig, ax = plt.subplots(figsize=(20, 10))  # Set the figure size

    # Round off the values in the dataframe to one decimal point
    dataframe = dataframe.round(1)

    # Plot the correlation matrix as a heatmap
    heatmap = sns.heatmap(data=dataframe, cmap=color_map_2_btow, annot=True, fmt='.1f', cbar=False,
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

# Making an empty df to load the ranking of various KPI's as shade versus orientation
df_rating = pd.DataFrame(columns=['RB_05_L', 'RB_05_M', 'RB_05_D',
                                  'RB_10_L', 'RB_10_M', 'RB_10_D',
                                  'VB_25_L', 'VB_25_M', 'VB_25_D',
                                  'VB_50_L', 'VB_50_M', 'VB_50_D' ])

df_energy = pd.DataFrame(columns=["orientation", "shade", "energy"])
df_thermal = pd.DataFrame(columns=["orientation", "shade", "thermal"])
df_ppd = pd.DataFrame(columns=["orientation", "shade", "thermal"])
df_udi_1 = pd.DataFrame(columns=["orientation", "shade", "udi"])
df_udi_3 = pd.DataFrame(columns=["orientation", "shade", "udi"])
df_udi_5 = pd.DataFrame(columns=["orientation", "shade", "udi"])
df_dgp_1 = pd.DataFrame(columns=["orientation", "shade", "dgp"])
df_dgp_3 = pd.DataFrame(columns=["orientation", "shade", "udi"])
df_dgp_5 = pd.DataFrame(columns=["orientation", "shade", "udi"])
df_heating = pd.DataFrame(columns=["orientation", "shade", "energy"])
df_cooling = pd.DataFrame(columns=["orientation", "shade", "energy"])
df_lighting = pd.DataFrame(columns=["orientation", "shade", "energy"])
df_thermal_weighted = pd.DataFrame(columns=["orientation", "shade", "thermal_weighted"])

orientations = ["South", "South West", "West", "North West", "North", "North East", "East", "South East"]
shades = ["RB_05_L", "RB_05_M", "RB_05_D", "RB_10_L", "RB_10_M", "RB_10_D",
          "VB_25_L", "VB_25_M", "VB_25_D", "VB_50_L", "VB_50_M", "VB_50_D"]

orientation_list = np.concatenate([np.repeat(orientations[i], 12) for i in range(len(orientations))])
shade_list = []
for i in range(8):
    shade_list += shades

df_energy["orientation"] = orientation_list
df_energy["shade"] = shade_list
df_thermal["orientation"] = orientation_list
df_thermal["shade"] = shade_list
df_ppd["orientation"] = orientation_list
df_ppd["shade"] = shade_list
df_udi_1["orientation"] = orientation_list
df_udi_1["shade"] = shade_list
df_udi_3["orientation"] = orientation_list
df_udi_3["shade"] = shade_list
df_udi_5["orientation"] = orientation_list
df_udi_5["shade"] = shade_list
df_dgp_1["orientation"] = orientation_list
df_dgp_1["shade"] = shade_list
df_dgp_3["orientation"] = orientation_list
df_dgp_3["shade"] = shade_list
df_dgp_5["orientation"] = orientation_list
df_dgp_5["shade"] = shade_list
df_heating["orientation"] = orientation_list
df_heating["shade"] = shade_list
df_cooling["orientation"] = orientation_list
df_cooling["shade"] = shade_list
df_lighting["orientation"] = orientation_list
df_lighting["shade"] = shade_list
df_thermal_weighted["orientation"] = orientation_list
df_thermal_weighted["shade"] = shade_list

row_order = ['South', 'South West', 'West', 'North West', 'North', 'North East', 'East', 'South East']
column_order = ['RB_05_L', 'RB_05_M', 'RB_05_D', 'RB_10_L', 'RB_10_M', 'RB_10_D',
                'VB_25_L', 'VB_25_M', 'VB_25_D', 'VB_50_L', 'VB_50_M', 'VB_50_D']

for i in range(0, 1):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_heating.loc[j*12:(j+1)*12-1, "heating"] = output.to_list()

for i in range(1, 2):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_cooling.loc[j*12:(j+1)*12-1, "cooling"] = output.to_list()

for i in range(2, 3):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_lighting.loc[j*12:(j+1)*12-1, "lighting"] = output.to_list()

for i in range(11, 12):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_energy.loc[j*12:(j+1)*12-1, "energy"] = output.to_list()

for i in range(3, 4):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_thermal.loc[j*12:(j+1)*12-1, "thermal"] = output.to_list()

for i in range(4, 5):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_ppd.loc[j*12:(j+1)*12-1, "ppd"] = output.to_list()

for i in range(5, 6):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_udi_1.loc[j*12:(j+1)*12-1, "udi_1"] = output.to_list()

for i in range(6, 7):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_udi_3.loc[j*12:(j+1)*12-1, "udi_3"] = output.to_list()

for i in range(7, 8):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_udi_5.loc[j*12:(j+1)*12-1, "udi_5"] = output.to_list()

for i in range(8, 9):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_dgp_1.loc[j*12:(j+1)*12-1, "dgp_1"] = output.to_list()

for i in range(9, 10):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_dgp_3.loc[j*12:(j+1)*12-1, "dgp_3"] = output.to_list()

for i in range(10, 11):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_dgp_5.loc[j*12:(j+1)*12-1, "dgp_5"] = output.to_list()

for i in range(11, 12):
    for j in range(len(orientations)):
        shade_sel = [((x * 8)+j) for x in range(12)]
        output = df_kpi.iloc[i, shade_sel]
        df_thermal_weighted.loc[j*12:(j+1)*12-1, "thermal_weighted"] = output.to_list()

print(df_thermal_weighted)

# #heatmap plotting of results
# orientation_cat = pd.Categorical(df_heating['orientation'], categories=row_order, ordered=True)
# df_heating['orientation'] = orientation_cat
# df_heat_heatmap = df_heating.pivot(index='orientation', columns='shade', values='heating')
# df_heat_heatmap = df_heat_heatmap[column_order].astype(float)
# sim_heatmap(df_heat_heatmap, "Heating energy", 20)
#
# orientation_cat = pd.Categorical(df_cooling['orientation'], categories=row_order, ordered=True)
# df_cooling['orientation'] = orientation_cat
# df_cool_heatmap = df_cooling.pivot(index='orientation', columns='shade', values='cooling')
# df_cool_heatmap = df_cool_heatmap[column_order].astype(float)
# sim_heatmap(df_cool_heatmap, "Cooling energy", 20)
#
# orientation_cat = pd.Categorical(df_lighting['orientation'], categories=row_order, ordered=True)
# df_lighting['orientation'] = orientation_cat
# df_light_heatmap = df_lighting.pivot(index='orientation', columns='shade', values='lighting')
# df_light_heatmap = df_light_heatmap[column_order].astype(float)
# sim_heatmap(df_light_heatmap, "Lighting energy", 20)

# # Plot each orientations time series in its own facet
# g = sns.catplot(
#     data=df_energy,
#     x="shade", y="energy", col="orientation", palette=color_map_gradient,
#     kind="bar", zorder=5, col_wrap=2,
#     height=3.5, aspect=1.3, legend=False)
#
# # Tweak the supporting aspects of the plot
# g.set(ylim=(110, 120))
# g.tight_layout()
# # Increase the resolution and save as PNG
# plt.savefig("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                 "02_Working/Jpeg/Survey/sim_energy.png", dpi=300)
# plt.show()
#
# # Plot each year's time series in its own facet
# g = sns.catplot(
#     data=df_thermal,
#     x="shade", y="thermal", col="orientation", palette=color_map_gradient,
#     kind="bar", zorder=5, col_wrap=2,
#     height=3.5, aspect=1.3, legend=False)
#
# # Tweak the supporting aspects of the plot
# g.set(ylim=(5450, 5650))
# g.tight_layout()
# # Increase the resolution and save as PNG
# plt.savefig("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#             "02_Working/Jpeg/Survey/sim_thermal.png", dpi=300)
# plt.show()

# # Plot each year's time series in its own facet
# g = sns.catplot(
#     data=df_udi_1,
#     x="shade", y="udi_1", col="orientation", palette=color_map_gradient,
#     kind="bar", zorder=5, col_wrap=2,
#     height=3.5, aspect=1.3, legend=False)
#
# # Tweak the supporting aspects of the plot
# g.set(ylim=(0, 100))
# g.tight_layout()
# # Increase the resolution and save as PNG
# plt.savefig("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                 "02_Working/Jpeg/Survey/sim_udi.png", dpi=300)
# plt.show()

# # Plot each year's time series in its own facet
# g = sns.catplot(
#     data=df_dgp_1,
#     x="shade", y="dgp_1", col="orientation", palette=color_map_gradient,
#     kind="bar", zorder=5, col_wrap=2,
#     height=3.5, aspect=1.3, legend=False)
#
# # Tweak the supporting aspects of the plot
# g.tight_layout()
# # Increase the resolution and save as PNG
# plt.savefig("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                 "02_Working/Jpeg/Survey/sim_dgp.png", dpi=300)
# plt.show()
#
# print(df_energy)

# # Plot each year's time series in its own facet
# g = sns.catplot(
#     data=df_thermal_weighted,
#     x="shade", y="thermal_weighted", col="orientation", palette=color_map_gradient,
#     kind="bar", zorder=5, col_wrap=2,
#     height=3.5, aspect=1.3, legend=False)
#
# # Tweak the supporting aspects of the plot
# g.set(ylim=(3200, 3350))
# g.tight_layout()
# # Increase the resolution and save as PNG
# plt.savefig("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#             "02_Working/Jpeg/Survey/sim_thermal_weighted.png", dpi=300)
# plt.show()

# Pivot the dataframe
# Convert the 'orientation' column to categorical data type with the specified order
df_energy['orientation'] = pd.Categorical(df_energy['orientation'], categories=row_order, ordered=True)
df_energy_pivoted = df_energy.pivot(index='shade', columns='orientation', values='energy')
df_energy_pivoted = df_energy_pivoted.reset_index()

# Pivot the dataframe
df_thermal['orientation'] = pd.Categorical(df_thermal['orientation'], categories=row_order, ordered=True)
df_thermal_pivoted = df_thermal.pivot(index='shade', columns='orientation', values='thermal')
df_thermal_pivoted = df_thermal_pivoted.reset_index()

# Pivot the dataframe
df_ppd['orientation'] = pd.Categorical(df_ppd['orientation'], categories=row_order, ordered=True)
df_ppd_pivoted = df_ppd.pivot(index='shade', columns='orientation', values='ppd')
df_ppd_pivoted = df_ppd_pivoted.reset_index()

# Pivot the dataframe
df_thermal_weighted['orientation'] = pd.Categorical(df_ppd['orientation'], categories=row_order, ordered=True)
df_thermal_weighted_pivoted = df_thermal_weighted.pivot(index='shade', columns='orientation', values='thermal_weighted')
df_thermal_weighted_pivoted = df_thermal_weighted_pivoted.reset_index()

# Pivot the dataframe
df_udi_1['orientation'] = pd.Categorical(df_udi_1['orientation'], categories=row_order, ordered=True)
df_udi_pivoted_1 = df_udi_1.pivot(index='shade', columns='orientation', values='udi_1')
df_udi_pivoted_1 = df_udi_pivoted_1.reset_index()

# Pivot the dataframe
df_udi_3['orientation'] = pd.Categorical(df_udi_3['orientation'], categories=row_order, ordered=True)
df_udi_pivoted_3 = df_udi_3.pivot(index='shade', columns='orientation', values='udi_3')
df_udi_pivoted_3 = df_udi_pivoted_3.reset_index()

# Pivot the dataframe
df_udi_5['orientation'] = pd.Categorical(df_udi_5['orientation'], categories=row_order, ordered=True)
df_udi_pivoted_5 = df_udi_5.pivot(index='shade', columns='orientation', values='udi_5')
df_udi_pivoted_5 = df_udi_pivoted_5.reset_index()

# Pivot the dataframe
df_dgp_1['orientation'] = pd.Categorical(df_dgp_1['orientation'], categories=row_order, ordered=True)
df_dgp_pivoted_1 = df_dgp_1.pivot(index='shade', columns='orientation', values='dgp_1')
df_dgp_pivoted_1 = df_dgp_pivoted_1.reset_index()

# Pivot the dataframe
df_dgp_3['orientation'] = pd.Categorical(df_dgp_3['orientation'], categories=row_order, ordered=True)
df_dgp_pivoted_3 = df_dgp_3.pivot(index='shade', columns='orientation', values='dgp_3')
df_dgp_pivoted_3 = df_dgp_pivoted_3.reset_index()

# Pivot the dataframe
df_dgp_5['orientation'] = pd.Categorical(df_dgp_5['orientation'], categories=row_order, ordered=True)
df_dgp_pivoted_5 = df_dgp_5.pivot(index='shade', columns='orientation', values='dgp_5')
df_dgp_pivoted_5 = df_dgp_pivoted_5.reset_index()

# #Batch plotting
# parallel(df_energy_pivoted, 'Energy use', 'energy')
# parallel(df_thermal_pivoted, 'Predicted mean vote', 'pmv')
# parallel(df_ppd_pivoted, 'Percentage People Dissatisfied', 'ppd')
# parallel(df_udi_pivoted_1, 'Useful daylight illuminance at 1m from window', 'udi_1')
# parallel(df_udi_pivoted_3, 'Useful daylight illuminance at 3m from window', 'udi_3')
# parallel(df_udi_pivoted_5, 'Useful daylight illuminance at 5m from window', 'udi_5')
# parallel(df_dgp_pivoted_1, 'Daylight glare probability at 1m from window', 'dgp_1')
# parallel(df_dgp_pivoted_3, 'Daylight glare probability at 3m from window', 'dgp_3')
# parallel(df_dgp_pivoted_5, 'Daylight glare probability at 5m from window', 'dgp_5')
# parallel(df_thermal_weighted_pivoted, 'Weighted thermal discomfort rating', 'thermal')

# print(df_dgp_pivoted_5)

combined_east = pd.concat({'df_energy': df_energy_pivoted['East'],
                           'df_thermal': df_thermal_weighted_pivoted['East'],
                           'df_udi_1': df_udi_pivoted_1['East'],
                           'df_udi_3': df_udi_pivoted_3['East'],
                           'df_udi_5': df_udi_pivoted_5['East'],
                           'df_dgp_1': df_dgp_pivoted_1['East'],
                           'df_dgp_3': df_dgp_pivoted_3['East'],
                           'df_dgp_5': df_dgp_pivoted_5['East']}, axis=1)

print(combined_east)

# Prepare data for the radial plot
values = combined_east.values
num_rows, num_columns = values.shape

# Set the angles
theta = np.linspace(0, 2 * np.pi, num_columns, endpoint=False)

# Create the plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
for i in range(num_rows):
    ax.plot(theta, values[i, :], label=combined_east.index[i])

# Set the radial labels
ax.set_xticks(theta)
ax.set_xticklabels(combined_east.columns, fontsize=8)

# Set the title and legend
ax.set_title('Radial Plot of East Column Values')
ax.legend()

# Show the plot
plt.show()