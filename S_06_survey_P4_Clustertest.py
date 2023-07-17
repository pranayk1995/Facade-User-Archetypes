import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd

# Load CSV as dataframe
df = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_clustered.csv')

ar = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/Archetype.csv')

fs = 12
pad = 12
color_0_background = '#f5f5f5'
color_1_base = '#6D909C'
method = 'ward'
group_grad_4 = sns.color_palette(['#8fa9b2', '#89b8a6', '#d6b680', '#AB8788'])
cmap_custom = ListedColormap(['#8fa9b2', '#89b8a6', '#d6b680', '#AB8788'])

# select columns to plot as a radial plot
cols_to_plot = ar.loc[:, ['sr_view4_rb_05', 'sr_view4_rb_10', 'sr_view4_vb_25',
                          'sr_view4_vb_50']]

# create figure and axes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
for i, row in ar.iterrows():
    values = row[cols_to_plot.columns].values
    angles = [n / float(len(cols_to_plot.columns)) * 2 * np.pi for n in range(len(cols_to_plot.columns))]
    angles += angles[:1]
    values = np.append(values, values[:1])
    ax.plot(angles, values, 'o-', linewidth=2, markersize=8)
ax.set_title('Radial Plot of Selected Columns', fontsize=20)
plt.show()

# select first 13 columns of dataframe
cols_to_plot = ar.loc[:, ['sr_view1_rb_05_L', 'sr_view1_rb_05_M', 'sr_view1_rb_05_D',
                          'sr_view1_vb_25_L', 'sr_view1_vb_25_D']]

# create figure and axes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
for i, row in ar.iterrows():
    values = row[cols_to_plot.columns].values
    angles = [n / float(len(cols_to_plot.columns)) * 2 * np.pi for n in range(len(cols_to_plot.columns))]
    angles += angles[:1]
    values = np.append(values, values[:1])
    ax.plot(angles, values, 'o-', linewidth=2, markersize=8)
ax.set_title('Radial Plot of Selected Columns', fontsize=20)
plt.show()

# select first 13 columns of dataframe
cols_to_plot = ar.loc[:, ['rb_05', 'rb_10', 'vb_25', 'vb_50']]

# create figure and axes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
for i, row in ar.iterrows():
    values = row[cols_to_plot.columns].values
    angles = [n / float(len(cols_to_plot.columns)) * 2 * np.pi for n in range(len(cols_to_plot.columns))]
    angles += angles[:1]
    values = np.append(values, values[:1])
    ax.plot(angles, values, 'o-', linewidth=2, markersize=8)
ax.set_title('Radial Plot of Selected Columns', fontsize=20)
plt.show()

# select first 13 columns of dataframe
cols_to_plot = ar.loc[:, ['weight_temperature', 'weight_energy', 'weight_daylight', 'weight_glare',
                          'weight_view', 'weight_interiors']]

# create figure and axes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
for i, row in ar.iterrows():
    values = row[cols_to_plot.columns].values
    angles = [n / float(len(cols_to_plot.columns)) * 2 * np.pi for n in range(len(cols_to_plot.columns))]
    angles += angles[:1]
    values = np.append(values, values[:1])
    ax.plot(angles, values, 'o-', linewidth=2, markersize=8)
ax.set_title('Radial Plot of Selected Columns', fontsize=20)
plt.show()
