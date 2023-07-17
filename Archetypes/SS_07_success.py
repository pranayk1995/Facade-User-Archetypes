import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define your custom colors
color_0_background = '#f5f5f5'
custom_colors_RGB = ['#336371', '#8E5F60']
custom_colors_R = ['#E3C8C8', '#6D4848']
color_1_base = '#6D909C'
color_1_base_dark = '#5B7C86'
color_1_base_light = '#C2D1D6'
group_1_blues = sns.color_palette([color_1_base_dark, color_1_base, color_1_base_light])


df = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/success.csv', index_col=0)



archetype_counts = (df.groupby('archetype').sum())
print(archetype_counts)

# Specify the value you want to divide by
archetype_0 = 35
archetype_1 = 17
archetype_2 = 80
archetype_3 = 39

# Divide the values in the specified columns by the divisor
archetype_counts.loc[0] = archetype_counts.loc[0] / archetype_0
archetype_counts.loc[1] = archetype_counts.loc[1] / archetype_1
archetype_counts.loc[2] = archetype_counts.loc[2] / archetype_2
archetype_counts.loc[3] = archetype_counts.loc[3] / archetype_3

success_rate_percentages_latex = archetype_counts.to_latex(float_format="%.2f")
print(success_rate_percentages_latex)