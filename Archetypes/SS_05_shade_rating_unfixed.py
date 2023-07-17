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
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_clustered_ss.csv')

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

cluster_weight = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                              "02_Working/Excel/02_Qualtrics Data/Archetype_ss.csv")

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

output_df = pd.DataFrame(columns=['column1_obj', 'column2_obj', 'column3_obj',
                                  'column1_ind', 'column2_ind', 'column3_ind',
                                  'column1_clu', 'column2_clu', 'column3_clu'])

# Iterate over the rows of the DataFrame using iterrows()
for index, row in df.iterrows():
    # Initial selection based on distance from window
    if row['cf_window_distance'] == 1:
        window_dist = 1
    elif row['cf_window_distance'] == 0.75:
        window_dist = 3
    else:
        window_dist = 5

    # Initial selection based on orientation of window
    if row['cf_window_orientation'] == 0:
        orientation = 0
    elif row['cf_window_orientation'] == 1:
        orientation = 1
    elif row['cf_window_orientation'] == 2:
        orientation = 2
    elif row['cf_window_orientation'] == 3:
        orientation = 3
    elif row['cf_window_orientation'] == 4:
        orientation = 4
    elif row['cf_window_orientation'] == 5:
        orientation = 5
    elif row['cf_window_orientation'] == 6:
        orientation = 6
    else:
        orientation = 7

    # Generate the variable names for the dataframes to be added using string formatting
    dgp_var = "dgp_{}".format(window_dist)
    udi_var = "udi_{}".format(window_dist)

    #Establishing cluster allocation of respondent as per kmeans
    occ_clu = row['archetype']
    print("this is archetype:", occ_clu)
    # indexed_weights = cluster_weight.loc[occ_clu, 'sr_view1_rb_05']
    # print(indexed_weights)

    # Running an initial ranking based on objective parameters
    ranking_objective = (energy + pmv + eval(dgp_var) + eval(udi_var))/7
    ranking_objective = ranking_objective.iloc[orientation]
    column_name_objective = ranking_objective.idxmax()
    top_3_columns_objective = ranking_objective.nlargest(3).index
    column1_obj = top_3_columns_objective[0]
    column2_obj = top_3_columns_objective[1]
    column3_obj = top_3_columns_objective[2]

    # print(heat)
    # print(cool)
    # print(light)
    # print(eval(dgp_var))
    # print(eval(udi_var))
    # print(ranking_objective)

    # Sort the columns in descending order based on their values
    sorted_columns = ranking_objective.sort_values(ascending=False)

    # Get the names of the sorted columns as a list
    sorted_column_names = sorted_columns.index.tolist()

    # Convert the list to a string with column names in decreasing rating order
    column_names_str_1 = ', '.join(sorted_column_names)

    value = ranking_objective

    # Create a dictionary with column names as keys and values as values
    new_values = {
        'RB_05_L': value['RB_05_L'],
        'RB_05_M': value['RB_05_M'],
        'RB_05_D': value['RB_05_D'],
        'RB_10_L': value['RB_10_L'],
        'RB_10_M': value['RB_10_M'],
        'RB_10_D': value['RB_10_D'],
        'VB_25_L': value['VB_25_L'],
        'VB_25_M': value['VB_25_M'],
        'VB_25_D': value['VB_25_D'],
        'VB_50_L': value['VB_50_L'],
        'VB_50_M': value['VB_50_M'],
        'VB_50_D': value['VB_50_D'],
        # 'Highest_original': column_best
    }

    # Update the current row of the DataFrame with the new values
    df.loc[index, new_values.keys()] = list(new_values.values())

    rowz = ['South', 'South West', 'West', 'North West', 'North', 'North East', 'East', 'South East']

    # The part from here on assigns scores for view quality through shades based on the user rating.
    vqi = pd.DataFrame(index=rowz, columns=['RB_05_L', 'RB_05_M', 'RB_05_D', 'RB_10_L', 'RB_10_M', 'RB_10_D',
                                            'VB_25_L', 'VB_25_M', 'VB_25_D', 'VB_50_L', 'VB_50_M', 'VB_50_D'])

    # Assign scores based on OF and slat size
    RB_05 = ((row['sr_view1_rb_05'] + row['sr_view2_rb_05'] + row['sr_view3_rb_05']) / 3) + 0.001
    RB_10 = ((row['sr_view1_rb_10'] + row['sr_view2_rb_10'] + row['sr_view3_rb_10']) / 3) + 0.001
    VB_25 = ((row['sr_view1_vb_25'] + row['sr_view2_vb_25'] + row['sr_view3_vb_25']) / 3) + 0.001
    VB_50 = ((row['sr_view1_vb_50'] + row['sr_view2_vb_50'] + row['sr_view3_vb_50']) / 3) + 0.001

    RB_L = row['sr_view1_rb_05_L'] + 0.001
    RB_M = row['sr_view1_rb_05_M'] + 0.001
    RB_D = row['sr_view1_rb_05_D'] + 0.001

    VB_L = row['sr_view1_vb_25_L'] + 0.001
    VB_D = row['sr_view1_vb_25_D'] + 0.001

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
    vqi.iloc[:, :] = values.reshape((8, 12))

    # The part from here on assigns scores for view quality through shades based on the user rating.

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
    int = pd.DataFrame(index=rowz, columns=['RB_05_L', 'RB_05_M', 'RB_05_D', 'RB_10_L', 'RB_10_M', 'RB_10_D',
                                                'VB_25_L', 'VB_25_M', 'VB_25_D', 'VB_50_L', 'VB_50_M', 'VB_50_D'])

    RB_05_L_interiors = (row['sr_view4_rb_05']) + 0.001
    RB_05_M_interiors = (row['sr_view4_rb_05']) + 0.001
    RB_05_D_interiors = (row['sr_view4_rb_05']) + 0.001
    RB_10_L_interiors = (row['sr_view4_rb_10']) + 0.001
    RB_10_M_interiors = (row['sr_view4_rb_10']) + 0.001
    RB_10_D_interiors = (row['sr_view4_rb_10']) + 0.001
    VB_25_L_interiors = (row['sr_view4_vb_25']) + 0.001
    VB_25_M_interiors = (row['sr_view4_vb_25']) + 0.001
    VB_25_D_interiors = (row['sr_view4_vb_25']) + 0.001
    VB_50_L_interiors = (row['sr_view4_vb_50']) + 0.001
    VB_50_M_interiors = (row['sr_view4_vb_50']) + 0.001
    VB_50_D_interiors = (row['sr_view4_vb_50']) + 0.001

    interior_scores = {'RB_05_L': RB_05_L_interiors, 'RB_05_M': RB_05_M_interiors, 'RB_05_D': RB_05_D_interiors,
                       'RB_10_L': RB_10_L_interiors, 'RB_10_M': RB_10_M_interiors, 'RB_10_D': RB_10_D_interiors,
                       'VB_25_L': VB_25_L_interiors, 'VB_25_M': VB_25_M_interiors, 'VB_25_D': VB_25_D_interiors,
                       'VB_50_L': VB_50_L_interiors, 'VB_50_M': VB_50_M_interiors, 'VB_50_D': VB_50_D_interiors}

    # Repeat each value 8 times
    interior_values = np.tile(list(interior_scores.values()), 8)

    # Assign the values to the DataFrame
    int.iloc[:, :] = interior_values.reshape((8, 12))

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

    factor_thermal = (row['ef_importance_temperature'] + row['ef_productivity_temperature'])/2
    factor_energy = ((row['psy_e_consciously_sustainable'] + row['psy_e_change_sustainable'] + row['psy_e_compromise_comfort_sustainabile'])/3)
    factor_udi = (row['ef_importance_daylight'] + row['ef_productivity_daylight'])/2
    factor_dgp = (row['ef_importance_glare'] + row['ef_productivity_glare'])/2
    factor_vqi = (row['ef_importance_view'] + row['ef_productivity_view'])/2
    factor_int = row['psy_c_importance_interior_features']

    summation = (factor_thermal + factor_energy + factor_udi + factor_dgp + factor_vqi + factor_int)
    weight_thermal = factor_thermal / summation
    weight_energy = factor_energy / summation
    weight_udi = factor_udi / summation
    weight_dgp = factor_dgp / summation
    weight_vqi = factor_vqi / summation
    weight_int = factor_int / summation

    weights_individual = [weight_energy, weight_thermal, weight_dgp, weight_udi, weight_vqi, weight_int]

    factor_thermal_cluster = (cluster_weight.loc[occ_clu, 'weight_temperature'])
    factor_energy_cluster = (cluster_weight.loc[occ_clu, 'weight_energy'])
    factor_udi_cluster = (cluster_weight.loc[occ_clu, 'weight_daylight'])
    factor_dgp_cluster = (cluster_weight.loc[occ_clu, 'weight_glare'])
    factor_vqi_cluster = (cluster_weight.loc[occ_clu, 'weight_view'])
    factor_int_cluster = (cluster_weight.loc[occ_clu, 'weight_interiors'])

    summation_cluster = (factor_thermal_cluster + factor_energy_cluster + factor_udi_cluster +
                 factor_dgp_cluster + factor_vqi_cluster + factor_int_cluster)
    weight_thermal_cluster = factor_thermal_cluster / summation_cluster
    weight_energy_cluster = factor_energy_cluster / summation_cluster
    weight_udi_cluster = factor_udi_cluster / summation_cluster
    weight_dgp_cluster = factor_dgp_cluster / summation_cluster
    weight_vqi_cluster = factor_vqi_cluster / summation_cluster
    weight_int_cluster = factor_int_cluster / summation_cluster

    weights_cluster = [weight_energy_cluster, weight_thermal_cluster, weight_dgp_cluster,
                       weight_udi_cluster, weight_vqi_cluster, weight_int_cluster]

    score_individual = (energy * weights_individual[0]) + (pmv * weights_individual[1]) + (eval(dgp_var) * weights_individual[2]) + (eval(udi_var) * weights_individual[3]) + (vqi * weights_individual[4]) + (int * weights_individual[5])

    score_cluster = (energy * weights_cluster[0]) + \
                    (pmv * weights_cluster[1]) + \
                    (eval(dgp_var) * weights_cluster[2]) + \
                    (eval(udi_var) * weights_cluster[3]) + \
                    (vqi_clu * weights_cluster[4]) + \
                    (int_clu * weights_cluster[5])

    # Find the column with the highest adjusted ranking score
    ranking_individual = score_individual.iloc[orientation]
    ranking_individual = ranking_individual.astype(float)
    column_name_individual = ranking_individual.idxmax()
    top_3_columns_individual = ranking_individual.nlargest(3).index
    column1_ind = top_3_columns_individual[0]
    column2_ind = top_3_columns_individual[1]
    column3_ind = top_3_columns_individual[2]

    # print(ranking_individual)

    # Find the column with the highest adjusted ranking score
    ranking_cluster = score_cluster.iloc[orientation]
    ranking_cluster = ranking_cluster.astype(float)
    column_name_cluster = ranking_cluster.idxmax()
    top_3_columns_cluster = ranking_cluster.nlargest(3).index
    column1_clu = top_3_columns_cluster[0]
    column2_clu = top_3_columns_cluster[1]
    column3_clu = top_3_columns_cluster[2]

    # print(ranking_cluster)

    print("Objective:", column_name_objective, "Preferred:", column_name_individual, "Clustered:", column_name_cluster)

    # Create a dictionary with column names as keys and values as values
    new_values = {
        'RB_05_L_Adj': value['RB_05_L'],
        'RB_05_M_Adj': value['RB_05_M'],
        'RB_05_D_Adj': value['RB_05_D'],
        'RB_10_L_Adj': value['RB_10_L'],
        'RB_10_M_Adj': value['RB_10_M'],
        'RB_10_D_Adj': value['RB_10_D'],
        'VB_25_L_Adj': value['VB_25_L'],
        'VB_25_M_Adj': value['VB_25_M'],
        'VB_25_D_Adj': value['VB_25_D'],
        'VB_50_L_Adj': value['VB_50_L'],
        'VB_50_M_Adj': value['VB_50_M'],
        'VB_50_D_Adj': value['VB_50_D'],
        'Highest_Scored_Column_obj': column_name_objective,
        'Highest_Scored_Column_ind': column_name_individual,
        'Highest_Scored_Column_clu': column_name_cluster,
    }

    # Update the current row of the DataFrame with the new values
    df.loc[index, new_values.keys()] = list(new_values.values())

    if column_name_cluster != column_name_objective:
        count += 1

    if column_name_cluster == column_name_individual:
        count_ml += 1

    print(ranking_objective)
    print(ranking_individual)
    print(ranking_cluster)
    output_df = output_df.append({'column1_obj': column1_obj, 'column2_obj': column2_obj,
                                  'column3_obj': column3_obj, 'column1_ind': column1_ind,
                                  'column2_ind': column2_ind, 'column3_ind': column3_ind,
                                  'column1_clu': column1_clu, 'column2_clu': column2_clu,
                                  'column3_clu': column3_clu}, ignore_index=True)


plt.rcParams.update({'figure.autolayout': True})
sns.set(rc={'figure.figsize': (20, 10)})
sns.set_style("darkgrid", {"axes.facecolor": color_0_background})

category_order = ['RB_05_L', 'RB_05_M', 'RB_05_D', 'RB_10_L', 'RB_10_M', 'RB_10_D',
                  'VB_25_L', 'VB_25_M', 'VB_25_D', 'VB_50_L', 'VB_50_M', 'VB_50_D']

# Plot a histogram of the count of each variable in the 'education' column
objective = output_df[['column1_obj', 'column2_obj', 'column3_obj']]
df_melted = objective.melt(var_name='column', value_name='objective')
plot_1 = sns.countplot(data=df_melted, hue='column', x='objective', order=category_order,
                       palette=group_1_blues)
plot_1.set_xticklabels(plot_1.get_xticklabels(), rotation=45)
update_plot(plot_1, 20)
plot_1.set_xlabel('Shades', fontsize=20)
plot_1.set_ylabel('Count', fontsize=20)
plot_1.set_ylim(0, 160)
# Add count labels on top of each bar
for p in plot_1.patches:
    plot_1.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=15)
plot_1.legend(labels=['Rank 1', 'Rank 2', 'Rank 3'], loc='upper right', bbox_to_anchor=(1, 1), fontsize=15)
plt.xticks()
plt.savefig("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
            "02_Working/Jpeg/Survey/evaluation_1.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot a histogram of the count of each variable in the 'education' column
individual = output_df[['column1_ind', 'column2_ind', 'column3_ind']]
df_melted = individual.melt(var_name='column', value_name='individual')
plot_1 = sns.countplot(data=df_melted, hue='column', x='individual', order=category_order,
                       palette=group_1_blues)
plot_1.set_xticklabels(plot_1.get_xticklabels(), rotation=45)
update_plot(plot_1, 20)
plot_1.set_xlabel('Shades', fontsize=20)
plot_1.set_ylabel('Count', fontsize=20)
plot_1.set_ylim(0, 160)
# Add count labels on top of each bar
for p in plot_1.patches:
    plot_1.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=15)
plot_1.legend(labels=['Rank 1', 'Rank 2', 'Rank 3'], loc='upper right', bbox_to_anchor=(1, 1), fontsize=15)
plt.xticks()
plt.savefig("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
            "02_Working/Jpeg/Survey/evaluation_2.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot a histogram of the count of each variable in the 'education' column
cluster = output_df[['column1_clu', 'column2_clu', 'column3_clu']]
df_melted = cluster.melt(var_name='column', value_name='cluster')
plot_1 = sns.countplot(data=df_melted, hue='column', x='cluster', order=category_order,
                       palette=group_1_blues)
plot_1.set_xticklabels(plot_1.get_xticklabels(), rotation=45)
update_plot(plot_1, 20)
plot_1.set_xlabel('Country', fontsize=20)
plot_1.set_ylabel('Count', fontsize=20)
plot_1.set_ylim(0, 160)
# Add count labels on top of each bar
for p in plot_1.patches:
    plot_1.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=15)
plot_1.legend(labels=['Rank 1', 'Rank 2', 'Rank 3'], loc='upper right', bbox_to_anchor=(1, 1), fontsize=15)
plt.xticks()
plt.savefig("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
            "02_Working/Jpeg/Survey/evaluation_3.png", dpi=300, bbox_inches='tight')
plt.show()


# count the frequency of each label
counts_obj = df['Highest_Scored_Column_obj'].value_counts()
counts_ind = df['Highest_Scored_Column_ind'].value_counts()
counts_clu = df['Highest_Scored_Column_clu'].value_counts()

print(counts_obj)
print(count/171)
print(count_ml/171)

df.to_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
            '02_Working/Excel/02_Qualtrics Data/survey_text_data_preprocess_rated.csv')