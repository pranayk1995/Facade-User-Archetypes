from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.manifold import SpectralEmbedding
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestRegressor
from psynlig import pca_residual_variance
from sklearn.metrics import mean_squared_error

# Load CSV as dataframe
df = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_preprocess_pre_encoded.csv')

fs = 20
pad = 12
color_0_background = '#f5f5f5'
color_0_background_d = '#b8b8b8'
color_1_base = '#6D909C'
color_1_base_dark = '#5B7C86'
color_1_base_light = '#C2D1D6'
color_2_base = '#AB8788'
color_3_base = '#8fb3b0'
color_4_base = '#d6b680'
group_1_blues = sns.color_palette([color_1_base_dark, color_1_base, color_1_base_light])
group_2_duals = sns.color_palette([color_1_base, color_2_base])
group_grad_4 = sns.color_palette(['#8fa9b2', '#89b8a6', '#d6b680', '#AB8788'])
group_grad_5 = sns.color_palette(['#8fa9b2', '#89b8a6', '#b6b596', '#d6b680', '#c2a8a9'])
group_grad_8 = sns.color_palette(['#8fa9b2', '#8cb2ab', '#89b8a6', '#a9b18f',
                                  '#b8ae85', '#c7aa7a', '#c4a993', '#c2a8a9'])
group_likert_5 = sns.color_palette(['#c2a8a9', '#cfc2c2', '#dbdbdb', '#adbdc2', '#8fa9b2'])




def scatter(transformed_datas, label, plot_num, ev, rmse):
    figs = plt.figure(figsize=(10, 10))
    axs = figs.add_subplot(111, projection='3d')
    axs.scatter(transformed_datas[:, 0], transformed_datas[:, 1], transformed_datas[:, 2], c=color_1_base)
    axs.w_xaxis.pane.set_facecolor(color_0_background)
    axs.w_yaxis.pane.set_facecolor(color_0_background)
    axs.w_zaxis.pane.set_facecolor(color_0_background)
    axs.set_xlabel('PC1', fontsize=fs, labelpad=pad)
    axs.set_ylabel('PC2', fontsize=fs, labelpad=pad)
    axs.set_zlabel('PC3', fontsize=fs, labelpad=pad)
    axs.tick_params(axis='x', labelsize=fs)
    axs.tick_params(axis='y', labelsize=fs)
    axs.tick_params(axis='z', labelsize=fs)
    axs.text2D(0.5, -0.05, label, transform=axs.transAxes, fontsize=fs, horizontalalignment='center')
    axs.text2D(0.5, -0.10, ev, transform=axs.transAxes, fontsize=fs, horizontalalignment='center')
    axs.text2D(0.5, -0.15, rmse, transform=axs.transAxes, fontsize=fs, horizontalalignment='center')
    plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                f"02_Working/Jpeg/self_supervised/01_dimensionality_{plot_num}.png", dpi=300, bbox_inches='tight')
    plt.show(block=True)


method_1 = ['psy_e_consciously_sustainable', 'psy_e_change_sustainable', 'psy_e_compromise_comfort_sustainabile',
            'ef_importance_temperature', 'ef_importance_view',  'ef_importance_daylight',
            'ef_importance_glare', 'ef_productivity_temperature', 'ef_productivity_view',
            'ef_productivity_daylight', 'ef_productivity_glare', 'psy_c_importance_interior_features']

method_2 = ['psy_e_consciously_sustainable', 'psy_e_change_sustainable', 'psy_e_compromise_comfort_sustainabile',
            'ef_importance_temperature', 'ef_importance_view',  'ef_importance_daylight',
            'ef_importance_glare', 'ef_productivity_temperature', 'ef_productivity_view',
            'ef_productivity_daylight', 'ef_productivity_glare', 'psy_c_importance_interior_features',
            'sf_use_view_outside', 'sf_use_adequate_lighting', 'sf_use_glare_mitigation',
            'sf_use_thermal', 'sf_use_spatial_aesthetic']

method_3 = ['psy_e_consciously_sustainable', 'psy_e_change_sustainable', 'psy_e_compromise_comfort_sustainabile',
            'ef_importance_temperature', 'ef_importance_view',  'ef_importance_daylight',
            'ef_importance_glare', 'ef_productivity_temperature', 'ef_productivity_view',
            'ef_productivity_daylight', 'ef_productivity_glare', 'psy_c_importance_interior_features',
            'sr_view1_rb_05', 'sr_view1_rb_10', 'sr_view1_vb_25', 'sr_view1_vb_50', 'sr_view2_rb_05',
            'sr_view2_rb_10', 'sr_view2_vb_25', 'sr_view2_vb_50', 'sr_view3_rb_05', 'sr_view3_rb_10',
            'sr_view3_vb_25', 'sr_view3_vb_50', 'sr_view4_rb_05', 'sr_view4_rb_10', 'sr_view4_vb_25',
            'sr_view4_vb_50', 'sr_view1_rb_05_L', 'sr_view1_rb_05_M', 'sr_view1_rb_05_D', 'sr_view1_vb_25_L',
            'sr_view1_vb_25_D']

method_4 = ['psy_e_consciously_sustainable', 'psy_e_change_sustainable', 'psy_e_compromise_comfort_sustainabile',
            'ef_importance_temperature', 'ef_importance_view',  'ef_importance_daylight',
            'ef_importance_glare', 'ef_productivity_temperature', 'ef_productivity_view',
            'ef_productivity_daylight', 'ef_productivity_glare', 'psy_c_importance_interior_features',
            'sr_view1_rb_05', 'sr_view1_rb_10', 'sr_view1_vb_25', 'sr_view1_vb_50', 'sr_view2_rb_05',
            'sr_view2_rb_10', 'sr_view2_vb_25', 'sr_view2_vb_50', 'sr_view3_rb_05', 'sr_view3_rb_10',
            'sr_view3_vb_25', 'sr_view3_vb_50', 'sr_view4_rb_05', 'sr_view4_rb_10', 'sr_view4_vb_25',
            'sr_view4_vb_50', 'sr_view1_rb_05_L', 'sr_view1_rb_05_M', 'sr_view1_rb_05_D', 'sr_view1_vb_25_L',
            'sr_view1_vb_25_D', 'sf_use_view_outside', 'sf_use_adequate_lighting',
            'sf_use_glare_mitigation', 'sf_use_thermal', 'sf_use_spatial_aesthetic']

df_method_1 = df[method_1]
df_method_2 = df[method_2]
df_method_3 = df[method_3]
df_method_4 = df[method_4]

# Method 1 transformation
df_method_1['weight_energy'] = df_method_1[['psy_e_consciously_sustainable',
                                            'psy_e_change_sustainable',
                                            'psy_e_compromise_comfort_sustainabile']].mean(axis=1)
df_method_1['weight_temperature'] = df_method_1[['ef_importance_temperature', 'ef_productivity_temperature']].mean(axis=1)
df_method_1['weight_daylight'] = df_method_1[['ef_importance_daylight', 'ef_productivity_daylight']].mean(axis=1)
df_method_1['weight_glare'] = df_method_1[['ef_importance_glare', 'ef_productivity_glare']].mean(axis=1)
df_method_1['weight_view'] = df_method_1[['ef_importance_view', 'ef_productivity_view']].mean(axis=1)
df_method_1['weight_interiors'] = df_method_1[['psy_c_importance_interior_features']]
df_method_1 = df_method_1.drop(['psy_e_consciously_sustainable', 'psy_e_change_sustainable',
                                'psy_e_compromise_comfort_sustainabile', 'ef_importance_temperature',
                                'ef_productivity_temperature', 'ef_importance_daylight',
                                'ef_productivity_daylight', 'ef_importance_glare', 'ef_productivity_glare',
                                'ef_importance_view', 'ef_productivity_view', 'psy_c_importance_interior_features'], axis=1)

print(df_method_1)

# Method 2 transformation
df_method_2['weight_energy'] = df_method_2[['psy_e_consciously_sustainable',
                                            'psy_e_change_sustainable',
                                            'psy_e_compromise_comfort_sustainabile']].mean(axis=1)
df_method_2['weight_temperature'] = df_method_2[['ef_importance_temperature',
                                                'ef_productivity_temperature',
                                                'sf_use_thermal']].mean(axis=1)
df_method_2['weight_daylight'] = df_method_2[['ef_importance_daylight',
                                              'ef_productivity_daylight',
                                              'sf_use_adequate_lighting']].mean(axis=1)
df_method_2['weight_glare'] = df_method_2[['ef_importance_glare',
                                           'ef_productivity_glare',
                                           'sf_use_glare_mitigation']].mean(axis=1)
df_method_2['weight_view'] = df_method_2[['ef_importance_view',
                                          'ef_productivity_view',
                                          'sf_use_view_outside']].mean(axis=1)
df_method_2['weight_interiors'] = df_method_2[['psy_c_importance_interior_features',
                                       'sf_use_spatial_aesthetic']].mean(axis=1)
df_method_2 = df_method_2.drop(['psy_e_consciously_sustainable', 'psy_e_change_sustainable',
                                'psy_e_compromise_comfort_sustainabile', 'ef_importance_temperature',
                                'ef_productivity_temperature', 'ef_importance_daylight',
                                'ef_productivity_daylight', 'ef_importance_glare', 'ef_productivity_glare',
                                'ef_importance_view', 'ef_productivity_view', 'psy_c_importance_interior_features',
                                'sf_use_thermal', 'sf_use_adequate_lighting', 'sf_use_glare_mitigation',
                                'sf_use_view_outside', 'sf_use_spatial_aesthetic'], axis=1)

#Method 3 transformation
df_method_3['weight_energy'] = df_method_3[['psy_e_consciously_sustainable',
                                    'psy_e_change_sustainable',
                                    'psy_e_compromise_comfort_sustainabile']].mean(axis=1)
df_method_3['weight_temperature'] = df_method_3[['ef_importance_temperature', 'ef_productivity_temperature']].mean(axis=1)
df_method_3['weight_daylight'] = df_method_3[['ef_importance_daylight', 'ef_productivity_daylight']].mean(axis=1)
df_method_3['weight_glare'] = df_method_3[['ef_importance_glare', 'ef_productivity_glare']].mean(axis=1)
df_method_3['weight_view'] = df_method_3[['ef_importance_view', 'ef_productivity_view']].mean(axis=1)
df_method_3['weight_interiors'] = df_method_3[['psy_c_importance_interior_features']]
df_method_3['rb_05'] = df_method_3[['sr_view1_rb_05', 'sr_view2_rb_05', 'sr_view3_rb_05']].mean(axis=1)
df_method_3['rb_10'] = df_method_3[['sr_view1_rb_10', 'sr_view2_rb_10', 'sr_view3_rb_10']].mean(axis=1)
df_method_3['vb_25'] = df_method_3[['sr_view1_vb_25', 'sr_view2_vb_25', 'sr_view3_vb_25']].mean(axis=1)
df_method_3['vb_50'] = df_method_3[['sr_view1_vb_50', 'sr_view2_vb_50', 'sr_view3_vb_50']].mean(axis=1)
df_method_3 = df_method_3.drop(['psy_e_consciously_sustainable', 'psy_e_change_sustainable',
                                'psy_e_compromise_comfort_sustainabile', 'ef_importance_temperature',
                                'ef_productivity_temperature', 'ef_importance_daylight',
                                'ef_productivity_daylight', 'ef_importance_glare', 'ef_productivity_glare',
                                'ef_importance_view', 'ef_productivity_view', 'psy_c_importance_interior_features',
                                'sr_view1_rb_05', 'sr_view2_rb_05', 'sr_view3_rb_05',
                                'sr_view1_rb_10', 'sr_view2_rb_10', 'sr_view3_rb_10',
                                'sr_view1_vb_25', 'sr_view2_vb_25', 'sr_view3_vb_25',
                                'sr_view1_vb_50', 'sr_view2_vb_50', 'sr_view3_vb_50'], axis=1)

#Method 4 transformation
df_method_4['weight_energy'] = df_method_4[['psy_e_consciously_sustainable',
                                            'psy_e_change_sustainable',
                                            'psy_e_compromise_comfort_sustainabile']].mean(axis=1)
df_method_4['weight_temperature'] = df_method_4[['ef_importance_temperature',
                                                 'ef_productivity_temperature',
                                                 'sf_use_thermal']].mean(axis=1)
df_method_4['weight_daylight'] = df_method_4[['ef_importance_daylight',
                                              'ef_productivity_daylight',
                                              'sf_use_adequate_lighting']].mean(axis=1)
df_method_4['weight_glare'] = df_method_4[['ef_importance_glare',
                                           'ef_productivity_glare',
                                           'sf_use_glare_mitigation']].mean(axis=1)
df_method_4['weight_view'] = df_method_4[['ef_importance_view',
                                          'ef_productivity_view',
                                          'sf_use_view_outside']].mean(axis=1)
df_method_4['weight_interiors'] = df_method_4[['psy_c_importance_interior_features']]
df_method_4['rb_05'] = df_method_4[['sr_view1_rb_05', 'sr_view2_rb_05', 'sr_view3_rb_05']].mean(axis=1)
df_method_4['rb_10'] = df_method_4[['sr_view1_rb_10', 'sr_view2_rb_10', 'sr_view3_rb_10']].mean(axis=1)
df_method_4['vb_25'] = df_method_4[['sr_view1_vb_25', 'sr_view2_vb_25', 'sr_view3_vb_25']].mean(axis=1)
df_method_4['vb_50'] = df_method_4[['sr_view1_vb_50', 'sr_view2_vb_50', 'sr_view3_vb_50']].mean(axis=1)
df_method_4 = df_method_4.drop(['psy_e_consciously_sustainable', 'psy_e_change_sustainable',
                                'psy_e_compromise_comfort_sustainabile', 'ef_importance_temperature',
                                'ef_productivity_temperature', 'ef_importance_daylight',
                                'ef_productivity_daylight', 'ef_importance_glare', 'ef_productivity_glare',
                                'ef_importance_view', 'ef_productivity_view', 'psy_c_importance_interior_features',
                                'sr_view1_rb_05', 'sr_view2_rb_05', 'sr_view3_rb_05',
                                'sr_view1_rb_10', 'sr_view2_rb_10', 'sr_view3_rb_10',
                                'sr_view1_vb_25', 'sr_view2_vb_25', 'sr_view3_vb_25',
                                'sr_view1_vb_50', 'sr_view2_vb_50', 'sr_view3_vb_50',
                                'sf_use_thermal', 'sf_use_adequate_lighting', 'sf_use_glare_mitigation',
                                'sf_use_view_outside', 'sf_use_spatial_aesthetic'], axis=1)

dataframe_list = [df_method_1, df_method_2, df_method_3, df_method_4]

dataframe_list = [df_method_3]

pca_explained_variances = []
num_rows = 19  # Specify the desired number of rows
column_names = ['method_3']  # Specify the column names
rmse_values = pd.DataFrame(index=range(num_rows), columns=column_names)

# Define the color mapping based on the index of the dataframes
color_mapping = {
    0: '#6D909C'}

plt.rcParams.update({'figure.autolayout': True})
sns.set(rc={'figure.figsize': (20, 10)})
sns.set_style("darkgrid", {"axes.facecolor": color_0_background})

for i, df_test in enumerate(dataframe_list):
    df_test.to_csv(f'C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                   f'02_Working/Excel/02_Qualtrics Data/features_method_{i}.csv', index=False)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_test)
    df_rmse_values = []
    residual_variances = []

    for n in range(1, df_test.shape[1] + 1):
        pca_test = PCA(n_components=n)
        pca_test.fit(df_test)
        transformed_pca = pca_test.transform(df_scaled)
        reconstructed_data = pca_test.inverse_transform(transformed_pca)
        rmse = np.sqrt(mean_squared_error(df_test, reconstructed_data))
        df_rmse_values.append(rmse)  # Append reconstruction error to inner list

        # Calculate residual variance
        explained_variance = pca_test.explained_variance_ratio_.sum()
        residual_variance = 1 - explained_variance
        residual_variances.append(residual_variance)  # Append residual variance to inner list

    color = color_mapping[i]  # Get the color based on the index of the dataframe
    plt.plot(range(1, df_test.shape[1] + 1), df_rmse_values, color=color, marker='o',
             markersize=8, linewidth=2.5, label=f"RMSE for feature set {i + 1}")
    plt.plot(range(1, df_test.shape[1] + 1), residual_variances, color=color, linestyle='--', marker='o',
             label=f"RV for feature set {i + 1}", markersize=8, linewidth=2.5)

plt.xlabel('Number of Components', fontsize=20)
plt.ylabel('Reconstruction Error / Residual Variance', fontsize=20)
plt.title('Reconstruction Error and Residual Variance for Different Numbers of Components', fontsize=20)
plt.legend(fontsize=15, bbox_to_anchor=(1.0005, 1), loc='upper left')
plt.xticks(range(1, 21), fontsize=20)  # Set x-axis ticks from 1 to 20
plt.yticks(fontsize=20)  # Set x-axis ticks from 1 to 20
plt.savefig("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
            "02_Working/Jpeg/Survey/explained_variance_test.png", dpi=300, bbox_inches='tight')
plt.show()

plt.rcParams.update({'figure.autolayout': True})
sns.set(rc={'figure.figsize': (10, 10)})
sns.set_style("whitegrid", {"axes.facecolor": 'white', 'grid.color': color_0_background_d})

component_loadings = []

for i, df_test in enumerate(dataframe_list):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_test)
    pca_test = PCA(n_components=3)
    pca_test.fit(df_test)
    transformed_pca = pca_test.transform(df_scaled)
    reconstructed_data = pca_test.inverse_transform(transformed_pca)
    rmse = np.sqrt(mean_squared_error(df_test, reconstructed_data))
    explained_variance = pca_test.explained_variance_ratio_.sum()
    component_loadings.append(pca_test.components_)  # Append component loadings to the list
    print(f"Component Loadings for Plot {i + 1}:")
    for j, feature_name in enumerate(df_test.columns):
        print(f"Feature '{feature_name}': {pca_test.components_[:, j]}")
    print()
