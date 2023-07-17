from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import SpectralEmbedding
import seaborn as sns
from scipy.cluster import hierarchy
import scipy.cluster.hierarchy as shc
from sklearn.neighbors import NearestCentroid

# Load CSV as dataframe
df = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_preprocess_pre_encoded.csv')

fs = 0
pad = 0
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
group_grad_10 = sns.color_palette(['#618d9a', '#a3b799', '#c8a783', '#ab8788', '#709d9e',
                                  '#7daba2', '#bdb78d', '#b99686', '#d6b680', '#89b8a6'])
group_likert_5 = sns.color_palette(['#c2a8a9', '#cfc2c2', '#dbdbdb', '#adbdc2', '#8fa9b2'])
method = 'ward'

def plot_3d_clusters(data, labels, title, plot_num, center_coords):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = group_grad_10
    for i in range(len(colors)):
        x = data[labels == i, 0]
        y = data[labels == i, 1]
        z = data[labels == i, 2]
        ax.scatter(x, y, z, c=colors[i], alpha=0.85)

    ax.scatter(center_coords[:, 0], center_coords[:, 1], center_coords[:, 2],
               c='red', marker='x', s=200, linewidths=3, alpha=0.8)

    ax.w_xaxis.pane.set_facecolor(color_0_background)
    ax.w_yaxis.pane.set_facecolor(color_0_background)
    ax.w_zaxis.pane.set_facecolor(color_0_background)
    ax.set_xlabel('PC1', fontsize=fs, labelpad=pad)
    ax.set_ylabel('PC2', fontsize=fs, labelpad=pad)
    ax.set_zlabel('PC3', fontsize=fs, labelpad=pad)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.tick_params(axis='z', labelsize=fs)
    ax.text2D(0.5, -0.05, title, transform=ax.transAxes, fontsize=fs, horizontalalignment='center')

    plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                f"02_Working/Jpeg/Clustering/{plot_num}.png", dpi=300, bbox_inches='tight')

    plt.show(block=True)


# Define target columns to be one-hot encoded
target_cols = ['pf_gender', 'pf_education', 'cf_worktype', 'cf_workplacetype',
               'cf_window_orientation', 'cf_window_shade', 'pf_activity']  # replace with names of your target columns

# Perform one-hot encoding on target columns
df = pd.get_dummies(df, columns=target_cols)

df.to_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
          '02_Working/Excel/02_Qualtrics Data/survey_text_data_preprocess_encoded_ss.csv')

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

dataframe_list = [df_method_3]

column_names = ['method_3']

color_mapping = {
    0: '#6D909C'}

plt.rcParams.update({'figure.autolayout': True})
sns.set(rc={'figure.figsize': (20, 10)})
sns.set_style("darkgrid", {"axes.facecolor": color_0_background})

# Define the range of clusters to test
cluster_range = range(2, 11)
silhouette_scores_list = []

for i, df_test in enumerate(dataframe_list):

    #Scaling of data and executing clustering
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_test)
    pca_hierarchical = PCA(n_components=3)
    pca_hierarchical.fit(df_scaled)
    transformed_pca = pca_hierarchical.transform(df_scaled)
    transformed_data = transformed_pca
    cluster_name = 'cluster_pca_' + str(i+1)

    # Define the range of clusters to test
    silhouette_scores = []

    # Loop through the cluster range and compute silhouette scores
    for n_clusters in cluster_range:
        # Fit clustering model and get labels
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels = clustering.fit_predict(transformed_data)

        # Compute silhouette score
        silhouette_avg = silhouette_score(transformed_data, labels)
        silhouette_scores.append(silhouette_avg)

    silhouette_scores_list.append(silhouette_scores)

# Create a line plot of silhouette scores vs number of clusters for each dataset
for i, silhouette_scores in enumerate(silhouette_scores_list):
    cluster_name = 'Cluster PCA ' + str(i + 1)
    color = color_mapping[i]
    plt.plot(cluster_range, silhouette_scores, label=cluster_name, color=color,
             marker='o', markersize=8, linewidth=2.5)

    # Set the y-axis limits
    plt.ylim(0.2, 0.35)

# Create a line plot of silhouette scores vs number of clusters
plt.xlabel('Number of Clusters', fontsize=20)
plt.ylabel('Silhouette Score', fontsize=20)
plt.title('Silhouette Score vs Number of Clusters', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=15, bbox_to_anchor=(1.0005, 1), loc='upper left')
plt.savefig("C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
            "02_Working/Jpeg/Survey/heirarchical_clustering.png", dpi=300, bbox_inches='tight')
plt.show()

plt.rcParams.update({'figure.autolayout': True})
sns.set(rc={'figure.figsize': (10, 10)})
sns.set_style("whitegrid", {"axes.facecolor": 'white', 'grid.color': color_0_background_d})

# for i, df_test in enumerate(dataframe_list):
#
#     #Scaling of data and executing clustering
#     scaler = StandardScaler()
#     df_scaled = scaler.fit_transform(df_test)
#     pca_hierarchical = PCA(n_components=3)
#     pca_hierarchical.fit(df_scaled)
#     transformed_pca = pca_hierarchical.transform(df_scaled)
#     transformed_data = transformed_pca
#
#     #Establishing cluster counts
#     if i == 0 or i == 1:
#         cluster_count = 3
#     elif i == 2 or i == 3:
#         cluster_count = 4
#     else:
#         cluster_count = 4
#
#     #Executing clustering with cluster_count number of clusters
#     hierarchy.set_link_color_palette(['#8fa9b2', '#89b8a6', '#b6b596', '#d6b680', '#c2a8a9'])
#     clustering = AgglomerativeClustering(n_clusters=cluster_count, linkage=method)
#     hiearchy_labels = clustering.fit_predict(transformed_data)
#     clf = NearestCentroid()
#     clf.fit(transformed_data, hiearchy_labels)
#     hierarchy_centroids = clf.centroids_
#
#     # dendogram = shc.dendrogram(shc.linkage(transformed_data, method=method))
#     #
#     # plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#     #             f"02_Working/Jpeg/self_supervised/dendrogram_{i}.png", dpi=300, bbox_inches='tight')

    # plt.show()
    #
    # hierarchy_labels = clustering.fit_predict(transformed_data)
    #
    # plotlabels = "Scatter for feature set " + str(i) + " with cluster count " + str(cluster_count)
    #
    # plot_number = "Agg_" + str(i)
    #
    # plot_3d_clusters(transformed_data, hierarchy_labels, plotlabels, plot_number, hierarchy_centroids)

# for i in range (2,11):
#     df_clustering = df_method_3
#     scaler = StandardScaler()
#     df_scaled = scaler.fit_transform(df_method_3)
#     pca_hierarchical = PCA(n_components=3)
#     pca_hierarchical.fit(df_scaled)
#     transformed_pca = pca_hierarchical.transform(df_scaled)
#     transformed_data = transformed_pca
#
#     cluster_count = i
#
#     # hierarchy.set_link_color_palette(['#8fa9b2', '#89b8a6', '#b6b596', '#d6b680', '#c2a8a9'])
#     clustering = AgglomerativeClustering(n_clusters=cluster_count, linkage=method)
#     hiearchy_labels = clustering.fit_predict(transformed_data)
#     clf = NearestCentroid()
#     clf.fit(transformed_data, hiearchy_labels)
#     hierarchy_centroids = clf.centroids_
#
#     plt.show()
#
#     hierarchy_labels = clustering.fit_predict(transformed_data)
#
#     plotlabels = "Range Scatter for feature set " + str(i) + " with cluster count " + str(cluster_count)
#
#     plot_number = "range_agg_" + str(i)
#
#     plot_3d_clusters(transformed_data, hierarchy_labels, plotlabels, plot_number, hierarchy_centroids)