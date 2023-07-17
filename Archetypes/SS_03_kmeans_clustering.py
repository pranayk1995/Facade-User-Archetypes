from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.neighbors import NearestCentroid
from sklearn.manifold import SpectralEmbedding
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS


# Load CSV as dataframe
df = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_preprocess_pre_encoded.csv')

fs = 0
pad = 0
color_0_background = '#f5f5f5'
color_1_background = '#adadad'
color_1_base = '#6D909C'
method = 'ward'
group_grad_4 = ['#8fa9b2', '#89b8a6', '#d6b680', '#AB8788']
cmap_custom = ListedColormap(group_grad_4)


def scatter_centers(transformed_data, labels, plot_num, centerz):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c=labels, cmap=cmap_custom, alpha=0.8)
    ax.scatter(centerz[:, 0], centerz[:, 1], centerz[:, 2], c='red', marker='x', s=200, linewidths=3, alpha=0.8)
    ax.w_xaxis.pane.set_facecolor(color_0_background)
    ax.w_yaxis.pane.set_facecolor(color_0_background)
    ax.w_zaxis.pane.set_facecolor(color_0_background)
    ax.set_xlabel('', fontsize=fs, labelpad=pad)
    ax.set_ylabel('', fontsize=fs, labelpad=pad)
    ax.set_zlabel('', fontsize=fs, labelpad=pad)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.tick_params(axis='z', labelsize=fs)
    plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                f"02_Working/Jpeg/self_supervised/{plot_num}.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show(block=True)


def plot_3d_clusters(data, labels, title, plot_num, center_coords):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = group_grad_4
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
                f"02_Working/Jpeg/self_supervised/{plot_num}.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show(block=True)


def scatter(transformed_datas, label, plot_num):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed_datas[:, 0], transformed_datas[:, 1], transformed_datas[:, 2], c=color_1_base)
    ax.w_xaxis.pane.set_facecolor(color_0_background)
    ax.w_yaxis.pane.set_facecolor(color_0_background)
    ax.w_zaxis.pane.set_facecolor(color_0_background)
    ax.set_xlabel('', fontsize=fs, labelpad=pad)
    ax.set_ylabel('', fontsize=fs, labelpad=pad)
    ax.set_zlabel('', fontsize=fs, labelpad=pad)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.tick_params(axis='z', labelsize=fs)
    ax.text2D(0.5, -0.05, label, transform=ax.transAxes, fontsize=fs, horizontalalignment='center')
    plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                f"02_Working/Jpeg/self_supervised/{plot_num}.png", dpi=300, bbox_inches='tight')
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

df_pref = df[method_3]

# # Method 1 transformation
# df_pref['weight_energy'] = df_pref[['psy_e_consciously_sustainable',
#                                     'psy_e_change_sustainable',
#                                     'psy_e_compromise_comfort_sustainabile']].mean(axis=1)
# df_pref['weight_temperature'] = df_pref[['ef_importance_temperature', 'ef_productivity_temperature']].mean(axis=1)
# df_pref['weight_daylight'] = df_pref[['ef_importance_daylight', 'ef_productivity_daylight']].mean(axis=1)
# df_pref['weight_glare'] = df_pref[['ef_importance_glare', 'ef_productivity_glare']].mean(axis=1)
# df_pref['weight_view'] = df_pref[['ef_importance_view', 'ef_productivity_view']].mean(axis=1)
# df_pref['weight_interiors'] = df_pref[['psy_c_importance_interior_features']]
# df_pref = df_pref.drop(['psy_e_consciously_sustainable', 'psy_e_change_sustainable',
#                         'psy_e_compromise_comfort_sustainabile', 'ef_importance_temperature',
#                         'ef_productivity_temperature', 'ef_importance_daylight',
#                         'ef_productivity_daylight', 'ef_importance_glare', 'ef_productivity_glare',
#                         'ef_importance_view', 'ef_productivity_view', 'psy_c_importance_interior_features'], axis=1)

# # Method 2 transformation
# df_pref['weight_energy'] = df_pref[['psy_e_consciously_sustainable',
#                                     'psy_e_change_sustainable',
#                                     'psy_e_compromise_comfort_sustainabile']].mean(axis=1)
# df_pref['weight_temperature'] = df_pref[['ef_importance_temperature',
#                                          'ef_productivity_temperature',
#                                          'sf_use_thermal']].mean(axis=1)
# df_pref['weight_daylight'] = df_pref[['ef_importance_daylight',
#                                       'ef_productivity_daylight',
#                                       'sf_use_adequate_lighting']].mean(axis=1)
# df_pref['weight_glare'] = df_pref[['ef_importance_glare',
#                                    'ef_productivity_glare',
#                                    'sf_use_glare_mitigation']].mean(axis=1)
# df_pref['weight_view'] = df_pref[['ef_importance_view',
#                                   'ef_productivity_view',
#                                   'sf_use_view_outside']].mean(axis=1)
# df_pref['weight_interiors'] = df_pref[['psy_c_importance_interior_features',
#                                        'sf_use_spatial_aesthetic']].mean(axis=1)
# df_pref = df_pref.drop(['psy_e_consciously_sustainable', 'psy_e_change_sustainable',
#                         'psy_e_compromise_comfort_sustainabile', 'ef_importance_temperature',
#                         'ef_productivity_temperature', 'ef_importance_daylight',
#                         'ef_productivity_daylight', 'ef_importance_glare', 'ef_productivity_glare',
#                         'ef_importance_view', 'ef_productivity_view', 'psy_c_importance_interior_features',
#                         'sf_use_thermal', 'sf_use_adequate_lighting', 'sf_use_glare_mitigation',
#                         'sf_use_view_outside', 'sf_use_spatial_aesthetic'], axis=1)

#Method 3 transformation
df_pref['weight_energy'] = df_pref[['psy_e_consciously_sustainable',
                                    'psy_e_change_sustainable',
                                    'psy_e_compromise_comfort_sustainabile']].mean(axis=1)
df_pref['weight_temperature'] = df_pref[['ef_importance_temperature', 'ef_productivity_temperature']].mean(axis=1)
df_pref['weight_daylight'] = df_pref[['ef_importance_daylight', 'ef_productivity_daylight']].mean(axis=1)
df_pref['weight_glare'] = df_pref[['ef_importance_glare', 'ef_productivity_glare']].mean(axis=1)
df_pref['weight_view'] = df_pref[['ef_importance_view', 'ef_productivity_view']].mean(axis=1)
df_pref['weight_interiors'] = df_pref[['psy_c_importance_interior_features']]
df_pref['rb_05'] = df_pref[['sr_view1_rb_05', 'sr_view2_rb_05', 'sr_view3_rb_05']].mean(axis=1)
df_pref['rb_10'] = df_pref[['sr_view1_rb_10', 'sr_view2_rb_10', 'sr_view3_rb_10']].mean(axis=1)
df_pref['vb_25'] = df_pref[['sr_view1_vb_25', 'sr_view2_vb_25', 'sr_view3_vb_25']].mean(axis=1)
df_pref['vb_50'] = df_pref[['sr_view1_vb_50', 'sr_view2_vb_50', 'sr_view3_vb_50']].mean(axis=1)
df_pref = df_pref.drop(['psy_e_consciously_sustainable', 'psy_e_change_sustainable',
                        'psy_e_compromise_comfort_sustainabile', 'ef_importance_temperature',
                        'ef_productivity_temperature', 'ef_importance_daylight',
                        'ef_productivity_daylight', 'ef_importance_glare', 'ef_productivity_glare',
                        'ef_importance_view', 'ef_productivity_view', 'psy_c_importance_interior_features',
                        'sr_view1_rb_05', 'sr_view2_rb_05', 'sr_view3_rb_05',
                        'sr_view1_rb_10', 'sr_view2_rb_10', 'sr_view3_rb_10',
                        'sr_view1_vb_25', 'sr_view2_vb_25', 'sr_view3_vb_25',
                        'sr_view1_vb_50', 'sr_view2_vb_50', 'sr_view3_vb_50'], axis=1)

# #Method 4 transformation
# df_pref['weight_energy'] = df_pref[['psy_e_consciously_sustainable',
#                                     'psy_e_change_sustainable',
#                                     'psy_e_compromise_comfort_sustainabile']].mean(axis=1)
# df_pref['weight_temperature'] = df_pref[['ef_importance_temperature',
#                                          'ef_productivity_temperature',
#                                          'sf_use_thermal']].mean(axis=1)
# df_pref['weight_daylight'] = df_pref[['ef_importance_daylight',
#                                       'ef_productivity_daylight',
#                                       'sf_use_adequate_lighting']].mean(axis=1)
# df_pref['weight_glare'] = df_pref[['ef_importance_glare',
#                                    'ef_productivity_glare',
#                                    'sf_use_glare_mitigation']].mean(axis=1)
# df_pref['weight_view'] = df_pref[['ef_importance_view',
#                                   'ef_productivity_view',
#                                   'sf_use_view_outside']].mean(axis=1)
# df_pref['rb_05'] = df_pref[['sr_view1_rb_05', 'sr_view2_rb_05', 'sr_view3_rb_05']].mean(axis=1)
# df_pref['rb_10'] = df_pref[['sr_view1_rb_10', 'sr_view2_rb_10', 'sr_view3_rb_10']].mean(axis=1)
# df_pref['vb_25'] = df_pref[['sr_view1_vb_25', 'sr_view2_vb_25', 'sr_view3_vb_25']].mean(axis=1)
# df_pref['vb_50'] = df_pref[['sr_view1_vb_50', 'sr_view2_vb_50', 'sr_view3_vb_50']].mean(axis=1)
# df_pref = df_pref.drop(['psy_e_consciously_sustainable', 'psy_e_change_sustainable',
#                         'psy_e_compromise_comfort_sustainabile', 'ef_importance_temperature',
#                         'ef_productivity_temperature', 'ef_importance_daylight',
#                         'ef_productivity_daylight', 'ef_importance_glare', 'ef_productivity_glare',
#                         'ef_importance_view', 'ef_productivity_view', 'psy_c_importance_interior_features',
#                         'sr_view1_rb_05', 'sr_view2_rb_05', 'sr_view3_rb_05',
#                         'sr_view1_rb_10', 'sr_view2_rb_10', 'sr_view3_rb_10',
#                         'sr_view1_vb_25', 'sr_view2_vb_25', 'sr_view3_vb_25',
#                         'sr_view1_vb_50', 'sr_view2_vb_50', 'sr_view3_vb_50',
#                         'sf_use_thermal', 'sf_use_adequate_lighting', 'sf_use_glare_mitigation',
#                         'sf_use_view_outside', 'sf_use_spatial_aesthetic'], axis=1)

#Scaling the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pref)
final_dataframe = df_scaled

#PCA
pca = PCA(n_components=3)
pca.fit(final_dataframe)
transformed_pca = pca.transform(final_dataframe)

transformed_data = transformed_pca

#Plotting the resulting reduced dimensions on a 3d scatterplot
scatter(transformed_data, 'PCA Plot', 'kmeans_1')

heirarchy = AgglomerativeClustering(n_clusters=4, linkage='ward')
heirarchy_labels = heirarchy.fit_predict(transformed_data)

silhouette_avg = silhouette_score(transformed_data, heirarchy_labels)
print("Silhouette Score for Agglomerative Clustering: ", silhouette_avg)

# Fit NearestCentroid model and get centroids
clf = NearestCentroid()
clf.fit(transformed_data, heirarchy_labels)
heirarchy_centroids = clf.centroids_
print("Centroids of Agglomerative Clusters:\n", heirarchy_centroids)

# plot_3d_clusters(transformed_data, heirarchy_labels, 'Agglomerative clustering', 'kmeans_2', heirarchy_centroids)

# Use clusters from hierarchical clustering as initial cluster centers for KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=1)
kmeans.fit(transformed_data)
kmeans_labels = kmeans.labels_

# Silhouette average of the kmeans clustering
kmeans_silhouette_avg = silhouette_score(transformed_data, kmeans_labels)
print("Silhouette Score for KMeans Clustering: ", kmeans_silhouette_avg)

# Get centroids of the k-means clusters
kmeans_centroids = kmeans.cluster_centers_

plot_3d_clusters(transformed_data, kmeans_labels, 'Kmeans clustering', 'kmeans_3', kmeans_centroids)

# Add cluster labels to the original DataFrame
df['archetype'] = kmeans.labels_
kmean_label = kmeans.labels_

# Inverse transform the cluster centers to get the original feature values
decoded_centers = pca.inverse_transform(kmeans_centroids)
unscaled = scaler.inverse_transform(decoded_centers)
df_unscaled = pd.DataFrame(unscaled)

print(df_unscaled)

# Define the new column names
new_columns = ['sr_view4_rb_05', 'sr_view4_rb_10', 'sr_view4_vb_25',
               'sr_view4_vb_50', 'sr_view1_rb_05_L', 'sr_view1_rb_05_M',
               'sr_view1_rb_05_D', 'sr_view1_vb_25_L', 'sr_view1_vb_25_D',
               'rb_05', 'rb_10', 'vb_25', 'vb_50',
               'weight_temperature', 'weight_energy', 'weight_daylight', 'weight_glare',
               'weight_view', 'weight_interiors']

# Rename the columns of df_unscaled
df_unscaled.columns = new_columns

# Print the updated DataFrame
print(df_unscaled)

# Calculate WCSS
wcss = kmeans.inertia_
print("Within-Cluster Sum of Squares (WCSS):", wcss)

#Save original dataframe with cluster index as labels
df.to_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
          '02_Working/Excel/02_Qualtrics Data/survey_text_data_clustered_ss.csv')

# Save dataframe with centroid values (not necessarily reliable due to a low explained variance)
df_unscaled.to_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                   '02_Working/Excel/02_Qualtrics Data/Archetype_ss.csv')

# Save the dataframe used for clustering (non-scaled)
df_pref.to_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
               '02_Working/Excel/02_Qualtrics Data/survey_text_data_preferences_ss.csv')