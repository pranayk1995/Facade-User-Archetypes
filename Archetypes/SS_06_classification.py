from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.colors as mcolors
from imblearn.over_sampling import RandomOverSampler, SMOTE
import numpy as np
from sklearn.impute import SimpleImputer

def scatter(transformed_datas, label, plot_num):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed_datas[:, 0], transformed_datas[:, 1], transformed_datas[:, 2], c=color_1_base)
    ax.w_xaxis.pane.set_facecolor(color_0_background)
    ax.w_yaxis.pane.set_facecolor(color_0_background)
    ax.w_zaxis.pane.set_facecolor(color_0_background)
    ax.set_xlabel('PC1', fontsize=fs, labelpad=pad)
    ax.set_ylabel('PC2', fontsize=fs, labelpad=pad)
    ax.set_zlabel('PC3', fontsize=fs, labelpad=pad)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.tick_params(axis='z', labelsize=fs)
    ax.text2D(0.5, -0.05, label, transform=ax.transAxes, fontsize=fs, horizontalalignment='center')
    plt.savefig(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                f"02_Working/Jpeg/self_supervised/{plot_num}.png", dpi=300, bbox_inches='tight')
    plt.show(block=True)

# Load CSV as dataframe
df = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_clustered_ss.csv', index_col=0)

df_new = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_preprocess_pre_encoded_newdata.csv')

df_central = pd.concat([df, df_new], ignore_index=True)

print(df_central)

# # Get the column names of df and df_new
# columns_df = set(df.columns)
# columns_df_new = set(df_new.columns)
#
# # Check if the columns are the same
# if columns_df == columns_df_new:
#     print("Both DataFrames have the same columns.")
# else:
#     # Find the columns that are present in one DataFrame but not in the other
#     missing_columns_df = columns_df - columns_df_new
#     missing_columns_df_new = columns_df_new - columns_df
#
#     print("Columns present in df but not in df_new:")
#     print(missing_columns_df)
#     print()
#
#     print("Columns present in df_new but not in df:")
#     print(missing_columns_df_new)

fs = 12
pad = 12
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

method_3 = ['psy_e_consciously_sustainable', 'psy_e_change_sustainable', 'psy_e_compromise_comfort_sustainabile',
            'ef_importance_temperature', 'ef_importance_view',  'ef_importance_daylight',
            'ef_importance_glare', 'ef_productivity_temperature', 'ef_productivity_view',
            'ef_productivity_daylight', 'ef_productivity_glare', 'psy_c_importance_interior_features',
            'sr_view1_rb_05', 'sr_view1_rb_10', 'sr_view1_vb_25', 'sr_view1_vb_50', 'sr_view2_rb_05',
            'sr_view2_rb_10', 'sr_view2_vb_25', 'sr_view2_vb_50', 'sr_view3_rb_05', 'sr_view3_rb_10',
            'sr_view3_vb_25', 'sr_view3_vb_50', 'sr_view4_rb_05', 'sr_view4_rb_10', 'sr_view4_vb_25',
            'sr_view4_vb_50', 'sr_view1_rb_05_L', 'sr_view1_rb_05_M', 'sr_view1_rb_05_D', 'sr_view1_vb_25_L',
            'sr_view1_vb_25_D']

df_pref = df[method_3]

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

#PCA
#Scaling the data
scaler = StandardScaler()
df_scaled_shortlist = scaler.fit_transform(df_pref)
final_dataframe_shortlist = df_scaled_shortlist
pca = PCA(n_components=3)
pca.fit(final_dataframe_shortlist)
transformed_pca = pca.transform(final_dataframe_shortlist)
transformed_data = transformed_pca
#Plotting the resulting reduced dimensions on a 3d scatterplot
scatter(transformed_data, 'PCA Plot', 'kmeans_1')

# Splitting the data into X and y
X = df_central.drop('archetype', axis=1)
y = df_central['archetype']

print(X)
print(y)

# Label encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Perform one-hot encoding on the categorical features in X
categorical_cols = ['pf_gender', 'pf_education', 'pf_activity', 'cf_worktype',
                    'cf_window_orientation', 'cf_workplacetype', 'cf_window_shade']

# Concatenate the encoded categorical features with the remaining numerical features
X_categorical = pd.get_dummies(X[categorical_cols])
X_encoded = pd.concat([X.drop(categorical_cols, axis=1), X_categorical], axis=1)

# Create separate DataFrames for X and y for the first 171 samples
X_first = X_encoded[:171].copy()
y_first = pd.DataFrame(y_encoded[:171], columns=['target']).copy()

# Create separate DataFrames for X and y for the next 18 samples
X_next = X_encoded[171:189].copy()
y_next = pd.DataFrame(y_encoded[171:189], columns=['target']).copy()

X_next.to_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
              '02_Working/Excel/02_Qualtrics Data/testnan.csv')

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_first, y_first, test_size=0.2, random_state=None)

# Perform feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_next_scaled = scaler.transform(X_next)

# Find the NaN values in X_next_scaled
nan_values = np.isnan(X_next_scaled)
if np.any(nan_values):
    # Find the indices of the NaN values
    nan_indices = np.argwhere(nan_values)
    print(nan_indices)
else:
    print("No NaN values found in X_next_scaled.")
# Create an instance of SimpleImputer with mean strategy
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_next_scaled)
X_next_scaled_imputed = imputer.transform(X_next_scaled)

# Apply random oversampling
ros = RandomOverSampler(random_state=None)
X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train_scaled, y_train)

# Apply SMOTE
smote = SMOTE(random_state=None)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

#
# # #Setting a hyperparameter loop
# # kernels = ['linear', 'rbf', 'poly']
# # gammas = [0.1, 1, 10, 100]
# # cs = [0.1, 1, 10, 100, 1000]
# # degrees = [0, 1, 2, 3, 4, 5, 6]
# #
# # best_score = 0  # Initialize the best score
# # best_params = {}  # Store the best hyperparameters
# #
# # for kernel in kernels:
# #     for gamma in gammas:
# #         for c in cs:
# #             for degree in degrees:
# #                 # Initialize and train the SVC model
# #                 svc = SVC(kernel=kernel, gamma=gamma, C=c, degree=degree)
# #                 svc.fit(X_train_smote, y_train_smote)
# #
# #                 # Make predictions on the test set
# #                 y_pred = svc.predict(X_test_scaled)
# #
# #                 # Evaluate the model
# #                 score = classification_report(y_test, y_pred, output_dict=True)['accuracy']
# #
# #                 # Check if the current combination of hyperparameters has a higher score
# #                 if score > best_score:
# #                     best_score = score
# #                     best_params = {'kernel': kernel, 'gamma': gamma, 'C': c, 'degree': degree}
# #
# # # Print the best hyperparameters and corresponding score
# # print("Best Hyperparameters:", best_params)
# # print("Best Score:", best_score)
#
#
# #Setting a hyperparameter loop
# kernels = ['linear', 'rbf', 'poly']
# gammas = [0.1, 1, 10, 100]
# cs = [0.1, 1, 10, 100, 1000]
# degrees = [0, 1, 2, 3, 4, 5, 6]

# Initialize and train the SVC model
svc = SVC(kernel='linear', gamma=1, C=0.1, degree=0)
svc.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
y_pred = svc.predict(X_test_scaled)
y_pred_decoded = label_encoder.inverse_transform(y_pred)
classification_report = classification_report(y_test, y_pred_decoded)
print(classification_report)

# Make predictions on the new data
y_new_pred = svc.predict(X_next_scaled_imputed)
print(y_new_pred)
y_new_pred_decoded = label_encoder.inverse_transform(y_new_pred)

#
# # Get the coefficients
# coefficients = svc.coef_
#
# # Get the feature names from X_encoded
# feature_names = X_encoded.columns
#
# # Pair the coefficients with the corresponding feature names
# feature_importance = dict(zip(feature_names, abs(coefficients[0])))
#
# # Sort the feature importance in descending order
# sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
#
# # Print the top k most important features
# k = 20  # Set the number of top features to display
# for feature, importance in sorted_importance[:k]:
#     print(f"Feature: {feature}, Importance: {importance}")