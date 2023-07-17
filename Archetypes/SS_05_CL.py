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
from sklearn.neighbors import KNeighborsClassifier


# Load CSV as dataframe
df = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_clustered_ss.csv')


print(df)

df_pref = df[['ef_importance_temperature', 'ef_importance_view', 'ef_importance_daylight', 'ef_importance_glare',
                'sr_view1_rb_05', 'sr_view1_rb_10',
                'sr_view1_vb_25', 'sr_view1_vb_50', 'sr_view2_rb_05',
                'sr_view2_rb_10', 'sr_view2_vb_25', 'sr_view2_vb_50',
                'sr_view3_rb_05', 'sr_view3_rb_10', 'sr_view3_vb_25',
                'sr_view3_vb_50', 'sr_view4_rb_05', 'sr_view4_rb_10',
                'sr_view4_vb_25', 'sr_view4_vb_50', 'sr_view1_rb_05_L',
                'sr_view1_rb_05_M', 'sr_view1_rb_05_D', 'sr_view1_vb_25_L',
                'sr_view1_vb_25_D', 'psy_e_consciously_sustainable', 'psy_e_change_sustainable',
                'psy_c_importance_interior_features', 'ef_productivity_temperature', 'ef_productivity_view',
                'ef_productivity_daylight', 'ef_productivity_glare', 'psy_e_compromise_comfort_sustainabile',
                'archetype']]

# Transoform the dataframe
df_pref['rb_05'] = df_pref[['sr_view1_rb_05', 'sr_view2_rb_05', 'sr_view3_rb_05']].mean(axis=1)
df_pref['rb_10'] = df_pref[['sr_view1_rb_10', 'sr_view2_rb_10', 'sr_view3_rb_10']].mean(axis=1)
df_pref['vb_25'] = df_pref[['sr_view1_vb_25', 'sr_view2_vb_25', 'sr_view3_vb_25']].mean(axis=1)
df_pref['vb_50'] = df_pref[['sr_view1_vb_50', 'sr_view2_vb_50', 'sr_view3_vb_50']].mean(axis=1)
df_pref['weight_temperature'] = df_pref[['ef_importance_temperature', 'ef_productivity_temperature']].mean(axis=1)
df_pref['weight_energy'] = df_pref[['psy_e_consciously_sustainable',
                                    'psy_e_change_sustainable',
                                    'psy_e_compromise_comfort_sustainabile']].mean(axis=1)
df_pref['weight_daylight'] = df_pref[['ef_importance_daylight', 'ef_productivity_daylight']].mean(axis=1)
df_pref['weight_glare'] = df_pref[['ef_importance_glare', 'ef_productivity_glare']].mean(axis=1)
df_pref['weight_view'] = df_pref[['ef_importance_view', 'ef_productivity_view']].mean(axis=1)
df_pref['weight_interiors'] = df_pref[['psy_c_importance_interior_features']]
df_pref = df_pref.drop(['sr_view1_rb_05', 'sr_view2_rb_05', 'sr_view3_rb_05',
                        'sr_view1_rb_10', 'sr_view2_rb_10', 'sr_view3_rb_10',
                        'sr_view1_vb_25', 'sr_view2_vb_25', 'sr_view3_vb_25',
                        'sr_view1_vb_50', 'sr_view2_vb_50', 'sr_view3_vb_50',
                        'ef_importance_daylight', 'ef_productivity_daylight', 'ef_importance_glare',
                        'ef_productivity_glare', 'ef_importance_view', 'ef_productivity_view',
                        'psy_c_importance_interior_features', 'psy_e_consciously_sustainable',
                        'psy_e_change_sustainable', 'psy_e_compromise_comfort_sustainabile',
                        'ef_importance_temperature', 'ef_productivity_temperature'], axis=1)

fs = 12
pad = 12
color_0_background = '#f5f5f5'
color_1_base = '#6D909C'
method = 'ward'
group_grad_4 = ['#8fa9b2', '#89b8a6', '#d6b680', '#AB8788']
cmap_custom = ListedColormap(group_grad_4)

# Splitting the data into X and y
X = df_pref.drop('archetype', axis=1)
y = df_pref['archetype']

# Label encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# # Perform one-hot encoding on the categorical features in X
# categorical_cols = ['pf_gender', 'pf_education', 'pf_activity', 'cf_worktype',
#                     'cf_window_orientation', 'cf_workplacetype', 'cf_window_shade']
#
# # # Concatenate the encoded categorical features with the remaining numerical features
# X_categorical = pd.get_dummies(X[categorical_cols])
# X_encoded = pd.concat([X.drop(categorical_cols, axis=1), X_categorical], axis=1)

X_encoded = X

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Perform feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

spec_test = SpectralEmbedding(n_components=3)
spec_test.fit(df_scaled)
transformed_spec = spec_test.fit_transform(df_scaled)

# # Initialize and train the SVC model
# svc = SVC()
# svc.fit(X_train_scaled, y_train)
#
# # Make predictions on the test set
# y_pred = svc.predict(X_test_scaled)
# # Decode the predicted labels
# y_pred_decoded = label_encoder.inverse_transform(y_pred)

#Fitting and evaluating the model
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Evaluate the model
classification_report = classification_report(y_test, y_pred_decoded)
print(classification_report)
