import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer

# Load CSV as dataframe
df = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_clustered_ss.csv', index_col=0)

df_new = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_preprocess_pre_encoded_newdata.csv')

df_central = pd.concat([df, df_new], ignore_index=True)

print(df_central)

# ----------------------------------------------------------------------------------------------------------------------


def plot_3d_clusters(data, labels, title, plot_num):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = group_grad_4
    for label in labels:
        i = label if label != -1 else 4  # Transform -1 to 4
        x = data[labels == label, 0]
        y = data[labels == label, 1]
        z = data[labels == label, 2]
        ax.scatter(x, y, z, c=colors[i], alpha=0.85)

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
    plt.show(block=True)


def plot_3d_clusters_4cols(data, labels, title, plot_num):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = group_grad_4
    for label in labels:
        if label != -1:  # Exclude label -1
            i = label  # Keep the original label
            x = data[labels == label, 0]
            y = data[labels == label, 1]
            z = data[labels == label, 2]
            ax.scatter(x, y, z, c=colors[i], alpha=0.85)

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
    plt.show(block=True)


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


# ----------------------------------------------------------------------------------------------------------------------

fs = 0
pad = 0
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
group_grad_4 = ['#8fa9b2', '#89b8a6', '#d6b680', '#AB8788', '#000000']

# ----------------------------------------------------------------------------------------------------------------------

method_3 = ['psy_e_consciously_sustainable', 'psy_e_change_sustainable', 'psy_e_compromise_comfort_sustainabile',
            'ef_importance_temperature', 'ef_importance_view',  'ef_importance_daylight',
            'ef_importance_glare', 'ef_productivity_temperature', 'ef_productivity_view',
            'ef_productivity_daylight', 'ef_productivity_glare', 'psy_c_importance_interior_features',
            'sr_view1_rb_05', 'sr_view1_rb_10', 'sr_view1_vb_25', 'sr_view1_vb_50', 'sr_view2_rb_05',
            'sr_view2_rb_10', 'sr_view2_vb_25', 'sr_view2_vb_50', 'sr_view3_rb_05', 'sr_view3_rb_10',
            'sr_view3_vb_25', 'sr_view3_vb_50', 'sr_view4_rb_05', 'sr_view4_rb_10', 'sr_view4_vb_25',
            'sr_view4_vb_50', 'sr_view1_rb_05_L', 'sr_view1_rb_05_M', 'sr_view1_rb_05_D', 'sr_view1_vb_25_L',
            'sr_view1_vb_25_D']

df_pref = df_central[method_3]

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

#Scaling the data
scaler = StandardScaler()
df_scaled_shortlist = scaler.fit_transform(df_pref)
final_dataframe_shortlist = df_scaled_shortlist
pca = PCA(n_components=3)
pca.fit(final_dataframe_shortlist)
transformed_pca = pca.transform(final_dataframe_shortlist)
transformed_data = transformed_pca

#Plotting the resulting reduced dimensions on a 3d scatterplot
scatter(transformed_data, 'PCA Plot', 'Supervised_01')
archetype_labels = df_central['archetype']
plot_3d_clusters_4cols(transformed_data, archetype_labels, 'Initial layout on K-means', 'Supervised_02')
plot_3d_clusters(transformed_data, archetype_labels, 'Initial layout on K-means', 'Supervised_03')

print(archetype_labels)

# ----------------------------------------------------------------------------------------------------------------------

# Splitting the data into X and y
X = df_central.drop('archetype', axis=1)
y = df_central['archetype']

# Label encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Perform one-hot encoding on the categorical columns
categorical_cols = ['pf_gender', 'pf_education', 'pf_activity', 'cf_worktype',
                    'cf_window_orientation', 'cf_workplacetype', 'cf_window_shade']
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Create separate DataFrames for X and y for the first 171 samples
X_first = X_encoded[:171].copy()
y_first = pd.DataFrame(y_encoded[:171], columns=['target']).copy()

# Create separate DataFrames for X and y for the next 18 samples
X_next = X_encoded[171:189].copy()
y_next = pd.DataFrame(y_encoded[171:189], columns=['target']).copy()



# ----------------------------------------------------------------------------------------------------------------------

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_first, y_first, test_size=0.2, random_state=44)

# Perform feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_next_scaled = scaler.transform(X_next)

# Create an instance of SimpleImputer with mean strategy
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_next_scaled)
X_next_scaled_imputed = imputer.transform(X_next_scaled)

# Number of iterations
num_iterations = 5

models = []
weights = []

# Apply random oversampling
ros = RandomOverSampler(random_state=None)
X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train_scaled, y_train)

# Apply SMOTE
smote = SMOTE(random_state=None)
X_train_smote, y_train_smote = smote.fit_resample(X_train_oversampled, y_train_oversampled)

for i in range(num_iterations):

    # Train a new logistic regression model using the best hyperparameters
    best_logreg = LogisticRegression(C=1.0, class_weight='balanced', dual=False, max_iter=1000, penalty='l1', solver='saga', tol=1e-05)
    best_logreg.fit(X_train_smote, y_train_oversampled.values.ravel())

    # Evaluate the model on the test set
    accuracy_select = best_logreg.score(X_test_scaled, y_test)

    # Append the model and its accuracy to the lists
    models.append(best_logreg)
    weights.append(accuracy_select)

print("Wait 1")

# Normalize the weights
weights_sum = sum(weights)
weights = [w / weights_sum for w in weights]

print("Wait 2")

# Create a VotingClassifier with the individual models and their weights
ensemble_model = VotingClassifier(estimators=[('model'+str(i+1), model) for i, model in enumerate(models)],
                                  voting='soft', weights=weights)

print("Wait 3")

# Fit the ensemble model using the entire training set
ensemble_model.fit(X_train_smote, y_train_smote.values.ravel())

# Evaluate the ensemble model on the test set
ensemble_accuracy = ensemble_model.score(X_test_scaled, y_test)

print("Ensemble Accuracy:", ensemble_accuracy)

# ----------------------------------------------------------------------------------------------------------------------

# Get the coefficients of the logistic regression model
coefficients = best_logreg.coef_

# Retrieve the feature names from the original feature matrix X_train_scaled
feature_names = X_train.columns

# Create a dictionary to store feature importance
feature_importance = {}

# Assign the feature importance values to each feature
for feature, importance in zip(feature_names, coefficients[0]):
    feature_importance[feature] = abs(importance)

# Sort the feature importance dictionary by importance values in descending order
sorted_feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}

# Print the feature importance
for feature, importance in sorted_feature_importance.items():
    print(f"{feature}: {importance}")

# ----------------------------------------------------------------------------------------------------------------------

y_new_pred = ensemble_model.predict(X_next_scaled_imputed)
print(y_new_pred)
print(type(y_new_pred))
y_new_pred_decoded = label_encoder.inverse_transform(y_new_pred)
print(y_new_pred_decoded)

y_new_pred_df = pd.DataFrame(y_new_pred, columns=['new_label'])

print(y_new_pred_df)
print(archetype_labels)

# ----------------------------------------------------------------------------------------------------------------------

#Adding y_new_pred to Archetype labels

archetype_labels_trimmed = archetype_labels[:-18]  # Remove the last 18 values

combined_labels = pd.concat([archetype_labels_trimmed, y_new_pred_df['new_label']], ignore_index=True)

plot_3d_clusters(transformed_data, combined_labels, 'Initial layout on K-means', 'Supervised_04')

# ----------------------------------------------------------------------------------------------------------------------