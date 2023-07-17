import pandas as pd
import numpy as np
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

# Load CSV as dataframe
df = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_clustered_ss.csv', index_col=0)

df_new = pd.read_csv('C:/Users/prana/OneDrive - Delft University of Technology/Thesis/'
                 '02_Working/Excel/02_Qualtrics Data/survey_text_data_preprocess_pre_encoded_newdata.csv')

df_central = df

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



# ----------------------------------------------------------------------------------------------------------------------

# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_first, y_first, test_size=0.2, random_state=None)
#
# # Perform feature scaling using StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# models = {
#     'Random Forest': RandomForestClassifier(),
#     'Logistic Regression': LogisticRegression(),
#     'SVC': SVC(),
#     'Gaussian Naive Bayes': GaussianNB(),
#     'K-Nearest Neighbors': KNeighborsClassifier(),
#     'Decision Tree': DecisionTreeClassifier(),
#     'AdaBoost': AdaBoostClassifier(),
#     'Gradient Boosting': GradientBoostingClassifier(),
#     'Extra Trees': ExtraTreesClassifier(),
#     'Multi-layer Perceptron': MLPClassifier()
# }
#
# # Define the number of iterations
# num_iterations = 100
#
# results = []
# for model_name, model in models.items():
#     # Initialize lists to store performance metrics for each iteration
#     precision_scores = []
#     recall_scores = []
#     f1_scores = []
#     support_scores = []
#
#     for _ in range(num_iterations):
#         # Initialize and train the model
#         model.fit(X_train_scaled, y_train)
#
#         # Make predictions on the test set
#         y_pred = model.predict(X_test_scaled)
#         y_pred_decoded = label_encoder.inverse_transform(y_pred)
#
#         # Generate classification report
#         report = classification_report(y_test, y_pred_decoded, output_dict=True)
#         weighted_avg = report['weighted avg']
#
#         # Store the performance metrics for each iteration
#         precision_scores.append(weighted_avg['precision'])
#         recall_scores.append(weighted_avg['recall'])
#         f1_scores.append(weighted_avg['f1-score'])
#         support_scores.append(weighted_avg['support'])
#
#     # Calculate the average performance metrics across iterations
#     avg_precision = np.mean(precision_scores)
#     avg_recall = np.mean(recall_scores)
#     avg_f1 = np.mean(f1_scores)
#     avg_support = np.mean(support_scores)
#
#     # Store the results in a dictionary
#     result = {
#         'Model': model_name,
#         'Precision': avg_precision,
#         'Recall': avg_recall,
#         'F1-Score': avg_f1,
#         'Support': avg_support
#     }
#     results.append(result)
#
# # Create a pandas DataFrame from the results
# results_df = pd.DataFrame(results)
#
# # Display the table
# print(results_df)
#
# # Create a pandas DataFrame from the results
# results_df = pd.DataFrame(results)
#
# # Convert the DataFrame to LaTeX table format
# latex_table = results_df.to_latex(index=False)
#
# # Print the LaTeX table
# print(latex_table)

# ----------------------------------------------------------------------------------------------------------------------

# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_first, y_first, test_size=0.2, random_state=None)
#
# # Perform feature scaling using StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Define the logistic regression model
# logreg = LogisticRegression()
#
# # Define the hyperparameters grid
# param_grid = {
#     'C': [0.01, 0.1, 1.0, 10.0],  # regularization parameter
#     'penalty': ['l1', 'l2'],  # regularization type
#     'solver': ['liblinear', 'saga'],  # algorithm for optimization
#     'max_iter': [200, 500, 1000, 2000],  # maximum number of iterations
#     'dual': [True, False],  # dual formulation
#     'class_weight': [None, 'balanced'],  # class weights
#     'tol': [1e-3, 1e-4, 1e-5]  # tolerance
# }
#
# # Apply random oversampling
# ros = RandomOverSampler(random_state=None)
# X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train_scaled, y_train)
#
# # Apply SMOTE
# smote = SMOTE(random_state=None)
# X_train_smote, y_train_smote = smote.fit_resample(X_train_oversampled, y_train_oversampled)
#
# # Perform grid search
# grid_search = GridSearchCV(logreg, param_grid, cv=5)
# grid_search.fit(X_train_smote, y_train_smote.values.ravel())
#
# # Get the grid search results
# results = grid_search.cv_results_
#
# # Extract the relevant information into a DataFrame
# param_table = pd.DataFrame({
#     'Parameters': results['params'],
#     'Mean Score': results['mean_test_score'],
#     'Standard Deviation': results['std_test_score']
# })
#
# # Sort the table by mean score in descending order
# param_table = param_table.sort_values('Mean Score', ascending=False)
#
# # Get the parameters with the highest accuracy
# best_parameters = param_table.iloc[0]['Parameters']
# best_accuracy = param_table.iloc[0]['Mean Score']
#
# # Train a new logistic regression model using the best hyperparameters
# best_logreg = LogisticRegression(**best_parameters)
# best_logreg.fit(X_train_scaled, y_train.values.ravel())
#
# # Evaluate the model on the test set
# accuracy_select = best_logreg.score(X_test_scaled, y_test)
#
# # Print the best parameters and accuracy
# print("Best Parameters:", best_parameters)
# print("Best Accuracy:", best_accuracy)
# print("test Accuracy", accuracy_select)

# ----------------------------------------------------------------------------------------------------------------------

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_first, y_first, test_size=0.2, random_state=None)

# Perform feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

num_iterations = 1000  # Number of iterations

models = []
weights = []

for i in range(num_iterations):
    # Apply random oversampling
    ros = RandomOverSampler(random_state=None)
    X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train_scaled, y_train)

    # Apply SMOTE
    smote = SMOTE(random_state=None)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_oversampled, y_train_oversampled)

    # Train a new logistic regression model using the best hyperparameters
    best_logreg = LogisticRegression()
    best_logreg.fit(X_train_smote, y_train_oversampled.values.ravel())

    # Evaluate the model on the test set
    accuracy_select = best_logreg.score(X_test_scaled, y_test)

    # Append the model and its accuracy to the lists
    models.append(best_logreg)
    weights.append(accuracy_select)

# Normalize the weights
weights_sum = sum(weights)
weights = [w / weights_sum for w in weights]

# Create a VotingClassifier with the individual models and their weights
ensemble_model = VotingClassifier(estimators=[('model'+str(i+1), model) for i, model in enumerate(models)],
                                  voting='soft', weights=weights)

# Fit the ensemble model using the entire training set
ensemble_model.fit(X_train_smote, y_train_smote.values.ravel())

# Evaluate the ensemble model on the test set
ensemble_accuracy = ensemble_model.score(X_test_scaled, y_test)

print("Ensemble Accuracy:", ensemble_accuracy)

# ----------------------------------------------------------------------------------------------------------------------