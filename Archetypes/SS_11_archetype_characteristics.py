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

archetypes = df['archetype'].unique()
mean_std_values = pd.DataFrame()

# Define the desired order of archetypes
desired_order = [0, 1, 2, 3]

for archetype in desired_order:
    archetype_data = df[df['archetype'] == archetype]
    archetype_mean = archetype_data.mean()
    archetype_std = archetype_data.std()
    archetype_mean_std = pd.concat([archetype_mean, archetype_std], axis=0)
    mean_std_values[f"archetype_{archetype}_mean"] = archetype_mean
    mean_std_values[f"archetype_{archetype}_std"] = archetype_std

mean_std_values = mean_std_values.transpose()
mean_std_values_transposed = mean_std_values.transpose()
print(mean_std_values_transposed)

rounded_values = mean_std_values_transposed.round(2)
latex_table = rounded_values.to_latex()
print(latex_table)
