import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import skew, norm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings(action="ignore")

################################################################

datacsv = pd.read_csv("healthcare-dataset-stroke-data.csv")
datatest = pd.read_csv("test.csv")
datatrain = pd.read_csv("train.csv")
datasubmission = pd.read_csv("sample_submission.csv")

data_w = pd.concat([datatrain, datacsv]).reset_index(drop=True)
data_w.columns = data_w.columns.str.replace(' ', '')  # Replacing the white spaces in columns' names

#~~ Getting the main parameters of the Normal Distribution ~~#
(mu, sigma) = norm.fit(data_w['stroke'])

plt.figure(figsize=(12, 6))
sns.distplot(data_w['stroke'], kde=True, hist=True, fit=norm, bins=6)
plt.title('Stroke Distribution vs Normal Distribution', fontsize=13)
plt.xlabel("Stroke Event", fontsize=12)
plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f})'.format(mu, sigma)], loc='best')

###################################################################

# Apply label encoder to categorical columns in datatrain and datatest
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
le = LabelEncoder()
for col in categorical_columns:
    datatrain[col] = le.fit_transform(datatrain[col].astype(str))
    datatest[col] = le.transform(datatest[col].astype(str))

# Concatenate datatrain and datatest
train_test = pd.concat([datatrain, datatest], axis=0, sort=False)

# Scale the data
scale = StandardScaler()
train_test_scaled = pd.DataFrame(scale.fit_transform(train_test), columns=train_test.columns)

# Split the data into datatrain and datatest
datatrain_scaled = train_test_scaled[:len(datatrain)]
test_df_scaled = train_test_scaled[len(datatrain):]

####################################################################

#~~ Skew and Kurtosis ~~#
shap_t, shap_p = stats.shapiro(data_w['stroke'])

#~~ Correlation Matrix ~~#
plt.figure(figsize=(12, 6))
sns.heatmap(data=data_w.corr(), annot=True, cmap="crest", fmt=".2g")

# Plots between categorical features and target
cat_cols = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]
fig, ax = plt.subplots(3, 3, figsize=(15, 8))
for idx, feature in enumerate(cat_cols):
    row = idx // 3
    col = idx % 3
    sns.countplot(data=data_w, x=feature, hue='stroke', ax=ax[row, col])
# Handle missing values
imp = SimpleImputer(missing_values=np.nan, strategy='median')
data_w['bmi'] = imp.fit_transform(data_w[['bmi']])
enc_col = ["gender", "work_type", 'ever_married', "Residence_type", "smoking_status"]
enc = LabelEncoder()
for col in enc_col:
    data_w[col] = enc.fit_transform(data_w[col].values.reshape(-1, 1))

# Distribution of numerical columns
plt.figure(figsize=(16, 5))
num_cols = ['age', 'avg_glucose_level', 'bmi']
for i, col in enumerate(num_cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(data=data_w, x=col, kde=True)
plt.tight_layout()
plt.show()

# Outlier detection using box plots
plt.figure(figsize=(16, 5))
num_cols = ['age', 'avg_glucose_level', 'bmi']
for i, col in enumerate(num_cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x=data_w[col], palette="Blues")
plt.show()

# Separating target and features
target = data_w['stroke']
datatest_id = datatest['id']
data_w2 = data_w.drop(['stroke', 'id'], axis=1)
datatrain = datatrain.drop(['id'], axis=1)

# Splitting the data into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(data_w2, target, test_size=0.25, random_state=3)

# Apply label encoder to categorical columns in X_train and X_test
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()
for col in categorical_columns:
    le = LabelEncoder()
    X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
    X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))

# Handling missing values in X_train and X_test
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_encoded)
X_test_imputed = imputer.transform(X_test_encoded)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_encoded.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_encoded.columns)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=3)
gb.fit(X_train_resampled, y_train_resampled)
y_pred = gb.predict(X_test_scaled)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Test-set accuracy score:", accuracy)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Prediction on test dataset
test_df = datatest.drop(['id'], axis=1)
test_df_encoded = test_df.copy()
for col in categorical_columns:
    le = LabelEncoder()
    test_df_encoded[col] = le.fit_transform(test_df_encoded[col].astype(str))
test_df_imputed = imputer.transform(test_df_encoded)
test_df_scaled = pd.DataFrame(scaler.transform(test_df_imputed), columns=test_df_encoded.columns)
submission_pred = gb.predict(test_df_scaled)
datasubmission['stroke'] = submission_pred
datasubmission.to_csv('submission.csv', index=False)

