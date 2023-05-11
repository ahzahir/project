-------------------------------------

## Code Overview

#### Import necessary libraries
The code imports various libraries such as `numpy`, `pandas`, `seaborn`, `scipy.stats`, `statsmodels.api`, `matplotlib.pyplot`, `scikit-learn` modules, `imbalanced-learn` (imblearn), and `warnings`.

#### Ignore warnings
This line of code sets the warning action to "ignore," which suppresses warnings that may be displayed during the code execution.

#### Read CSV files
The code reads four CSV files named "healthcare-dataset-stroke-data.csv," "test.csv," "train.csv," and "sample_submission.csv" using pandas' `read_csv()` function. The data from these files is stored in separate DataFrames: `datacsv`, `datatest`, `datatrain`, and `datasubmission`.

#### Concatenate and reset index
The code concatenates the `datatrain` and `datacsv` DataFrames using pandas' `concat()` function and then resets the index of the concatenated DataFrame (`data_w`) using `reset_index()`. This step combines the training data and additional stroke data into a single DataFrame.

#### Rename column names
The code replaces any white spaces in the column names of `data_w` using the `columns.str.replace()` method. This step ensures that the column names do not contain white spaces.

#### Calculate parameters of Normal Distribution
The code calculates the main parameters (mean and standard deviation) of the Normal Distribution for the 'stroke' column in `data_w` using `norm.fit()`. These parameters are used to fit a Normal Distribution curve to the data.

#### Plot distribution graph
The code creates a figure and axes using `plt.figure()` and `plt.subplots()` functions and plots a distribution graph of the 'stroke' column in `data_w` using seaborn's `distplot()` function. The graph shows the distribution of stroke events and overlays a fitted Normal Distribution curve based on the calculated parameters.

#### Apply label encoding to categorical columns
The code applies label encoding to categorical columns ('gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status') in `datatrain` and `datatest` DataFrames using sklearn's `LabelEncoder()`. Label encoding converts categorical values into numerical labels.

#### Concatenate and scale the data
The code concatenates `datatrain` and `datatest` DataFrames into a single DataFrame named `train_test`. It then applies standard scaling to the data in `train_test` using sklearn's `StandardScaler()`.

#### Split the scaled data
The code splits the scaled data (`train_test_scaled`) back into separate DataFrames (`datatrain_scaled` and `test_df_scaled`) using the length of `datatrain`.

#### Perform statistical tests and visualizations

##### Shapiro-Wilk test
The code performs the Shapiro-Wilk test (`stats.shapiro()`) on the 'stroke' column in `data_w` to check for normality. The test results (`shap_t` and `shap_p`) are stored for later use.

#### Correlation matrix heatmap
The code creates a correlation matrix using `data_w.corr()` and visualizes it using seaborn's `heatmap()` function. The heatmap displays the correlations between different columns in `data_w`.

#### Count plots
The code creates subplots and uses seaborn's `countplot()` function to generate count plots for each categorical feature in `data_w` (e.g., 'gender', 'hypertension', 'heart_disease') with respect to the 'stroke' column. This helps visualize the distribution of stroke events within each category.

#### Handle missing values
The code handles missing values in the data by using sklearn's `SimpleImputer()` with the strategy set to 'median'. Specifically, it replaces missing values in the 'bmi' column of `data_w` with the median value calculated from the non-missing values.

#### Encode categorical columns
The code performs label encoding on selected categorical columns ('gender', 'work_type', 'ever_married', 'Residence_type', 'smoking_status') in `data_w` using sklearn's `LabelEncoder()`. This step converts the categorical values into numerical labels.

#### Distribution plots
The code creates a figure with subplots to display the distribution of numerical columns ('age', 'avg_glucose_level', 'bmi') in `data_w`. It uses seaborn's `histplot()` function to generate histograms with kernel density estimation (KDE) curves for each numerical column.

#### Outlier detection
The code creates a figure with subplots to detect outliers in the numerical columns ('age', 'avg_glucose_level', 'bmi') of `data_w`. It uses seaborn's `boxplot()` function to generate box plots for each numerical column, which helps visualize the distribution and identify potential outliers.

#### Separating target and features
The code separates the target variable ('stroke') from the input features and assigns them to separate variables. It assigns the 'stroke' column to the target variable and drops the 'stroke' and 'id' columns from `data_w`, storing the resulting DataFrame as `data_w2`. Similarly, the 'id' column is dropped from `datatrain` DataFrame.

#### Splitting the data
The code splits the data (`data_w2` and `target`) into training and testing datasets using sklearn's `train_test_split()` function. It assigns 75% of the data to the training set (`X_train` and `y_train`) and 25% to the testing set (`X_test` and `y_test`). The `random_state` is set to 3 for reproducibility.

#### Apply label encoding to categorical columns
The code applies label encoding to categorical columns ('gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status') in the training set (`X_train_encoded`) and testing set (`X_test_encoded`). This step ensures that the categorical values are encoded as numerical labels consistently.

#### Handling missing values
The code uses the previously created imputer to handle missing values in the training set (`X_train_encoded`) and testing set (`X_test_encoded`). It replaces the missing values with the median values calculated from the non-missing values.

#### Scaling the data
The code scales the training set (`X_train_imputed`) and testing set (`X_test_imputed`) using the previously created scaler. This step standardizes the numerical features to have zero mean and unit variance, ensuring that they are on a similar scale.

#### Handling class imbalance with SMOTE
The code addresses class imbalance in the training data by using the Synthetic Minority Over-sampling Technique (SMOTE) from the imbalanced-learn library. It applies SMOTE to the training set (`X_train_scaled` and `y_train`) using `smote.fit_resample()`, which generates synthetic samples of the minority class (stroke) to balance the dataset. The resampled training set is stored in `X_train_resampled` and `y_train_resampled`.

#### Gradient Boosting Classifier
The code creates an instance of the GradientBoostingClassifier from sklearn's ensemble module. It initializes the classifier with `random_state=3` for reproducibility. Then, it fits the classifier to the resampled training data (`X_train_resampled` and `y_train_resampled`) using the `fit()` method.

#### Prediction and evaluation
The code uses the trained Gradient Boosting Classifier to make predictions on the scaled testing set (`X_test_scaled`) using the predict() method. The predicted values are stored in `y_pred`. It then evaluates the performance of the classifier by calculating the accuracy score using `accuracy_score()` and printing the classification report using `classification_report()`. The accuracy score and classification report are printed to the console.

#### Confusion matrix visualization
The code generates a confusion matrix using sklearn's `confusion_matrix()`. It creates a heatmap visualization of the confusion matrix using seaborn's `heatmap()` function, with annotations and formatted as integers. The x-axis represents the predicted labels, and the y-axis represents the actual labels.

#### Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

#### Create heatmap of confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

#### Prediction on the test dataset
The code prepares the test dataset (`datatest`) for prediction by dropping the 'id' column. It then encodes the categorical columns in the test dataset using label encoding (`LabelEncoder()`) similarly to the training set. Missing values in the test dataset are handled by applying the previously fitted imputer to replace them with the median values. The test dataset is scaled using the previously fitted scaler. Finally, the Gradient Boosting Classifier predicts the stroke events for the test dataset (`test_df_scaled`), and the predictions are stored in `submission_pred`.

#### Prepare test dataset for prediction
test_df = datatest.drop('id', axis=1)
test_df_encoded = label_encoder.transform(test_df[categorical_cols])
test_df_imputed = imputer.transform(test_df_encoded)
test_df_scaled = scaler.transform(test_df_imputed)

#### Make predictions on the test dataset
submission_pred = clf.predict(test_df_scaled)

#### Generating submission file
The code assigns the predicted values (`submission_pred`) to the 'stroke' column of the `datasubmission` DataFrame. Finally, it saves the updated DataFrame as a CSV file named 'submission.csv' using pandas' `to_csv()` function, with `index=False` to exclude the index column from the output file.

#### Assign predicted values to submission DataFrame
datasubmission['stroke'] = submission_pred

#### Save submission DataFrame to CSV
datasubmission.to_csv('submission.csv', index=False)

-------------------------------------
