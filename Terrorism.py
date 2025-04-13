import pandas as pd

# Replace 'your_file.csv' with the actual file name or path
file_path = 'DataSet.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Display the first 5 rows
print(df.head())

from sklearn.preprocessing import LabelEncoder


# 1. Missing value identification
print("Missing values per column:")
print(df.isnull().sum())

# 2. Label encoding (for object/string columns only)
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)  # Ensure consistent string type
    df[col] = label_encoder.fit_transform(df[col])

# Preview the processed data
print("\nData after label encoding:")
print(df.head())


# Replace 'TargetColumnName' with the actual name of your target column
target_column = 'attacktype1_txt'

# Split the data
X = df.drop(columns=[target_column])
Y = df[target_column]

# Preview the result
print("Features (X):")
print(X.head())

print("\nTarget (Y):")
print(Y.head())

import matplotlib.pyplot as plt
import seaborn as sns


# 1. Basic Info
print("\nDataset Info:")
print(df.info())

# 2. Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# 3. Value counts of target variable
print(f"\nTarget variable distribution:\n{df[target_column].value_counts()}")

# 4. Correlation matrix
print("\nCorrelation matrix:")
corr = df.corr()
print(corr)

# Plotting correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 5. Distribution of features
df.hist(figsize=(15, 10), bins=20)
plt.suptitle("Histograms of Features", fontsize=16)
plt.show()

# 6. Boxplots to detect outliers
for column in X.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

# Optional: Pairplot (only if dataset is small)
# sns.pairplot(df, hue=target_column)
# plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Drop columns with any NaN values
df = df.dropna(axis=1)

# Recreate X and Y after dropping NaN columns
X = df.drop(columns=[target_column])
Y = df[target_column]

print("Remaining columns after dropping NaN-containing columns:")
print(X.columns)


# 1. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 2. Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 3. Make predictions
y_pred = model.predict(X_test)

# 4. Accuracy & classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy*100:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 6. Error metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
misclassified = (y_test != y_pred).sum()

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Number of misclassified samples: {misclassified}")





import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset and preprocess it
file_path = 'DataSet.csv'
df = pd.read_csv(file_path)

# Drop target column from features (before model training)
X = df.drop(columns=['country_txt'])  # Drop the target column from features
Y = df['country_txt']  # Target column

# Handle missing values (Imputation)
# Impute numeric columns with the mean
numeric_imputer = SimpleImputer(strategy='mean')  
X[X.select_dtypes(include=['float64', 'int64']).columns] = numeric_imputer.fit_transform(X.select_dtypes(include=['float64', 'int64']))

# Impute categorical columns with the mode (most frequent value)
categorical_imputer = SimpleImputer(strategy='most_frequent')  
X[X.select_dtypes(include=['object']).columns] = categorical_imputer.fit_transform(X.select_dtypes(include=['object']))

# Label encoding for categorical columns
label_encoder = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = label_encoder.fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy & classification report
accuracy = accuracy_score(y_test, y_pred)
# print(f"\nAccuracy: {accuracy:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# Prediction function for new data
def predict_country(iyear, imonth, iday, city, attacktype1_txt, targtype1_txt, corp1, target1, 
                    natlty1_txt, gname, weaptype1_txt, weapsubtype1_txt, weapdetail, nkill, nwound, 
                    propextent_txt, ransomamt):
    
    # Ensure the input data matches the feature columns used in training
    input_data = {
        'iyear': [iyear],
        'imonth': [imonth],
        'iday': [iday],
        'city': [city],
        'attacktype1_txt': [attacktype1_txt],
        'targtype1_txt': [targtype1_txt],
        'corp1': [corp1],
        'target1': [target1],
        'natlty1_txt': [natlty1_txt],
        'gname': [gname],
        'weaptype1_txt': [weaptype1_txt],
        'weapsubtype1_txt': [weapsubtype1_txt],
        'weapdetail': [weapdetail],
        'nkill': [nkill],
        'nwound': [nwound],
        'propextent_txt': [propextent_txt],
        'ransomamt': [ransomamt]
    }

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Handle missing values in input (same as training data)
    input_df[input_df.select_dtypes(include=['float64', 'int64']).columns] = numeric_imputer.transform(input_df.select_dtypes(include=['float64', 'int64']))
    input_df[input_df.select_dtypes(include=['object']).columns] = categorical_imputer.transform(input_df.select_dtypes(include=['object']))

    # 1. Label Encoding for categorical columns (same as during training)
    categorical_columns = input_df.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        input_df[col] = label_encoder.fit_transform(input_df[col])

    # 2. Ensure columns in the input match the model's expected input
    input_df = input_df[X.columns]  # Reorder the input to match the training set columns

    # 3. Predict with the trained model
    prediction = model.predict(input_df)

    # Return the predicted country (target)
    return prediction[0]

# Example of how to call the function for prediction
predicted_country = predict_country(
    iyear=2025, imonth=4, iday=13, city="New York", attacktype1_txt="Bombing", 
    targtype1_txt="Civilian", corp1="Some Corp", target1="Target Name", 
    natlty1_txt="USA", gname="Terrorist Group", weaptype1_txt="Explosive", 
    weapsubtype1_txt="Dynamite", weapdetail="Dynamite used", nkill=10, 
    nwound=15, propextent_txt="High", ransomamt=500000
)

print(f"Predicted country: {predicted_country}")
