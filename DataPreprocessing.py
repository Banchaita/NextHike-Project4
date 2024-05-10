import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("D:/Nexthikes/project4/Processed_Flipdata - Processed_Flipdata.csv")

# # Check for missing values
print("Missing values before handling:")
print(data.isnull().sum())

# Define threshold for outliers (e.g., z-score > 3 or < -3)
threshold = 3

# Calculate z-scores for numerical columns
z_scores = stats.zscore(data.select_dtypes(include='number'))

# Find outliers
outliers = (z_scores > threshold) | (z_scores < -threshold)

# Replace outliers with NaNs
data[outliers] = np.nan

# Check again for missing values after handling outliers
print("\nMissing values after handling outliers:")
print(data.isnull().sum())

# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(data, columns=['Model', 'Colour'])
print("\n Perform one-hot encoding for categorical variables:")
print(df_encoded)

# Print every line with the encoded data
for index, row in df_encoded.iterrows():
    print(row)

# Optionally, you can save the encoded DataFrame to a new CSV file
df_encoded.to_csv('encoded_file.csv', index=False)


# Convert categorical variables into one-hot encoding
onehot_encoder = OneHotEncoder(sparse_output=False)
encoded_columns = pd.DataFrame(onehot_encoder.fit_transform(data[['Model', 'Colour']]))

# Getting feature names
feature_names = onehot_encoder.get_feature_names_out(['Model', 'Colour'])
encoded_columns.columns = feature_names

# Concatenate the one-hot encoded columns with the original data
data_processed = pd.concat([data.drop(['Model', 'Colour'], axis=1), encoded_columns], axis=1)
print("\n Concatenate the one-hot encoded columns with the original data")
print(data_processed)

# Separate features (X) and target variable (y)
X = data.drop('Model', axis=1)  # Assuming 'target_variable_column_name' is the name of your target variable
y = data['Colour']

# Perform one-hot encoding for categorical variables
categorical_cols = ['Model', 'Colour']  # List of categorical columns to encode
X_categorical = X[categorical_cols]
X_numerical = X.drop(categorical_cols, axis=1)

encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = encoder.fit_transform(X_categorical)

# Combine encoded categorical features and numerical features
X_processed = pd.concat([pd.DataFrame(X_encoded), X_numerical], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Print the first few lines of the processed dataset
print("Processed Dataset:")
print(X_processed.head())




