import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("D:/Nexthikes/project4/Processed_Flipdata - Processed_Flipdata.csv")

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# # Get basic information about the dataset
print("\nInfo about the dataset:")
print(data.info())

# Get the shape of the dataset (number of rows and columns)
print("\nShape of the dataset:")
print(data.shape)

# # Summary statistics of numerical columns
print("\nSummary statistics of numerical columns:")
print(data.describe())

# # Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Check for duplicates
print("\nNumber of duplicate rows in the dataset:")
print(data.duplicated().sum())

# # Explore unique values in categorical columns
print("\nUnique values in categorical columns:")
for column in data.select_dtypes(include=['object']):
    print(column + ": ", data[column].unique())

# Range of values for each numerical feature
print("\nRange of values for numerical features:")
for column in data.select_dtypes(include=['int64', 'float64']).columns:
    min_val = data[column].min()
    max_val = data[column].max()
    print(f"{column}: {min_val} - {max_val}")

# Range of values for each feature
print("\nRange of values for each feature:")
for column in data.columns:
    if data[column].dtype != 'object':  # Only consider numerical features
        min_val = data[column].min()
        max_val = data[column].max()
        print(f"{column}: min={min_val}, max={max_val}")

# Filter out non-numeric columns
numeric_columns = data.select_dtypes(include=['int', 'float']).columns

# # Calculate correlation matrix for numerical variables
correlation_matrix = data[numeric_columns].corr()

# # Display correlation matrix
print("\nCorrelation matrix:")
print(correlation_matrix)

# Loop through each column and get the range of values for each feature
for column in data.columns:
    if data[column].dtype in ['int64', 'float64']:  # Check if the column is numerical
        min_val = data[column].min()
        max_val = data[column].max()
        print(f"Range of values for {column}: {min_val} to {max_val}")

# You can also get unique values for categorical columns
for column in data.columns:
    if data[column].dtype == 'object':  # Check if the column is categorical
        unique_values = data[column].unique()
        print(f"Unique values for {column}: {unique_values}")

# Visualize numerical variables
print("\nVisualizing numerical variables:")
for col in data.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 6))
    plt.hist(data[col], bins=20)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()
