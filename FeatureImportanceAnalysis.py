# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


# Load your dataset
data = pd.read_csv('D:/Nexthikes/project4/New.csv')

# Split data into features and target variable
X = data.drop('Prize', axis=1)
y = data['Prize']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Generate some example data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Get feature importances
importances = clf.feature_importances_

# Create a DataFrame to store feature names and importances
feature_importances_df = pd.DataFrame({'Feature': range(X.shape[1]), 'Importance': importances})

# Sort the DataFrame by feature importances
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.yticks(feature_importances_df['Feature'])
plt.show()

# Export feature importances to an Excel sheet
feature_importances_df.to_excel('feature_importances.xlsx', index=False)
