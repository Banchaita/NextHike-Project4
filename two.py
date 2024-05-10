# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Read data from Excel sheet
data = pd.read_csv('D:/Nexthikes/project4/New.csv')

# Step 2: Data preprocessing (if needed)
# For example, handle missing values and encode categorical variables

# Step 3: Split data into training and testing sets
X = data.drop('Prize', axis=1)
y = data['Prize']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build a decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 6: Deploy the model (if needed)
# You can save the trained model for future use
