## Import necessary libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor



# # Load your dataset into a pandas DataFrame
dataOne = pd.read_csv("D:/Nexthikes/project4/encoded_file.csv")


# Define the features (X) and the target variable (y)

X = dataOne.drop(columns=['Battery_'])
y = dataOne['Battery_']

# Split the dataset into training and testing sets (e.g., 70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save the training and testing sets into separate Excel files
X_train.to_excel("Batteryset.xlsx", index=False)


# Load the dataset
data = pd.read_csv('D:/Nexthikes/project4/New.csv')

# Separate features and target variable
X = data.drop('Prize', axis=1)
y = data['Prize']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model
model = DecisionTreeRegressor(random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# # Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Export the model
with open("price_prediction_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Export predictions to Excel
predictions_df = pd.DataFrame({"Actual Price": y_test, "Predicted Price": predictions})
print(predictions_df)
predictions_df.to_excel("price_predictions.xlsx", index=False)