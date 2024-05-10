import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv("D:/Nexthikes/project4/New.csv")

# Split the data into features (X) and target variable (y)
X = data.drop(columns=["Prize"])
y = data["Prize"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the prices
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

# Export evaluation results to Excel
evaluation_results = pd.DataFrame({"MAE": [mae], "RMSE": [rmse]})
evaluation_results.to_excel("model_evaluation_results.xlsx", index=False)
