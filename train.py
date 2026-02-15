import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("dataset/winequalityN.csv")

X = data.drop("quality", axis=1)
y = data["quality"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"MSE: {mse}")
print(f"R2: {r2}")

# Save model
joblib.dump(model, "output/model.pkl")

# Save results JSON
results = {
    "MSE": mse,
    "R2": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f)
