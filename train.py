import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ------------------------------
# 1️⃣ Load Dataset
# ------------------------------

data = pd.read_csv("dataset/winequalityN.csv")

# Clean column names
data.columns = data.columns.str.strip()

print("Columns:", data.columns)


# ------------------------------
# 2️⃣ Remove Categorical Columns (like 'type')
# ------------------------------

# Keep only numeric columns
data = data.select_dtypes(include=[np.number])

# Remove missing values if any
data = data.dropna()


# ------------------------------
# 3️⃣ Separate Features & Target
# ------------------------------

target_column = "quality"

X = data.drop(target_column, axis=1)
y = data[target_column]


# ------------------------------
# 4️⃣ Train-Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ------------------------------
# 5️⃣ Scaling
# ------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------
# 6️⃣ Train Model
# ------------------------------

model = LinearRegression()
model.fit(X_train_scaled, y_train)


# ------------------------------
# 7️⃣ Prediction
# ------------------------------

y_pred = model.predict(X_test_scaled)


# ------------------------------
# 8️⃣ Metrics
# ------------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")


# ------------------------------
# 9️⃣ Save Outputs
# ------------------------------

os.makedirs("output", exist_ok=True)

joblib.dump(model, "output/model.pkl")

results = {
    "MSE": float(mse),
    "R2_Score": float(r2)
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Model and results saved successfully.")
