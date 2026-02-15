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

data = pd.read_csv("dataset/winequalityN.csv", sep=';')

# Remove spaces from column names
data.columns = data.columns.str.strip()

print("Columns in dataset:", data.columns)


# ------------------------------
# 2️⃣ Keep Only Numeric Columns
# ------------------------------

data = data.select_dtypes(include=['number'])


# ------------------------------
# 3️⃣ Detect Target Column Automatically
# ------------------------------

if "quality" in data.columns:
    target_column = "quality"
else:
    # If column name slightly different
    target_column = data.columns[-1]  # Assume last column is target

print("Target column used:", target_column)


# ------------------------------
# 4️⃣ Separate Features & Target
# ------------------------------

X = data.drop(target_column, axis=1)
y = data[target_column]


# ------------------------------
# 5️⃣ Train-Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ------------------------------
# 6️⃣ Feature Scaling
# ------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------
# 7️⃣ Model Training
# ------------------------------

model = LinearRegression()
model.fit(X_train_scaled, y_train)


# ------------------------------
# 8️⃣ Prediction
# ------------------------------

y_pred = model.predict(X_test_scaled)


# ------------------------------
# 9️⃣ Evaluation Metrics
# ------------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")


# ------------------------------
# 🔟 Create Output Directory
# ------------------------------

os.makedirs("output", exist_ok=True)


# ------------------------------
# 1️⃣1️⃣ Save Model
# ------------------------------

joblib.dump(model, "output/model.pkl")


# ------------------------------
# 1️⃣2️⃣ Save Results JSON
# ------------------------------

results = {
    "MSE": float(mse),
    "R2_Score": float(r2)
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Model and results saved successfully.")
