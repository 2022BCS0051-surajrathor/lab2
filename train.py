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

# If using red wine dataset
data = pd.read_csv("dataset/winequality-red.csv", sep=';')

# If your file name is different, change it above


# ------------------------------
# 2️⃣ Remove Non-Numeric Columns (Safety)
# ------------------------------

data = data.select_dtypes(include=['number'])


# ------------------------------
# 3️⃣ Separate Features & Target
# ------------------------------

X = data.drop("quality", axis=1)
y = data["quality"]


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
# 5️⃣ Feature Scaling
# ------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------
# 6️⃣ Model Training
# ------------------------------

model = LinearRegression()
model.fit(X_train_scaled, y_train)


# ------------------------------
# 7️⃣ Prediction
# ------------------------------

y_pred = model.predict(X_test_scaled)


# ------------------------------
# 8️⃣ Evaluation Metrics
# ------------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")


# ------------------------------
# 9️⃣ Create Output Directory
# ------------------------------

os.makedirs("output", exist_ok=True)


# ------------------------------
# 🔟 Save Model
# ------------------------------

joblib.dump(model, "output/model.pkl")


# ------------------------------
# 1️⃣1️⃣ Save Results as JSON
# ------------------------------

results = {
    "MSE": float(mse),
    "R2_Score": float(r2)
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Model and results saved successfully.")
