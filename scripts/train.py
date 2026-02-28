import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("dataset/winequalityN.csv")
data.columns = data.columns.str.strip()

data = data.select_dtypes(include=[np.number]).dropna()

X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Experiment 4 - Ridge alpha 1.0 testsize 0.3")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

os.makedirs("output", exist_ok=True)
joblib.dump(model, "output/model.pkl")

with open("output/results.json", "w") as f:
    json.dump({"MSE": float(mse), "R2": float(r2)}, f, indent=4)