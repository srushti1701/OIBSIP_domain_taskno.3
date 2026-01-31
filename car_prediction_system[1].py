import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ✅ Load dataset (change file name)
df = pd.read_csv("car data.csv")

print("✅ Shape:", df.shape)
print("\n✅ Columns:", df.columns.tolist())
print("\n✅ First 5 rows:\n", df.head())

# ✅ Basic cleaning
df.columns = df.columns.str.strip()
df = df.dropna()

# ✅ TARGET column (price) - common names
possible_targets = ["Selling_Price", "selling_price", "Price", "price", "Car_Price"]
target_col = None
for c in possible_targets:
    if c in df.columns:
        target_col = c
        break

if target_col is None:
    print("\n❌ Target column not found. Your dataset must have Price/Selling_Price.")
    print("👉 Available columns:", df.columns.tolist())
    exit()

# ✅ Features + Target
X = df.drop(columns=[target_col])
y = df[target_col]

# ✅ Identify categorical & numeric columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\n✅ Categorical Columns:", cat_cols)
print("✅ Numeric Columns:", num_cols)

# ✅ Preprocessing: OneHotEncode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# ✅ Model
model = RandomForestRegressor(n_estimators=300, random_state=42)

# ✅ Pipeline
pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Train
pipe.fit(X_train, y_train)

# ✅ Predict
y_pred = pipe.predict(X_test)

# ✅ Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Performance:")
print("MAE  =", mae)
print("RMSE =", rmse)
print("R²   =", r2)

# ✅ Plot: Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.grid(True)
plt.show()

# ✅ Custom Prediction (example)
print("\n✅ Custom Prediction Example:")

sample = X.iloc[[0]]   # dataset ka 1st row sample input
pred_price = pipe.predict(sample)[0]

print("Input Row:\n", sample)
print("\n✅ Predicted Price:", pred_price)
