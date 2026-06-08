# ============================================================
# House Price Predictor — Linear Regression
# Dataset : California Housing (sklearn built-in)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 1. Load Data ────────────────────────────────────────────
print("=" * 55)
print("   HOUSE PRICE PREDICTOR — Linear Regression")
print("=" * 55)

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target  # Price in $100,000s

print(f"\n Dataset shape : {df.shape}")
print(f" Features      : {list(data.feature_names)}")
print(f"\n First 5 rows:\n{df.head()}")

# ── 2. Exploratory Data Analysis ────────────────────────────
print("\n── Data Statistics ──")
print(df.describe().round(2))

print(f"\n Missing values : {df.isnull().sum().sum()}")
print(f" Price range    : ${df['Price'].min()*100:.0f}k — ${df['Price'].max()*100:.0f}k")
print(f" Avg Price      : ${df['Price'].mean()*100:.0f}k")

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()
print("\n Saved: correlation_heatmap.png")

# ── 3. Preprocessing ─────────────────────────────────────────
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n Train samples : {X_train.shape[0]}")
print(f" Test  samples : {X_test.shape[0]}")

# ── 4. Train Models ──────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression" : Ridge(alpha=1.0),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        "MSE" : mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE" : mean_absolute_error(y_test, y_pred),
        "R2"  : r2_score(y_test, y_pred),
    }

# ── 5. Results ───────────────────────────────────────────────
print("\n── Model Results ──")
print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("-" * 50)
for name, m in results.items():
    print(f"{name:<22} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['R2']:>8.4f}")

# ── 6. Actual vs Predicted Plot ──────────────────────────────
best_model = LinearRegression()
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color="steelblue", s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Price ($100k)")
plt.ylabel("Predicted Price ($100k)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()
print("\n Saved: actual_vs_predicted.png")

# ── 7. Feature Importance ────────────────────────────────────
coef_df = pd.DataFrame({
    "Feature"    : data.feature_names,
    "Coefficient": best_model.coef_
}).sort_values("Coefficient", ascending=False)

print("\n── Feature Coefficients (Linear Regression) ──")
print(coef_df.to_string(index=False))

# ── 8. Sample Prediction ─────────────────────────────────────
print("\n── Sample Prediction ──")
sample = X_test_scaled[:1]
pred   = best_model.predict(sample)[0]
actual = y_test.iloc[0]
print(f" Predicted Price : ${pred * 100:.2f}k")
print(f" Actual Price    : ${actual * 100:.2f}k")
print(f" Difference      : ${abs(pred - actual) * 100:.2f}k")

print("\n Done! Check correlation_heatmap.png and actual_vs_predicted.png")
