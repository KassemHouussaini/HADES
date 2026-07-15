# === HTC XGBoost TRAINER (reduced feature set) ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# --------------------------------------------------------
# 1️⃣ Load and prepare data
# --------------------------------------------------------
TRAIN_XLSX = "HTC_dataset_Tippayawong.xlsx"

# --- Only these features ---
feature_cols = [
    "time (min)",
    "temp(C)",
    "VM(%)",
    "FC(%)",
    "Ci(%)",
    "Hi(%)",
    "Ni(%)",
    "Si(%)",
    "Ashi",
    "Sv_BL"
]

# --- Targets remain the same ---
target_cols = [
    "Bio_char(%)",
    "HHVo(Mj/kg)",
    "Co(%)",
    "Ho(%)",
    "Oo(%)",
    "No(%)",
    "So(%)",
    "Asho"
]

# Load and clean data
df = pd.read_excel(TRAIN_XLSX)
X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
y = df[target_cols].apply(pd.to_numeric, errors="coerce")
mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
X, y = X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)

# --------------------------------------------------------
# 2️⃣ Train/test split
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------
# 3️⃣ Define and train model
# --------------------------------------------------------
base = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    tree_method="hist",
)
model = MultiOutputRegressor(base)
model.fit(X_train, y_train)

# --------------------------------------------------------
# 4️⃣ Evaluate quickly
# --------------------------------------------------------
def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(((a - b)**2).mean()))

y_pred = model.predict(X_test)
print("\n=== TRAIN: Test-set metrics (reduced features) ===")
for i, col in enumerate(target_cols):
    r2 = r2_score(y_test[col], y_pred[:, i])
    e = rmse(y_test[col], y_pred[:, i])
    print(f"{col:>12} | R² = {r2:6.3f} | RMSE = {e:8.3f}")

# --------------------------------------------------------
# 5️⃣ Save model + metadata
# --------------------------------------------------------
joblib.dump(model, "xgb_tippayawong_reduced_model.pkl")
joblib.dump(feature_cols, "xgb_tippayawong_reduced_features.pkl")
joblib.dump(target_cols, "xgb_tippayawong_reduced_targets.pkl")
print("\n✅ Saved reduced model and feature/target metadata successfully!")
