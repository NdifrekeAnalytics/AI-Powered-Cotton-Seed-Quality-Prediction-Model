# fix_preprocessor.py
# ============================================================
# Rebuilds 05_preprocessor.joblib and 05_data_splits.joblib
# from scratch using the current sklearn version (1.7.2).
#
# Replaces Block 5 (05_preprocessing_v2.py).
# Run this ONCE from your project folder in base environment:
#   python fix_preprocessor.py
#
# Then run 09_serialization.py to rebuild the bundle.
# ============================================================

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import RobustScaler, OrdinalEncoder
from sklearn.impute           import SimpleImputer
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder

warnings.filterwarnings("ignore")
print(f"sklearn version: {sklearn.__version__}")

# ── PATHS ─────────────────────────────────────────────────────
PROJECT_DIR = (
    r"C:\Users\HomePC\OneDrive - Chemonics\Desktop"
    r"\10Alytics Data Science Class\Amdari Internship"
    r"\Projects\Cotton Seed Germination Quality Prediction"
)

CLEAN_PATH   = os.path.join(PROJECT_DIR, "03_clean_model_ready.parquet")
FEAT_PATH    = os.path.join(PROJECT_DIR, "selected_features.json")
PREP_OUT     = os.path.join(PROJECT_DIR, "05_preprocessor.joblib")
SPLITS_OUT   = os.path.join(PROJECT_DIR, "05_data_splits.joblib")

# Verify required files exist
for p, label in [(CLEAN_PATH, "03_clean_model_ready.parquet"),
                  (FEAT_PATH,  "selected_features.json")]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Required file not found: {label}\nPath: {p}\n"
            f"Ensure Block 3 has been run and saved outputs to PROJECT_DIR."
        )

# ── LOAD DATA ─────────────────────────────────────────────────
print("\n[1] Loading clean dataset ...")
df = pd.read_parquet(CLEAN_PATH)
print(f"    Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

with open(FEAT_PATH, encoding="utf-8") as f:
    feature_names = json.load(f)

feature_names = [c for c in feature_names if c in df.columns]
print(f"    Features found in dataset: {len(feature_names)}")

# ── TARGETS ───────────────────────────────────────────────────
print("\n[2] Preparing targets ...")
df = df.dropna(subset=["WG", "CT"], how="all")
df["WG"] = df["WG"].fillna(df["WG"].median())
df["CT"] = df["CT"].fillna(df["CT"].median())

# Vigor classification
def classify_vigor(row):
    wg    = row["WG"]
    ct    = row["CT"]
    ratio = ct / wg if wg > 0 else 0.0
    if wg >= 80 and ct >= 70 and ratio >= 0.85:
        return "Excellent"
    if wg >= 70 and ct >= 55 and ratio >= 0.65:
        return "Good"
    return "Poor"

df["vigor_class"] = df.apply(classify_vigor, axis=1)

le_vigor = LabelEncoder()
df["vigor_class_enc"] = le_vigor.fit_transform(df["vigor_class"])
class_mapping = dict(zip(le_vigor.classes_,
                          le_vigor.transform(le_vigor.classes_)))
print(f"    Vigor classes: {class_mapping}")

# ── FEATURE MATRIX ────────────────────────────────────────────
print("\n[3] Building feature matrix ...")
X = df[feature_names].copy()

# Fill any remaining NaN with -1 sentinel (structural) or median
for col in X.columns:
    if X[col].isna().any():
        null_pct = X[col].isna().mean()
        if null_pct > 0.3:
            X[col] = X[col].fillna(-1)   # structural NaN → sentinel
        else:
            X[col] = X[col].fillna(X[col].median())

y_wg = df["WG"].values.astype(float)
y_ct = df["CT"].values.astype(float)
y_vc = df["vigor_class_enc"].values.astype(int)

print(f"    X shape: {X.shape}")
print(f"    WG range: {y_wg.min():.1f} – {y_wg.max():.1f}")
print(f"    CT range: {y_ct.min():.1f} – {y_ct.max():.1f}")

# ── STRATIFIED SPLIT 70 / 15 / 15 ────────────────────────────
print("\n[4] Stratified 70/15/15 split on vigor_class ...")
X_tv, X_test, ywg_tv, ywg_test, yct_tv, yct_test, yvc_tv, yvc_test = \
    train_test_split(X, y_wg, y_ct, y_vc,
                     test_size=0.15, random_state=42,
                     stratify=y_vc)

X_train, X_val, ywg_train, ywg_val, yct_train, yct_val = \
    train_test_split(X_tv, ywg_tv, yct_tv,
                     test_size=0.15/(1-0.15), random_state=42,
                     stratify=yvc_tv)

print(f"    Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# ── COLUMN TYPES FOR PIPELINE ─────────────────────────────────
print("\n[5] Defining column types ...")

ORDINAL_COLS = [c for c in [
    "irrigation_type_enc","var_bin","maczone",
    "origin_region_enc","max_stage"
] if c in feature_names]

BINARY_COLS  = [c for c in [
    "thryvon_flag","new_variety_flag","aba_stress_flag",
    "aba_tested_flag","wg_tested_twice","wg_tested_three_times",
    "wg_single_test_only","both_initial_tests_done",
    "wg_degraded_flag","wg_ct_spread_flag",
    "aba_level_elevated","aba_level_high","seed_dual_defect",
    "high_moisture_flag","early_harvest_flag","long_wg_lag_flag",
] if c in feature_names]

NUMERIC_COLS = [c for c in feature_names
                if c not in ORDINAL_COLS + BINARY_COLS]

print(f"    Numeric:  {len(NUMERIC_COLS)}")
print(f"    Ordinal:  {len(ORDINAL_COLS)}")
print(f"    Binary:   {len(BINARY_COLS)}")

# ── BUILD PREPROCESSING PIPELINE ─────────────────────────────
print("\n[6] Building preprocessing pipeline ...")

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  RobustScaler()),
])

ordinal_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )),
])

binary_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

transformers = []
if NUMERIC_COLS:
    transformers.append(("numeric", numeric_pipe, NUMERIC_COLS))
if ORDINAL_COLS:
    transformers.append(("ordinal", ordinal_pipe, ORDINAL_COLS))
if BINARY_COLS:
    transformers.append(("binary",  binary_pipe,  BINARY_COLS))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    verbose_feature_names_out=False,
)

# ── FIT ON TRAINING DATA ONLY ─────────────────────────────────
print("\n[7] Fitting preprocessor on training data ...")
preprocessor.fit(X_train)

X_train_t = preprocessor.transform(X_train)
X_val_t   = preprocessor.transform(X_val)
X_test_t  = preprocessor.transform(X_test)

print(f"    Transformed shape: {X_train_t.shape}")

# Recover output feature names
try:
    out_names = list(preprocessor.get_feature_names_out())
except AttributeError:
    out_names = feature_names[:X_train_t.shape[1]]
print(f"    Output features: {len(out_names)}")

# ── SAVE ──────────────────────────────────────────────────────
print("\n[8] Saving artefacts ...")

joblib.dump(preprocessor, PREP_OUT)
print(f"    Preprocessor -> {PREP_OUT}")

splits_dict = {
    "X_train"      : X_train_t,
    "X_val"        : X_val_t,
    "X_test"       : X_test_t,
    "ywg_train"    : ywg_train,
    "ywg_val"      : ywg_val,
    "ywg_test"     : ywg_test,
    "yct_train"    : yct_train,
    "yct_val"      : yct_val,
    "yct_test"     : yct_test,
    "feature_names": out_names,
    "le_vigor"     : le_vigor,
    "class_mapping": class_mapping,
    "X_test_raw"   : X_test.reset_index(drop=True),
}

joblib.dump(splits_dict, SPLITS_OUT)
print(f"    Data splits  -> {SPLITS_OUT}")

print(f"""
=================================================================
fix_preprocessor.py COMPLETE
sklearn version used: {sklearn.__version__}

Files saved:
  05_preprocessor.joblib
  05_data_splits.joblib

NEXT STEP:
  Run 09_serialization.py to rebuild the deployment bundle:
    python 09_serialization.py
=================================================================
""")
