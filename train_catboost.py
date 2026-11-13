# =========================================
# CATBOOST BUILDING AGE MODEL
# Non-leaking pipeline with BGRI-disjoint split
# - BGRI21 used ONLY for grouping
# - Type included as categorical predictor
# - 85% development / 15% held-out test
# - GroupKFold CV inside development
# - Feature-set comparison + random hyperparameter search
# - Final test only if RUN_FINAL_TEST = True
# =========================================

import argparse
import os
import json
import random
import warnings
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ============================================================
# 0) SETTINGS
# ============================================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser(description="Train the Lisbon CatBoost building-age model.")
parser.add_argument("csv_path", help="Path to the labelled building inventory CSV.")
parser.add_argument("--output-dir", default="CatBoost_Age_Model_Output")
args = parser.parse_args()
CSV_PATH = args.csv_path
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_OUT = os.path.join(OUTPUT_DIR, "catboost_age_model.cbm")
BEST_CONFIG_JSON = os.path.join(OUTPUT_DIR, "best_model_config.json")
SPLIT_DEV_GROUPS_CSV = os.path.join(OUTPUT_DIR, "dev_bgri_groups.csv")
SPLIT_TEST_GROUPS_CSV = os.path.join(OUTPUT_DIR, "test_bgri_groups.csv")
CV_RESULTS_CSV = os.path.join(OUTPUT_DIR, "cv_results.csv")
TEST_PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "test_predictions.csv")
FEATURE_IMPORTANCE_CSV = os.path.join(OUTPUT_DIR, "feature_importance.csv")

RUN_FINAL_TEST = False   # keep False during development
SAVE_SPLIT = True

TEST_SIZE = 0.15
N_SPLITS = 5
N_PARAM_SAMPLES = 20
MAX_SPLIT_TRIES = 500
MIN_COMMON_CLASS_COUNT = 20   # classes with at least this many samples should appear in held-out test if possible

# ============================================================
# 1) LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)

# ============================================================
# 2) REQUIRED COLUMNS
# ============================================================
needed = [
    "BGRI21",
    "Age_Buildings",
    "Type",
    "BGRI_density",
    "DIST_Marques",
    "DIST_Pavilhao_Atlantico",
    "DIST_Sa_Carneiro_Square",
    "DIST_Terreiro_Paco",
    "BGRI_LiDAR_intensity_STD",
    "BGRI21_N_EDIFICIOS_1_OU_2_PISOS",
    "LiDAR_elevation_Mean",
    "LiDAR_Red_RGB_Mean",
    "Compactness",
    "PCT95"
]

missing_cols = [c for c in needed if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

df = df[needed].copy()

# ============================================================
# 3) BASIC CLEANING
# ============================================================
df["BGRI21"] = df["BGRI21"].astype(str).str.strip()
df["Type"] = df["Type"].astype(str).str.strip()
df["Age_Buildings"] = df["Age_Buildings"].astype(str).str.strip()

df = df[
    df["BGRI21"].notna() & (df["BGRI21"] != "") &
    df["Type"].notna() & (df["Type"] != "") &
    df["Age_Buildings"].notna() & (df["Age_Buildings"] != "")
].copy()

# ============================================================
# 4) HARMONIZE AGE CLASSES
# ============================================================
df["Age_Buildings"] = df["Age_Buildings"].replace({
    "pre XVII": "pre XVIII",
    "XVII": "pre XVIII"
})

class_levels = [
    "pre XVIII",
    "XVIII",
    "XIX",
    "1900 - 1930",
    "1930 - 1960",
    "1960 - 1983",
    "pos 1983"
]

df = df[df["Age_Buildings"].isin(class_levels)].copy()

label_map = {label: i for i, label in enumerate(class_levels)}
inv_label_map = {i: label for label, i in label_map.items()}
df["target"] = df["Age_Buildings"].map(label_map)

# ============================================================
# 5) NUMERIC COERCION
# ============================================================
num_cols = [
    "BGRI_density",
    "DIST_Marques",
    "DIST_Pavilhao_Atlantico",
    "DIST_Sa_Carneiro_Square",
    "DIST_Terreiro_Paco",
    "BGRI_LiDAR_intensity_STD",
    "BGRI21_N_EDIFICIOS_1_OU_2_PISOS",
    "LiDAR_elevation_Mean",
    "LiDAR_Red_RGB_Mean",
    "Compactness",
    "PCT95"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=num_cols).copy()

# Keep Type as categorical predictor for CatBoost
df["Type"] = df["Type"].astype("category")

# ============================================================
# 6) HELPER FUNCTIONS
# ============================================================
def ordinal_metrics(y_true_labels, y_pred_labels):
    age_map = {
        "pre XVIII": 1700,
        "XVIII": 1750,
        "XIX": 1850,
        "1900 - 1930": 1915,
        "1930 - 1960": 1945,
        "1960 - 1983": 1970,
        "pos 1983": 2000
    }
    true_year = np.array([age_map[v] for v in y_true_labels], dtype=float)
    pred_year = np.array([age_map[v] for v in y_pred_labels], dtype=float)

    r2 = r2_score(true_year, pred_year)
    mae = mean_absolute_error(true_year, pred_year)
    rmse = np.sqrt(mean_squared_error(true_year, pred_year))
    return r2, mae, rmse

def print_class_counts(name, y_codes):
    counts = pd.Series(y_codes).map(inv_label_map).value_counts().reindex(class_levels, fill_value=0)
    print(f"\nClass counts ({name}):")
    print(counts)

def make_heldout_group_split(df_in, test_size=0.15, max_tries=500, seed=42):
    """
    Creates a BGRI-disjoint dev/test split.
    Tries to keep common classes represented in the test set.
    """
    rng = np.random.RandomState(seed)

    all_groups = np.array(sorted(df_in["BGRI21"].unique()))
    n_test_groups = max(1, int(round(test_size * len(all_groups))))

    global_counts = df_in["Age_Buildings"].value_counts()
    common_classes = set(global_counts[global_counts >= MIN_COMMON_CLASS_COUNT].index.tolist())

    for attempt in range(max_tries):
        shuffled_groups = all_groups.copy()
        rng.shuffle(shuffled_groups)

        test_groups = set(shuffled_groups[:n_test_groups])
        dev_groups = set(shuffled_groups[n_test_groups:])

        dev_df = df_in[df_in["BGRI21"].isin(dev_groups)].copy()
        test_df = df_in[df_in["BGRI21"].isin(test_groups)].copy()

        dev_classes = set(dev_df["Age_Buildings"].unique())
        test_classes = set(test_df["Age_Buildings"].unique())

        # Development must contain all classes
        if set(class_levels) - dev_classes:
            continue

        # Test should contain all common classes
        if common_classes - test_classes:
            continue

        return dev_df, test_df, dev_groups, test_groups, attempt + 1

    raise RuntimeError("Failed to create a valid held-out BGRI split. Increase MAX_SPLIT_TRIES.")

def prepare_X_y_groups(df_in, features):
    X = df_in[features].copy()
    y = df_in["target"].values
    groups = df_in["BGRI21"].values
    return X, y, groups

# ============================================================
# 7) FEATURE SETS TO TEST
# ============================================================
feature_sets = {
    "all_features_plus_type": num_cols + ["Type"],

    "no_distance_plus_type": [
        "BGRI_density",
        "BGRI_LiDAR_intensity_STD",
        "BGRI21_N_EDIFICIOS_1_OU_2_PISOS",
        "LiDAR_elevation_Mean",
        "LiDAR_Red_RGB_Mean",
        "Compactness",
        "PCT95",
        "Type"
    ],

    "lidar_morph_type": [
        "LiDAR_elevation_Mean",
        "LiDAR_Red_RGB_Mean",
        "Compactness",
        "PCT95",
        "Type"
    ],

    "all_features_no_type": num_cols
}

# ============================================================
# 8) HYPERPARAMETER SEARCH SPACE
# ============================================================
param_space = {
    "depth": [6, 7, 8, 9, 10],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "l2_leaf_reg": [1, 3, 5, 7, 10],
    "random_strength": [0, 0.5, 1, 2],
    "border_count": [128, 254, 512],
    "bagging_temperature": [0, 0.5, 1, 2, 3]
}

param_list = list(ParameterSampler(param_space, n_iter=N_PARAM_SAMPLES, random_state=SEED))

# ============================================================
# 9) CREATE FROZEN HELD-OUT SPLIT
# ============================================================
dev_df, test_df, dev_groups, test_groups, split_attempt = make_heldout_group_split(
    df_in=df,
    test_size=TEST_SIZE,
    max_tries=MAX_SPLIT_TRIES,
    seed=SEED
)

print(f"Held-out split created in {split_attempt} attempt(s).")
print(f"Rows -> DEV: {len(dev_df)} | TEST: {len(test_df)}")
print(f"Unique BGRIs -> DEV: {dev_df['BGRI21'].nunique()} | TEST: {test_df['BGRI21'].nunique()}")

print_class_counts("DEV", dev_df["target"].values)
print_class_counts("TEST", test_df["target"].values)

if SAVE_SPLIT:
    pd.DataFrame({"BGRI21": sorted(dev_groups)}).to_csv(SPLIT_DEV_GROUPS_CSV, index=False)
    pd.DataFrame({"BGRI21": sorted(test_groups)}).to_csv(SPLIT_TEST_GROUPS_CSV, index=False)
    print("\nSaved frozen dev/test BGRI group lists.")

# ============================================================
# 10) GROUP-K-FOLD CV SEARCH ON DEVELOPMENT SET
# ============================================================
gkf = GroupKFold(n_splits=N_SPLITS)

results = []
best_score = -np.inf
best_config = None

for fs_name, features in feature_sets.items():
    print(f"\n=== Testing feature set: {fs_name} ===")

    X_dev, y_dev, g_dev = prepare_X_y_groups(dev_df, features)
    cat_feature_indices = [i for i, col in enumerate(features) if col == "Type"]

    for p_i, params in enumerate(param_list, start=1):
        fold_scores = []

        for fold_id, (train_idx, val_idx) in enumerate(gkf.split(X_dev, y_dev, groups=g_dev), start=1):
            X_train = X_dev.iloc[train_idx].copy()
            X_val = X_dev.iloc[val_idx].copy()

            y_train = y_dev[train_idx]
            y_val = y_dev[val_idx]

            train_pool = Pool(X_train, label=y_train, cat_features=cat_feature_indices)
            val_pool = Pool(X_val, label=y_val, cat_features=cat_feature_indices)

            model = CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="TotalF1",
                iterations=4000,
                early_stopping_rounds=150,
                auto_class_weights="SqrtBalanced",
                random_seed=SEED,
                verbose=False,
                **params
            )

            model.fit(train_pool, eval_set=val_pool, use_best_model=True)

            pred_val = model.predict(val_pool).reshape(-1).astype(int)
            fold_macro_f1 = f1_score(y_val, pred_val, average="macro")
            fold_scores.append(fold_macro_f1)

        mean_macro_f1 = float(np.mean(fold_scores))
        sd_macro_f1 = float(np.std(fold_scores))

        row = {
            "feature_set": fs_name,
            "features": "|".join(features),
            "mean_cv_macro_f1": mean_macro_f1,
            "sd_cv_macro_f1": sd_macro_f1,
            **params
        }
        results.append(row)

        print(
            f"Feature set: {fs_name} | "
            f"param set {p_i}/{len(param_list)} | "
            f"mean CV Macro-F1 = {mean_macro_f1:.4f} | "
            f"sd = {sd_macro_f1:.4f}"
        )

        if mean_macro_f1 > best_score:
            best_score = mean_macro_f1
            best_config = {
                "feature_set_name": fs_name,
                "features": features,
                "cat_feature_indices": cat_feature_indices,
                "params": params,
                "best_mean_cv_macro_f1": mean_macro_f1,
                "best_sd_cv_macro_f1": sd_macro_f1
            }

cv_results_df = pd.DataFrame(results).sort_values("mean_cv_macro_f1", ascending=False)
cv_results_df.to_csv(CV_RESULTS_CSV, index=False)

print("\n=== BEST DEVELOPMENT CONFIGURATION ===")
print(f"Best feature set: {best_config['feature_set_name']}")
print(f"Best features: {best_config['features']}")
print(f"Best hyperparameters: {best_config['params']}")
print(f"Best mean CV Macro-F1: {best_config['best_mean_cv_macro_f1']:.4f}")
print(f"Best CV SD: {best_config['best_sd_cv_macro_f1']:.4f}")

with open(BEST_CONFIG_JSON, "w", encoding="utf-8") as f:
    json.dump(best_config, f, indent=2)

# ============================================================
# 11) TRAIN FINAL MODEL ON FULL DEVELOPMENT SET
# ============================================================
best_features = best_config["features"]
best_params = best_config["params"]
best_cat_idx = best_config["cat_feature_indices"]

X_dev_final = dev_df[best_features].copy()
y_dev_final = dev_df["target"].values

dev_pool_final = Pool(X_dev_final, label=y_dev_final, cat_features=best_cat_idx)

final_model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="TotalF1",
    iterations=4000,
    auto_class_weights="SqrtBalanced",
    random_seed=SEED,
    verbose=100,
    **best_params
)

final_model.fit(dev_pool_final)
final_model.save_model(MODEL_OUT)

print(f"\nSaved final development-trained model to:\n{MODEL_OUT}")

# ============================================================
# 12) OPTIONAL FINAL HELD-OUT TEST
# ============================================================
RUN_FINAL_TEST = True

if RUN_FINAL_TEST:
    X_test = test_df[best_features].copy()
    y_test = test_df["target"].values
    test_pool = Pool(X_test, label=y_test, cat_features=best_cat_idx)

    pred_test = final_model.predict(test_pool).reshape(-1).astype(int)

    acc = accuracy_score(y_test, pred_test)
    macro_f1 = f1_score(y_test, pred_test, average="macro")
    weighted_f1 = f1_score(y_test, pred_test, average="weighted")

    y_true_labels = pd.Series(y_test).map(inv_label_map).values
    y_pred_labels = pd.Series(pred_test).map(inv_label_map).values
    r2_ord, mae_ord, rmse_ord = ordinal_metrics(y_true_labels, y_pred_labels)

    print("\n=== FINAL HELD-OUT TEST RESULTS ===")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print(f"Ordinal R2:  {r2_ord:.4f}")
    print(f"Ordinal MAE: {mae_ord:.2f} years")
    print(f"Ordinal RMSE:{rmse_ord:.2f} years")

    print("\n=== CONFUSION MATRIX (TEST) ===")
    cm = confusion_matrix(y_test, pred_test, labels=list(range(len(class_levels))))
    cm_df = pd.DataFrame(cm, index=class_levels, columns=class_levels)
    print(cm_df)

    print("\n=== CLASSIFICATION REPORT (TEST) ===")
    print(classification_report(
        y_test,
        pred_test,
        labels=list(range(len(class_levels))),
        target_names=class_levels,
        digits=4
    ))

    pred_out = test_df[["BGRI21", "Age_Buildings", "Type"]].copy()
    pred_out["true_code"] = y_test
    pred_out["pred_code"] = pred_test
    pred_out["pred_label"] = pd.Series(pred_test).map(inv_label_map).values
    pred_out.to_csv(TEST_PREDICTIONS_CSV, index=False)

    print(f"\nSaved test predictions to:\n{TEST_PREDICTIONS_CSV}")

    feat_imp = pd.DataFrame({
        "Feature": best_features,
        "Importance": final_model.get_feature_importance(dev_pool_final)
    }).sort_values("Importance", ascending=False)

    feat_imp.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    print("\n=== FEATURE IMPORTANCE ===")
    print(feat_imp.to_string(index=False))

# ============================================================
# 13) FINAL DEPLOYMENT MODEL ON 100% OF LABELED DATA
# ============================================================

print("\n=== TRAINING FINAL DEPLOYMENT MODEL ON 100% OF LABELED DATA ===")

X_full = df[best_features].copy()
y_full = df["target"].values

full_pool = Pool(X_full, label=y_full, cat_features=best_cat_idx)

deployment_model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="TotalF1",
    iterations=4000,
    auto_class_weights="SqrtBalanced",
    random_seed=SEED,
    verbose=100,
    **best_params
)

deployment_model.fit(full_pool)

MODEL_OUT_FULL = os.path.join(OUTPUT_DIR, "catboost_age_model_FULL_100pct.cbm")
deployment_model.save_model(MODEL_OUT_FULL)

print(f"\nSaved final 100% deployment model to:\n{MODEL_OUT_FULL}")