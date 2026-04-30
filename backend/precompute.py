"""
HeartLens - Precompute Pipeline
Downloads UCI Heart Disease data, trains XGBoost, computes SHAP + UMAP,
and saves everything to data/ for the Flask API to serve.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)
import xgboost as xgb
import shap
import umap
import joblib

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── 1. Load UCI Heart Disease Dataset ──────────────────────────────────────────

SITE_URLS = {
    "Cleveland":    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "Hungary":      "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "Switzerland":  "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "VA Long Beach":"https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def load_data():
    """Load all 4 sites and combine into one DataFrame."""
    frames = []
    for site, url in SITE_URLS.items():
        print(f"  Loading {site}...")
        try:
            df = pd.read_csv(url, names=COLUMNS, na_values="?", header=None)
        except Exception:
            # Fallback: try with different separators
            try:
                df = pd.read_csv(url, names=COLUMNS, na_values="?", header=None, sep=r"\s+")
            except Exception as e:
                print(f"    WARNING: Could not load {site}: {e}")
                continue
        df["site"] = site
        frames.append(df)
        print(f"    Loaded {len(df)} records")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total records: {len(combined)}")
    return combined


def preprocess(df):
    """
    Preprocess: impute missing values, binarize target, encode categoricals.
    Returns (X_processed DataFrame, y Series, raw_df with imputed values).
    """
    # Binarize target: 0 = no disease, 1+ = disease present
    df["target"] = (df["target"] > 0).astype(int)

    # Separate features
    feature_cols = [c for c in COLUMNS if c != "target"]

    # Define types
    continuous = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    # Impute missing values
    for col in continuous:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    for col in categorical:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val.iloc[0])
        else:
            df[col] = df[col].fillna(0)

    # Save raw feature values before encoding (for frontend display)
    raw_features = df[feature_cols].copy()

    # One-hot encode categoricals with >2 levels
    onehot_cols = ["cp", "restecg", "slope", "thal"]
    for col in onehot_cols:
        df[col] = df[col].astype(int)

    X_encoded = pd.get_dummies(df[feature_cols], columns=onehot_cols, drop_first=False)

    # Standard-scale continuous features
    scaler = StandardScaler()
    X_encoded[continuous] = scaler.fit_transform(X_encoded[continuous])

    y = df["target"]

    return X_encoded, y, raw_features, scaler, df


# ── 2. Train XGBoost ──────────────────────────────────────────────────────────

def train_model(X, y):
    """Train XGBoost with 5-fold stratified CV. Single-threaded to avoid memory issues."""
    print("\n  Splitting data: 80% train / 20% hold-out test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    print("\n  Training XGBoost with GridSearchCV (n_jobs=1, memory-safe)...")

    param_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200],
    }

    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        nthread=1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\n  Best params: {grid_search.best_params_}")
    print(f"  Best CV accuracy (train folds): {grid_search.best_score_:.4f}")

    # ── Evaluate on hold-out test set ──────────────────────────────────────────
    y_pred      = best_model.predict(X_test)
    y_prob      = best_model.predict_proba(X_test)[:, 1]

    test_acc    = accuracy_score(y_test, y_pred)
    test_roc    = roc_auc_score(y_test, y_prob)
    test_pr_auc = average_precision_score(y_test, y_prob)
    cm          = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*50}")
    print("  HOLD-OUT TEST SET EVALUATION")
    print(f"{'='*50}")
    print(f"  Accuracy        : {test_acc:.4f}")
    print(f"  ROC-AUC         : {test_roc:.4f}")
    print(f"  PR-AUC          : {test_pr_auc:.4f}")
    print(f"\n  Confusion Matrix (threshold=0.5):")
    print(f"              Pred 0   Pred 1")
    print(f"  Actual 0  :  {tn:5d}    {fp:5d}   (TN / FP)")
    print(f"  Actual 1  :  {fn:5d}    {tp:5d}   (FN / TP)")
    print(f"\n  Sensitivity (Recall) : {tp/(tp+fn):.4f}  ← missed disease rate")
    print(f"  Specificity          : {tn/(tn+fp):.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Disease','Disease'])}")
    print(f"{'='*50}")

    eval_metrics = {
        "cv_accuracy":   round(float(grid_search.best_score_), 4),
        "test_accuracy": round(test_acc, 4),
        "test_roc_auc":  round(test_roc, 4),
        "test_pr_auc":   round(test_pr_auc, 4),
        "test_sensitivity": round(float(tp/(tp+fn)), 4),
        "test_specificity": round(float(tn/(tn+fp)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "best_params":   grid_search.best_params_,
    }

    return best_model, eval_metrics


# ── 3. SHAP Values ────────────────────────────────────────────────────────────

def compute_shap(model, X):
    """Compute TreeSHAP values for all patients."""
    print("\n  Computing TreeSHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    base_value = float(explainer.expected_value)
    print(f"  Base value (expected output): {base_value:.4f}")
    print(f"  SHAP matrix shape: {shap_values.shape}")
    return shap_values, base_value, explainer


# ── 4. UMAP Embeddings ───────────────────────────────────────────────────────

def compute_umap(X):
    """Compute 2D UMAP projection."""
    print("\n  Computing UMAP embeddings...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, low_memory=True)
    embedding = reducer.fit_transform(X)
    print(f"  UMAP shape: {embedding.shape}")
    return embedding


# ── 5. Save Everything ────────────────────────────────────────────────────────

def save_outputs(model, scaler, X_encoded, y, raw_features, full_df,
                 shap_values, base_value, umap_embedding, eval_metrics):
    """Save model, data, SHAP, UMAP as files for the Flask API."""

    # Save model
    joblib.dump(model, os.path.join(DATA_DIR, "xgboost_model.joblib"))
    joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.joblib"))
    print("  Saved model and scaler")

    # Save encoded feature names
    feature_names = X_encoded.columns.tolist()
    with open(os.path.join(DATA_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Compute prediction probabilities
    probs = model.predict_proba(X_encoded)[:, 1]

    # Build patient records
    patients = []
    raw_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    for i in range(len(full_df)):
        features_dict = {}
        for col in raw_cols:
            val = raw_features.iloc[i][col]
            features_dict[col] = float(val) if pd.notna(val) else None

        # SHAP values mapped to encoded feature names
        shap_dict = {}
        for j, fname in enumerate(feature_names):
            shap_dict[fname] = round(float(shap_values[i, j]), 4)

        # Aggregate SHAP to original features (sum one-hot encoded)
        shap_original = {}
        for col in raw_cols:
            matching = [k for k in shap_dict if k == col or k.startswith(col + "_")]
            shap_original[col] = round(sum(shap_dict[k] for k in matching), 4)

        patient = {
            "id": i,
            "site": full_df.iloc[i]["site"],
            "features": features_dict,
            "riskProb": round(float(probs[i]), 4),
            "target": int(y.iloc[i]),
            "shapValues": shap_original,
            "shapValuesEncoded": shap_dict,
            "baseValue": round(base_value, 4),
            "umapX": round(float(umap_embedding[i, 0]), 4),
            "umapY": round(float(umap_embedding[i, 1]), 4),
        }
        patients.append(patient)

    with open(os.path.join(DATA_DIR, "patients.json"), "w") as f:
        json.dump(patients, f)
    print(f"  Saved {len(patients)} patient records")

    # Save encoded X for what-if recomputation
    X_encoded.to_csv(os.path.join(DATA_DIR, "X_encoded.csv"), index=False)
    y.to_csv(os.path.join(DATA_DIR, "y.csv"), index=False)

    # Summary stats
    stats = {
        "total_patients": len(patients),
        "sites": full_df["site"].value_counts().to_dict(),
        "target_distribution": y.value_counts().to_dict(),
        "base_value": round(base_value, 4),
        "feature_names_encoded": feature_names,
        "feature_names_original": raw_cols,
        **eval_metrics,
    }
    with open(os.path.join(DATA_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print("  Saved stats.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HeartLens - Precompute Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading UCI Heart Disease Dataset...")
    df = load_data()

    print("\n[2/5] Preprocessing...")
    X_encoded, y, raw_features, scaler, full_df = preprocess(df)
    print(f"  Encoded features: {X_encoded.shape[1]}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")

    print("\n[3/5] Training XGBoost...")
    model, eval_metrics = train_model(X_encoded, y)

    print("\n[4/5] Computing SHAP + UMAP...")
    shap_values, base_value, explainer = compute_shap(model, X_encoded)
    umap_embedding = compute_umap(X_encoded)

    print("\n[5/5] Saving outputs...")
    save_outputs(model, scaler, X_encoded, y, raw_features, full_df,
                 shap_values, base_value, umap_embedding, eval_metrics)

    print("\n" + "=" * 60)
    print("Precompute complete! All files saved to data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
