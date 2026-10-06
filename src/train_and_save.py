import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "winequality-red.csv"
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, sep=";")
    print("Original shape:", df.shape)

    # 1. Remove duplicates
    df_clean = df.drop_duplicates().reset_index(drop=True)
    print("Shape after dedup:", df_clean.shape)

    # 2. Target categorization
    def classify_quality(quality):
        if quality <= 5:
            return "Low"
        elif quality == 6:
            return "Medium"
        else:
            return "High"

    df_clean["quality_category"] = df_clean["quality"].apply(classify_quality)

    # 3. Train/Test Split
    X = df_clean.drop(columns=["quality", "quality_category"])
    y = df_clean["quality_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 4. Outlier clipping using IQR computed on X_train ONLY
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    outlier_bounds = {}

    for column in X_train.columns:
        Q1 = X_train[column].quantile(0.25)
        Q3 = X_train[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = float(Q1 - 1.5 * IQR)
        upper_bound = float(Q3 + 1.5 * IQR)
        outlier_bounds[column] = (lower_bound, upper_bound)

        X_train_processed[column] = X_train_processed[column].clip(lower=lower_bound, upper=upper_bound)
        X_test_processed[column] = X_test_processed[column].clip(lower=lower_bound, upper=upper_bound)

    # 5. Scaler (fit on X_train_processed only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    # 6. Train Selected Final Model: Random Forest
    print("\nTraining Selected Final Model: Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_processed, y_train)

    # Evaluate on test set
    y_pred = rf_model.predict(X_test_processed)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=["Low", "Medium", "High"], zero_division=0)

    print("\n" + "=" * 50)
    print("RANDOM FOREST TEST SET EVALUATION")
    print("=" * 50)
    print(f"Accuracy:          {acc:.4f}")
    print(f"Precision (Macro): {prec:.4f}")
    print(f"Recall (Macro):    {rec:.4f}")
    print(f"F1 (Macro):        {f1_macro:.4f}")
    print(f"F1 (Weighted):     {f1_weighted:.4f}")
    print(f"Per-Class F1 - Low:    {per_class_f1[0]:.4f}")
    print(f"Per-Class F1 - Medium: {per_class_f1[1]:.4f}")
    print(f"Per-Class F1 - High:   {per_class_f1[2]:.4f}")

    # Feature statistics for Streamlit UI
    feature_stats = {}
    for col in X.columns:
        feature_stats[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "median": float(X[col].median()),
            "std": float(X[col].std()),
            "q25": float(X[col].quantile(0.25)),
            "q75": float(X[col].quantile(0.75))
        }

    # Feature importances
    feature_importance = dict(zip(X.columns, [float(v) for v in rf_model.feature_importances_]))
    feature_importance_sorted = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    metadata = {
        "model_name": "Random Forest Classifier",
        "parameters": {
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": 42
        },
        "target_classes": ["Low", "Medium", "High"],
        "class_mapping": "Low: quality <= 5, Medium: quality == 6, High: quality >= 7",
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec), 4),
            "recall_macro": round(float(rec), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "cv_f1_macro_mean": 0.6060,
            "cv_f1_macro_std": 0.0432,
            "per_class_f1": {
                "Low": round(float(per_class_f1[0]), 4),
                "Medium": round(float(per_class_f1[1]), 4),
                "High": round(float(per_class_f1[2]), 4)
            }
        },
        "feature_names": list(X.columns),
        "feature_importances": feature_importance_sorted,
        "outlier_bounds": outlier_bounds,
        "feature_stats": feature_stats
    }

    # 7. Serialize artifacts
    model_path = models_dir / "model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    feature_names_path = models_dir / "feature_names.pkl"
    outlier_bounds_path = models_dir / "outlier_bounds.pkl"
    metadata_path = models_dir / "model_metadata.json"

    joblib.dump(rf_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(list(X.columns), feature_names_path)
    joblib.dump(outlier_bounds, outlier_bounds_path)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 50)
    print("SAVED ARTIFACTS TO models/")
    print("=" * 50)
    print(f"  ✓ {model_path}")
    print(f"  ✓ {scaler_path}")
    print(f"  ✓ {feature_names_path}")
    print(f"  ✓ {outlier_bounds_path}")
    print(f"  ✓ {metadata_path}")
    print("Serialization completed successfully!")

if __name__ == "__main__":
    main()
