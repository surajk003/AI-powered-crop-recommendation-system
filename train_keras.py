
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def detect_target_column(df: pd.DataFrame) -> str:
    for candidate in ("label", "crop", "target", "class", "Crop"):
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return col
    return df.columns[-1]


def build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    # determine numeric vs categorical based on current dtypes
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numeric_cols))
    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor


def clean_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Perform basic cleaning on the dataframe before feature processing.

    Steps:
    - Drop exact duplicates
    - Trim whitespace on string columns and convert empty strings to NaN
    - Try to coerce numeric-like columns to numeric
    - Drop rows with missing target
    - Drop constant columns
    """
    df = df.copy()

    # Drop duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    if len(df) != n_before:
        print(f"Dropped {n_before - len(df)} duplicate rows")

    # Trim object (string) columns
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan})

    # Coerce numeric-like columns (except target) to numeric
    for col in df.columns:
        if col == target_col:
            continue
        sample = df[col].dropna().head(100).astype(str)
        if len(sample) == 0:
            continue
        num_like = sample.str.match(r"^[+-]?\d+(\.\d+)?$").sum()
        if num_like / len(sample) > 0.6:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing target
    if target_col in df.columns:
        n_before = len(df)
        df = df[~df[target_col].isna()]
        if len(df) != n_before:
            print(f"Dropped {n_before - len(df)} rows with missing target '{target_col}'")

    # Drop constant columns
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const_cols:
        print(f"Dropping constant columns: {const_cols}")
        df = df.drop(columns=const_cols)

    return df





def main(
    data_path: str,
    model_out: str = "crop_model.joblib",
    preprocessor_out: str = "crop_preprocessor.joblib",
    meta_out: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 50,
    batch_size: int = 32,
):
    if not os.path.exists(data_path):
        raise SystemExit(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if df.shape[0] == 0:
        raise SystemExit("Empty dataset")

    target = detect_target_column(df)
    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found")

    # Basic data cleaning before preprocessing
    df = clean_dataframe(df, target)

    X = df.drop(columns=[target])
    y_raw = df[target]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    feature_cols = list(X.columns)

    preprocessor = build_preprocessor(X, feature_cols)
    X_proc = preprocessor.fit_transform(X)

    input_dim = X_proc.shape[1]
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=test_size, random_state=random_state, stratify=y if n_classes > 1 else None
    )

    # Train a RandomForestClassifier on the processed features
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    print("Starting training RandomForest...")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    joblib.dump(clf, model_out)
    joblib.dump(preprocessor, preprocessor_out)

    meta = {
        "model_file": os.path.basename(model_out),
        "preprocessor_file": os.path.basename(preprocessor_out),
        "features": feature_cols,
        "classes": le.classes_.tolist(),
        "target": target,
    }
    if meta_out is None:
        meta_out = model_out + ".meta.json"

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to: {model_out}")
    print(f"Saved preprocessor to: {preprocessor_out}")
    print(f"Saved metadata to: {meta_out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Keras model for crop recommendation")
    p.add_argument("--data", required=True, help="Path to CSV data file")
    p.add_argument("--model-out", default="crop_model.joblib", help="Output model path (joblib)")
    p.add_argument("--preprocessor-out", default="crop_preprocessor.joblib", help="Output preprocessor (joblib)")
    p.add_argument("--meta-out", help="Metadata JSON output path (optional)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        data_path=args.data,
        model_out=args.model_out,
        preprocessor_out=args.preprocessor_out,
        meta_out=args.meta_out,
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
