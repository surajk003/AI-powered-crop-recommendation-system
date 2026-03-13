"""
generate_meta.py

Creates a minimal metadata JSON for the bundled RandomForest model by
loading `crop_model.joblib` and `crop_preprocessor.joblib` and extracting
feature names and class labels when possible.

This file is safe to run locally and will write `crop_model.joblib.meta.json`.
"""
import json
import os
import joblib

MODEL = "crop_model.joblib"
PRE = "crop_preprocessor.joblib"
OUT = MODEL + ".meta.json"

if not os.path.exists(MODEL):
    raise SystemExit(f"Model file not found: {MODEL}")
if not os.path.exists(PRE):
    raise SystemExit(f"Preprocessor file not found: {PRE}")

clf = joblib.load(MODEL)
pre = joblib.load(PRE)

meta = {}

# Try to obtain classes from the classifier
try:
    classes = getattr(clf, "classes_", None)
    if classes is not None:
        meta["classes"] = list(map(str, classes.tolist())) if hasattr(classes, "tolist") else list(map(str, classes))
except Exception:
    meta["classes"] = None

# Try to find input feature names from the preprocessor
features = None
try:
    # ColumnTransformer fitted should have .feature_names_in_
    features = getattr(pre, "feature_names_in_", None)
    if features is None:
        # try Attribute on transformers
        if hasattr(pre, "transformers_"):
            # transformers_ is list of tuples
            # each tuple: (name, transformer, columns)
            cols = []
            for name, trans, cols_spec in pre.transformers_:
                if isinstance(cols_spec, (list, tuple)):
                    cols.extend(list(cols_spec))
            if cols:
                features = cols
except Exception:
    features = None

meta["features"] = list(features) if features is not None else []

# Save target if available
meta["target"] = None

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Wrote metadata to {OUT}")
