"""
Export the trained model to app/model_artifacts.json, then PROVE the app's scoring code
gives the same answers as the training pipeline.

Run from the project root (after modeling.py and evaluation.py):
    python src/export_artifacts.py

Why JSON instead of the .joblib files?
  - joblib files are pickles: they can break across library versions and are git-ignored
  - the JSON is tiny, human-readable, safe to commit, and lets the deployed app run
    without scikit-learn or any of the training code
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
APP_DIR = ROOT / "app"
sys.path.insert(0, str(APP_DIR))  # so we can import the app's scoring module

import scoring  # noqa: E402
from preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FLAG_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    TARGET,
    load_splits,
)

for needed in ("preprocessor.joblib", "logreg_scratch.joblib", "calibrator.joblib",
               "decision_config.json", "model_config.json"):
    if not (MODEL_DIR / needed).exists():
        raise SystemExit(f"Missing models/{needed}. Run modeling.py and evaluation.py first.")

pre = joblib.load(MODEL_DIR / "preprocessor.joblib")
model = joblib.load(MODEL_DIR / "logreg_scratch.joblib")
platt = joblib.load(MODEL_DIR / "calibrator.joblib")
with open(MODEL_DIR / "decision_config.json") as f:
    decision = json.load(f)
with open(MODEL_DIR / "model_config.json") as f:
    model_cfg = json.load(f)
with open(PROCESSED_DIR / "cleaning_params.json") as f:
    cleaning = json.load(f)
train_df, val_df, test_df = load_splits()

# Typical (1st to 99th percentile) ranges of the raw inputs, so the app can warn when a
# user asks about someone unlike anyone the model was trained on.
RANGE_COLS = ["person_age", "person_income", "person_emp_length", "loan_amnt",
              "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
typical = {c: [float(train_df[c].quantile(0.01)), float(train_df[c].quantile(0.99))] for c in RANGE_COLS}

artifacts = {
    "preprocessor": {
        "numeric_features": NUMERIC_FEATURES,
        "flag_features": FLAG_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "mean": [float(pre.mean_[c]) for c in NUMERIC_FEATURES],
        "std": [float(pre.std_[c]) for c in NUMERIC_FEATURES],
        "categories": pre.categories_,
        "feature_names": pre.feature_names_,
    },
    "model": {"coef": [float(c) for c in model.coef_], "intercept": float(model.intercept_)},
    "calibration": {"a": float(platt.a_), "b": float(platt.b_)},
    "decision": {
        "threshold": decision["threshold"],
        "cost_false_negative": decision["cost_false_negative"],
        "cost_false_positive": decision["cost_false_positive"],
    },
    "cleaning": {
        "emp_length_median": cleaning["emp_length_median_train"],
        "int_rate_median_by_grade": cleaning["loan_int_rate_median_by_grade_train"],
        "int_rate_overall_median": cleaning["loan_int_rate_overall_median_train"],
    },
    "typical_ranges": typical,
    "test_metrics": decision["test_metrics_calibrated"],
    "model_info": {
        "penalty": model_cfg["penalty"],
        "C": model_cfg["C"],
        "imbalance_strategy": model_cfg["imbalance_strategy"],
        "train_default_rate": float(train_df[TARGET].mean()),
    },
}
out_path = APP_DIR / "model_artifacts.json"
with open(out_path, "w") as f:
    json.dump(artifacts, f, indent=2)
print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

# ----------------------------------------------------------------------------
# Verification: app scoring vs the training pipeline, on the real test set
# ----------------------------------------------------------------------------
art = scoring.load_artifacts(out_path)

# (a) design matrix and probabilities from already-engineered rows
X_pipe = pre.transform(test_df)
X_app = scoring.design_matrix(art, test_df)
p_pipe = platt.predict_proba(model.decision_function(X_pipe))
_, p_app = scoring.score(art, X_app)
print("\nCheck 1: design matrix and probabilities from engineered test rows")
print(f"  max |matrix difference|      : {np.abs(X_pipe - X_app).max():.2e}")
print(f"  max |probability difference| : {np.abs(p_pipe - p_app).max():.2e}")

# (b) the full path a user takes: raw typed-in fields -> engineered features -> probability
sample = test_df.sample(min(500, len(test_df)), random_state=0)
p_expected = platt.predict_proba(model.decision_function(pre.transform(sample)))
p_raw_path = []
for _, row in sample.iterrows():
    raw = {
        "person_age": row["person_age"],
        "person_income": row["person_income"],
        "person_home_ownership": row["person_home_ownership"],
        "person_emp_length": None if row["emp_length_missing"] == 1 else row["person_emp_length"],
        "loan_intent": row["loan_intent"],
        "loan_grade": row["loan_grade"],
        "loan_amnt": row["loan_amnt"],
        "loan_int_rate": None if row["int_rate_missing"] == 1 else row["loan_int_rate"],
        "cb_person_default_on_file": row["cb_person_default_on_file"],
        "cb_person_cred_hist_length": row["cb_person_cred_hist_length"],
    }
    p_raw_path.append(scoring.score_applicant(art, raw)[0])
diff = np.abs(np.array(p_raw_path) - p_expected)
print("\nCheck 2: raw user inputs -> features -> probability (500 test rows)")
print(f"  max |probability difference| : {diff.max():.2e}")
print(f"  rows differing by > 1e-6     : {(diff > 1e-6).sum()}")

ok = np.abs(p_pipe - p_app).max() < 1e-9 and diff.max() < 5e-3
print("\n" + ("PASS: the app reproduces the training pipeline." if ok else
              "WARNING: mismatch. Check the differing rows before deploying."))
print("\nNext: streamlit run app/app.py")
