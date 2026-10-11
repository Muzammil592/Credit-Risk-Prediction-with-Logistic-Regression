"""
Scoring logic for the Streamlit app.

Deliberately dependency-light (numpy + pandas only) and free of pickles: it reads
`model_artifacts.json`, which `src/export_artifacts.py` writes from the trained model.
That keeps the deployed app small and lets the same code be tested without Streamlit.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT_PATH = Path(__file__).parent / "model_artifacts.json"


def load_artifacts(path=None):
    with open(path or ARTIFACT_PATH) as f:
        return json.load(f)


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _age_group(age):
    for upper, label in [(25, "<=25"), (35, "26-35"), (45, "36-45"), (60, "46-60")]:
        if age <= upper:
            return label
    return "60+"


def engineer(art, raw):
    """
    Turn the fields a person would type into the model's input columns, using exactly the
    same rules as src/data_cleaning.py.

    `raw` needs: person_age, person_income, person_home_ownership, loan_intent,
    loan_grade, loan_amnt, cb_person_default_on_file, cb_person_cred_hist_length, plus
    person_emp_length and loan_int_rate, either of which may be None (unknown).
    """
    c = art["cleaning"]
    emp, rate = raw.get("person_emp_length"), raw.get("loan_int_rate")
    emp_missing, rate_missing = emp is None, rate is None
    if emp_missing:
        emp = c["emp_length_median"]
    if rate_missing:
        rate = c["int_rate_median_by_grade"].get(raw["loan_grade"], c["int_rate_overall_median"])

    age = float(raw["person_age"])
    income, amnt = float(raw["person_income"]), float(raw["loan_amnt"])
    hist = float(raw["cb_person_cred_hist_length"])
    pct = round(amnt / income, 2)  # the dataset stores this ratio to 2 decimals

    return {
        "person_age": age,
        "person_income": income,
        "person_emp_length": float(emp),
        "loan_amnt": amnt,
        "loan_int_rate": float(rate),
        "loan_percent_income": pct,
        "cb_person_cred_hist_length": hist,
        "log_income": math.log1p(income),
        "log_loan_amnt": math.log1p(amnt),
        "emp_length_to_age": float(emp) / age,
        "cred_hist_to_age": hist / age,
        "interest_cost_to_income": amnt * float(rate) / 100.0 / income,
        "high_burden": int(pct > 0.30),
        "emp_length_missing": int(emp_missing),
        "int_rate_missing": int(rate_missing),
        "age_group": _age_group(age),
        "person_home_ownership": raw["person_home_ownership"],
        "loan_intent": raw["loan_intent"],
        "loan_grade": raw["loan_grade"],
        "cb_person_default_on_file": raw["cb_person_default_on_file"],
    }


def design_matrix(art, df):
    """Same transformation as src/preprocessing.py: standardise, flags, one-hot (first level dropped)."""
    p = art["preprocessor"]
    mean = np.array(p["mean"], dtype=float)
    std = np.array(p["std"], dtype=float)
    num = (df[p["numeric_features"]].to_numpy(float) - mean) / std
    flags = df[p["flag_features"]].to_numpy(float)
    dummies = []
    for col in p["categorical_features"]:
        values = df[col].astype(str)
        for level in p["categories"][col][1:]:
            dummies.append((values == level).to_numpy(float))
    return np.column_stack([num, flags] + dummies)


def score(art, X):
    """Return (log-odds score, calibrated default probability) for each row of X."""
    z = X @ np.array(art["model"]["coef"]) + art["model"]["intercept"]
    p = _sigmoid(art["calibration"]["a"] * z + art["calibration"]["b"])
    return z, p


def score_applicant(art, raw):
    """Score one applicant. Returns (probability, design row, engineered fields)."""
    feats = engineer(art, raw)
    X = design_matrix(art, pd.DataFrame([feats]))
    _, p = score(art, X)
    return float(p[0]), X[0], feats


def contributions(art, x_row):
    """
    Each feature's push on the log-odds of default: coefficient * (standardised value).
    Positive = raises risk. Numeric features are relative to the average applicant,
    one-hot features relative to their baseline level.
    """
    contrib = np.array(art["model"]["coef"]) * x_row
    return pd.Series(contrib, index=art["preprocessor"]["feature_names"])


def probability_curve(art, raw, field, values):
    """Default probability as one input varies and everything else stays fixed."""
    rows = [engineer(art, {**raw, field: v}) for v in values]
    _, p = score(art, design_matrix(art, pd.DataFrame(rows)))
    return p
