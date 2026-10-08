"""
Turns the Day 1 train/val/test CSVs into model-ready numeric matrices.

Everything that LEARNS from data (scaling means/stds, which categories exist) is
fitted on the TRAIN split only. The fitted object is saved so the Day 4 app can
apply exactly the same transformation to a single new applicant.

To try different feature sets, just edit the three lists below.
(Day 1 also created emp_length_to_age, cred_hist_to_age, interest_cost_to_income and
age_group. They are left out here to keep the coefficients easy to interpret, but you
can add them and see what happens to the VIF table and the validation scores.)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TARGET = "loan_status"

NUMERIC_FEATURES = [
    "person_age",
    "log_income",
    "person_emp_length",
    "log_loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
FLAG_FEATURES = ["high_burden", "emp_length_missing", "int_rate_missing"]
CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


class Preprocessor:
    """Standardise numeric columns and one-hot encode categoricals (first level dropped)."""

    def fit(self, df: pd.DataFrame):
        self.mean_ = df[NUMERIC_FEATURES].mean()
        std = df[NUMERIC_FEATURES].std(ddof=0)
        self.std_ = std.replace(0, 1.0)
        self.categories_ = {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL_FEATURES}

        # The first level of each category is the "baseline" (dropped), which avoids
        # perfectly collinear dummy columns.
        self.feature_names_ = list(NUMERIC_FEATURES) + list(FLAG_FEATURES)
        for c in CATEGORICAL_FEATURES:
            self.feature_names_ += [f"{c}={v}" for v in self.categories_[c][1:]]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = ((df[NUMERIC_FEATURES] - self.mean_) / self.std_).to_numpy(float)
        flags = df[FLAG_FEATURES].to_numpy(float)
        dummies = []
        for c in CATEGORICAL_FEATURES:
            col = df[c].astype(str)
            for v in self.categories_[c][1:]:
                dummies.append((col == v).to_numpy(float))
        # A category never seen in training becomes all zeros, i.e. the baseline level.
        return np.column_stack([num, flags] + dummies)

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def load_splits():
    """Read the three CSVs written by data_cleaning.py."""
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train, val, test


def vif_table(X: np.ndarray, names) -> pd.DataFrame:
    """
    Variance inflation factors via the inverse correlation matrix.
    Rule of thumb: VIF above ~5 means a feature is largely explained by the others;
    above ~10 is a strong multicollinearity warning.
    """
    corr = np.corrcoef(X, rowvar=False)
    inv = np.linalg.pinv(corr)
    return (
        pd.DataFrame({"feature": names, "VIF": np.diag(inv)})
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )
