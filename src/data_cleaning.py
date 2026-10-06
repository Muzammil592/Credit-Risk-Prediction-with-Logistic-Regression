"""
Day 1: Data cleaning, EDA, feature engineering, and train/val/test split
for the Credit Risk dataset (target: loan_status, 1 = default).

Run from the project root:
    python src/day1_eda_cleaning.py

Design choices (worth mentioning in your README):
  1. Deterministic cleaning (duplicates, impossible values) happens BEFORE the split.
  2. Anything that LEARNS from data (imputation medians) is fitted on TRAIN only
     and then applied to val/test, to avoid data leakage.
  3. EDA plots use the TRAIN split only, so the val/test sets stay untouched.
  4. Encoding and scaling are deliberately left for Day 2 (fit on train only).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Paths and settings
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "dataset" / "credit_risk_dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "loan_status"
SEED = 42
sns.set_theme(style="whitegrid")

NUMERIC_COLS = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
CATEGORICAL_COLS = [
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file",
]


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()


def section(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ----------------------------------------------------------------------------
# 1. Load and first look
# ----------------------------------------------------------------------------
section("1. LOAD AND FIRST LOOK")
df = pd.read_csv(RAW_PATH)
print(f"Shape: {df.shape}")
print("\nDtypes:\n", df.dtypes)
print("\nHead:\n", df.head())
print("\nNumeric summary:\n", df[NUMERIC_COLS].describe().T)

# ----------------------------------------------------------------------------
# 2. Data quality audit
# ----------------------------------------------------------------------------
section("2. DATA QUALITY AUDIT")
n_raw = len(df)

n_dupes = df.duplicated().sum()
print(f"Exact duplicate rows: {n_dupes}")

missing = df.isna().sum()
print("\nMissing values:\n", missing[missing > 0])
print("Missing %:\n", (df.isna().mean() * 100).round(2)[missing > 0])

print("\nSuspicious values:")
print(f"  person_age > 100:        {(df['person_age'] > 100).sum()}")
print(f"  person_age > 80:         {(df['person_age'] > 80).sum()}")
print(f"  person_emp_length > 60:  {(df['person_emp_length'] > 60).sum()}")
print(f"  person_income > 1M:      {(df['person_income'] > 1_000_000).sum()}")
emp_impossible = (df["person_emp_length"] > (df["person_age"] - 14))
print(f"  emp_length > age - 14:   {emp_impossible.sum()}")
hist_impossible = df["cb_person_cred_hist_length"] > (df["person_age"] - 14)
print(f"  cred_hist > age - 14:    {hist_impossible.sum()}")

# Is loan_percent_income consistent with loan_amnt / person_income?
calc_ratio = df["loan_amnt"] / df["person_income"]
print(f"\nMax |loan_percent_income - loan_amnt/income|: "
      f"{(df['loan_percent_income'] - calc_ratio).abs().max():.3f}")

# ----------------------------------------------------------------------------
# 3. Deterministic cleaning (safe to do before splitting)
# ----------------------------------------------------------------------------
section("3. CLEANING")
df = df.drop_duplicates().reset_index(drop=True)

# Flag missingness BEFORE imputing: missing interest rate / emp length can itself be signal
df["emp_length_missing"] = df["person_emp_length"].isna().astype(int)
df["int_rate_missing"] = df["loan_int_rate"].isna().astype(int)

# Remove impossible records (rules documented so they can go in the README)
bad_age = df["person_age"] > 100
bad_emp = df["person_emp_length"] > 60
bad_emp_vs_age = df["person_emp_length"] > (df["person_age"] - 14)
bad_hist_vs_age = df["cb_person_cred_hist_length"] > (df["person_age"] - 14)
drop_mask = bad_age | bad_emp | bad_emp_vs_age | bad_hist_vs_age
print(f"Rows dropped for impossible values: {drop_mask.sum()}")
df = df[~drop_mask].reset_index(drop=True)

print(f"Rows: {n_raw} raw -> {len(df)} after cleaning")
print(f"Default rate: {df[TARGET].mean():.3%}")

# ----------------------------------------------------------------------------
# 4. Stratified train / validation / test split (70 / 15 / 15)
# ----------------------------------------------------------------------------
section("4. TRAIN / VAL / TEST SPLIT")
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df[TARGET], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df[TARGET], random_state=SEED
)
for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{name:5s}: {len(part):6d} rows | default rate {part[TARGET].mean():.3%}")

# ----------------------------------------------------------------------------
# 5. Imputation (statistics learned from TRAIN only)
# ----------------------------------------------------------------------------
section("5. IMPUTATION (fit on train only)")
emp_median = float(train_df["person_emp_length"].median())
overall_rate_median = float(train_df["loan_int_rate"].median())
# Interest rate is strongly tied to loan grade, so impute by grade
rate_by_grade = train_df.groupby("loan_grade")["loan_int_rate"].median().to_dict()
print(f"emp_length median (train): {emp_median}")
print("loan_int_rate median by grade (train):")
for g, v in sorted(rate_by_grade.items()):
    print(f"  {g}: {v:.2f}")


def impute(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["person_emp_length"] = frame["person_emp_length"].fillna(emp_median)
    grade_fill = frame["loan_grade"].map(rate_by_grade).fillna(overall_rate_median)
    frame["loan_int_rate"] = frame["loan_int_rate"].fillna(grade_fill)
    return frame


train_df, val_df, test_df = impute(train_df), impute(val_df), impute(test_df)
assert train_df[NUMERIC_COLS].isna().sum().sum() == 0
assert val_df[NUMERIC_COLS].isna().sum().sum() == 0
assert test_df[NUMERIC_COLS].isna().sum().sum() == 0
print("No missing values remain in any split.")

# ----------------------------------------------------------------------------
# 6. Feature engineering (row-wise, so no leakage risk)
# ----------------------------------------------------------------------------
section("6. FEATURE ENGINEERING")


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    # Log transforms tame heavy right skew in income and loan amount
    frame["log_income"] = np.log1p(frame["person_income"])
    frame["log_loan_amnt"] = np.log1p(frame["loan_amnt"])
    # Stability / maturity ratios
    frame["emp_length_to_age"] = frame["person_emp_length"] / frame["person_age"]
    frame["cred_hist_to_age"] = frame["cb_person_cred_hist_length"] / frame["person_age"]
    # Burden flag: loan is more than 30% of annual income (heuristic threshold)
    frame["high_burden"] = (frame["loan_percent_income"] > 0.30).astype(int)
    # Interest rate relative to the loan amount scale
    frame["interest_cost_to_income"] = (
        frame["loan_amnt"] * frame["loan_int_rate"] / 100 / frame["person_income"]
    )
    # Age bands for non-linear age effects
    frame["age_group"] = pd.cut(
        frame["person_age"],
        bins=[0, 25, 35, 45, 60, 200],
        labels=["<=25", "26-35", "36-45", "46-60", "60+"],
    ).astype(str)
    return frame


train_df, val_df, test_df = engineer(train_df), engineer(val_df), engineer(test_df)
new_feats = [
    "log_income", "log_loan_amnt", "emp_length_to_age", "cred_hist_to_age",
    "high_burden", "interest_cost_to_income", "age_group",
    "emp_length_missing", "int_rate_missing",
]
print("Engineered / added features:", new_feats)

# ----------------------------------------------------------------------------
# 7. EDA on TRAIN only
# ----------------------------------------------------------------------------
section("7. EDA (train split only)")
eda = train_df.copy()

# 7.1 Class balance
fig, ax = plt.subplots(figsize=(4.5, 4))
eda[TARGET].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
ax.set_xticklabels(["Non-default (0)", "Default (1)"], rotation=0)
ax.set_title("Class balance (train)")
ax.set_ylabel("Count")
savefig("01_class_balance.png")

# 7.2 Numeric distributions, raw
dist_cols = NUMERIC_COLS
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), dist_cols):
    sns.histplot(eda[col], bins=40, kde=False, ax=ax)
    ax.set_title(col)
axes.ravel()[-1].axis("off")
savefig("02_numeric_distributions.png")

# 7.3 Skew check: raw vs log income
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(eda["person_income"], bins=50, ax=axes[0])
axes[0].set_title("person_income (raw)")
sns.histplot(eda["log_income"], bins=50, ax=axes[1])
axes[1].set_title("log_income")
savefig("03_income_log_transform.png")

# 7.4 Numeric features vs target
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
box_cols = ["person_age", "log_income", "person_emp_length", "loan_amnt",
            "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
for ax, col in zip(axes.ravel(), box_cols):
    sns.boxplot(data=eda, x=TARGET, y=col, ax=ax, showfliers=False)
    ax.set_title(f"{col} by default")
axes.ravel()[-1].axis("off")
savefig("04_numeric_vs_target.png")

# 7.5 Default rate by categorical feature
cat_plot_cols = CATEGORICAL_COLS + ["age_group"]
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, col in zip(axes.ravel(), cat_plot_cols):
    rate = eda.groupby(col)[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=rate.index, y=rate.values, ax=ax, color="#4C72B0")
    ax.axhline(eda[TARGET].mean(), color="red", ls="--", lw=1, label="overall rate")
    ax.set_title(f"Default rate by {col}")
    ax.set_ylabel("Default rate")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
savefig("05_default_rate_by_category.png")

# 7.6 Correlation heatmap
corr_cols = NUMERIC_COLS + ["log_income", "emp_length_to_age", "cred_hist_to_age",
                            "interest_cost_to_income", TARGET]
plt.figure(figsize=(11, 9))
sns.heatmap(eda[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix (train)")
savefig("06_correlation_heatmap.png")

# 7.7 Does missingness carry signal?
print("\nDefault rate by missing flags (train):")
print(eda.groupby("int_rate_missing")[TARGET].agg(["mean", "count"]))
print(eda.groupby("emp_length_missing")[TARGET].agg(["mean", "count"]))

# 7.8 Key numbers for your README
print("\nDefault rate by loan_grade:")
print(eda.groupby("loan_grade")[TARGET].agg(["mean", "count"]).round(3))
print("\nTop correlations with target:")
print(eda[corr_cols].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3))

# 7.9 Multicollinearity preview (full VIF can come on Day 2/3)
print("\nCorrelation between overlapping features:")
print(eda[["loan_int_rate", "loan_percent_income", "loan_amnt", "person_income"]].corr().round(2))

# ----------------------------------------------------------------------------
# 8. Save outputs
# ----------------------------------------------------------------------------
section("8. SAVE")
train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

cleaning_params = {
    "seed": SEED,
    "split": "70/15/15 stratified on loan_status",
    "rows_raw": int(n_raw),
    "duplicates_removed": int(n_dupes),
    "rows_dropped_impossible": int(drop_mask.sum()),
    "emp_length_median_train": emp_median,
    "loan_int_rate_median_by_grade_train": {k: float(v) for k, v in rate_by_grade.items()},
    "loan_int_rate_overall_median_train": overall_rate_median,
    "drop_rules": [
        "person_age > 100",
        "person_emp_length > 60",
        "person_emp_length > person_age - 14",
        "cb_person_cred_hist_length > person_age - 14",
    ],
}
with open(PROCESSED_DIR / "cleaning_params.json", "w") as f:
    json.dump(cleaning_params, f, indent=2)

print(f"Saved train/val/test and cleaning_params.json to {PROCESSED_DIR}")
print(f"Saved figures to {FIG_DIR}")
print("\nDay 1 complete. Next: encoding, scaling, and logistic regression from scratch (Day 2).")