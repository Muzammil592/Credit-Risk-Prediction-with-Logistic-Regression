"""
Day 3: evaluation, calibration, business-cost threshold, and interpretation.

Run from the project root (after data_cleaning.py and modeling.py):
    python src/evaluation.py

Rules that keep this honest
  - Calibration and the decision threshold are chosen on the VALIDATION set.
  - The TEST set is used once, at the end, only to report how those choices perform.

Steps
  1. Load the saved model and fix probabilities with Platt scaling
  2. Train a gradient-boosting baseline for comparison
  3. Test-set metrics: ROC-AUC, Gini, KS, PR-AUC, Brier, log-loss
  4. Choose the decision threshold by business cost (not the default 0.5)
  5. Risk-decile table (does higher predicted risk really mean more defaults?)
  6. Odds ratios with bootstrap confidence intervals
  7. Save the calibrator and decision settings for the app
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from calibration import PlattCalibrator
from logreg_scratch import LogisticRegressionScratch
from preprocessing import TARGET, load_splits

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

SEED = 42
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# BUSINESS ASSUMPTIONS. These are illustrative, so change them to match your story.
# Cost of approving a loan that then defaults (lost principal), relative to the cost
# of rejecting a loan that would have been repaid (lost interest/profit).
# With COST_FP = 1, COST_FN = 5 means "a missed default hurts 5x more than a lost customer".
# ----------------------------------------------------------------------------
COST_FN = 5.0
COST_FP = 1.0
SENSITIVITY_RATIOS = [1, 2, 5, 10, 20]
N_BOOTSTRAP = 200


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def resample(X, y, mode, seed):
    """Same random over/under-sampling used during training (training data only)."""
    rg = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if mode == "oversample":
        pos = rg.choice(pos, size=len(neg), replace=True)
    elif mode == "undersample":
        neg = rg.choice(neg, size=len(pos), replace=False)
    keep = rg.permutation(np.concatenate([pos, neg]))
    return X[keep], y[keep]


def metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": float(np.max(tpr - fpr)),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)),
    }


def cost_per_applicant(y, pred):
    fn = np.sum((pred == 0) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    return (fn * COST_FN + fp * COST_FP) / len(y)


# ----------------------------------------------------------------------------
# 1. Load model, calibrate
# ----------------------------------------------------------------------------
section("1. LOAD MODEL AND CALIBRATE PROBABILITIES")
for needed in ("preprocessor.joblib", "logreg_scratch.joblib", "model_config.json"):
    if not (MODEL_DIR / needed).exists():
        raise SystemExit(f"Missing models/{needed}. Run python src/modeling.py first.")

pre = joblib.load(MODEL_DIR / "preprocessor.joblib")
model = joblib.load(MODEL_DIR / "logreg_scratch.joblib")
with open(MODEL_DIR / "model_config.json") as f:
    config = json.load(f)
print(f"Loaded model: penalty={config['penalty']}, C={config['C']}, "
      f"imbalance strategy = {config['imbalance_strategy']}")

train_df, val_df, test_df = load_splits()
X_train, X_val, X_test = (pre.transform(d) for d in (train_df, val_df, test_df))
y_train = train_df[TARGET].to_numpy(int)
y_val = val_df[TARGET].to_numpy(int)
y_test = test_df[TARGET].to_numpy(int)

z_val, z_test = model.decision_function(X_val), model.decision_function(X_test)
p_val_raw, p_test_raw = model.predict_proba(X_val), model.predict_proba(X_test)

platt = PlattCalibrator().fit(z_val, y_val)               # fitted on validation only
p_val, p_test = platt.predict_proba(z_val), platt.predict_proba(z_test)

iso = IsotonicRegression(out_of_bounds="clip").fit(p_val_raw, y_val)
p_test_iso = iso.predict(p_test_raw)

print(f"Platt scaling: p = sigmoid({platt.a_:.3f} * score + {platt.b_:.3f})")
print(f"\nTest default rate (actual)        : {y_test.mean():.3f}")
print(f"Mean predicted prob, raw model    : {p_test_raw.mean():.3f}")
print(f"Mean predicted prob, after Platt  : {p_test.mean():.3f}")

# ----------------------------------------------------------------------------
# 2. Gradient boosting baseline
# ----------------------------------------------------------------------------
section("2. TREE-BASED BASELINE (gradient boosting)")
gb = HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=500, early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=20, random_state=SEED,
).fit(X_train, y_train)
p_test_gb = gb.predict_proba(X_test)[:, 1]
print(f"Gradient boosting stopped after {gb.n_iter_} trees")

# ----------------------------------------------------------------------------
# 3. Test-set metrics
# ----------------------------------------------------------------------------
section("3. TEST-SET METRICS")
rows = {
    "Logistic (scratch), raw": metrics(y_test, p_test_raw),
    "Logistic (scratch), calibrated": metrics(y_test, p_test),
    "Logistic (scratch), isotonic": metrics(y_test, p_test_iso),
    "Gradient boosting": metrics(y_test, p_test_gb),
}
metrics_df = pd.DataFrame(rows).T.rename_axis("model").reset_index()
metrics_df.to_csv(REPORT_DIR / "day3_test_metrics.csv", index=False)
print(metrics_df.round(4).to_string(index=False))
print("\nReading guide: ROC-AUC/Gini/KS/PR-AUC measure ranking (higher is better).")
print("Brier and log-loss measure probability quality (lower is better); calibration should")
print("improve them without changing the ranking metrics.")

# ROC + PR curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for label, p in [("Logistic (calibrated)", p_test), ("Gradient boosting", p_test_gb)]:
    fpr, tpr, _ = roc_curve(y_test, p)
    axes[0].plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(y_test, p):.3f})")
    prec, rec, _ = precision_recall_curve(y_test, p)
    axes[1].plot(rec, prec, label=f"{label} (AP {average_precision_score(y_test, p):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test)")
axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1, label="No-skill baseline")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test)")
axes[0].legend(loc="lower right")
axes[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig(FIG_DIR / "10_roc_pr_curves.png", dpi=150)
plt.close()

# Calibration (reliability) curves
plt.figure(figsize=(6.5, 6))
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
for label, p in [("Logistic, raw", p_test_raw), ("Logistic, Platt", p_test),
                 ("Gradient boosting", p_test_gb)]:
    frac, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
    plt.plot(mean_pred, frac, marker="o", label=label)
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed default rate")
plt.title("Reliability curve (test)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "11_calibration.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# 4. Threshold chosen by business cost (on VALIDATION)
# ----------------------------------------------------------------------------
section("4. DECISION THRESHOLD BY BUSINESS COST")
grid = np.round(np.arange(0.01, 0.99, 0.005), 3)


def cost_curve(y, p, c_fn, c_fp):
    out = []
    for t in grid:
        pred = p >= t
        fn = np.sum(~pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        out.append((fn * c_fn + fp * c_fp) / len(y))
    return np.array(out)


val_costs = cost_curve(y_val, p_val, COST_FN, COST_FP)
test_costs = cost_curve(y_test, p_test, COST_FN, COST_FP)
THRESHOLD = float(grid[np.argmin(val_costs)])
theory = COST_FP / (COST_FP + COST_FN)
print(f"Cost assumption: missed default = {COST_FN:g}, rejected good loan = {COST_FP:g}")
print(f"Theoretical optimum for calibrated probabilities: {theory:.3f}")
print(f"Empirical optimum on validation                 : {THRESHOLD:.3f}")


def describe(pred, y):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    approved = (pred == 0).sum()
    return {
        "approval_rate": approved / len(y),
        "recall_defaults": tp / (tp + fn),
        "precision_defaults": tp / (tp + fp) if (tp + fp) else 0.0,
        "default_rate_among_approved": fn / approved if approved else 0.0,
        "cost_per_applicant": cost_per_applicant(y, pred),
    }


policies = {
    f"Chosen threshold ({THRESHOLD:.3f})": (p_test >= THRESHOLD).astype(int),
    "Default threshold (0.50)": (p_test >= 0.5).astype(int),
    "Approve everyone": np.zeros_like(y_test),
    "Reject everyone": np.ones_like(y_test),
}
policy_df = pd.DataFrame({k: describe(v, y_test) for k, v in policies.items()}).T
policy_df.to_csv(REPORT_DIR / "day3_policy_comparison.csv")
print("\nTest-set comparison of decision policies:")
print(policy_df.round(4).to_string())
chosen_cost = policy_df.iloc[0]["cost_per_applicant"]
for name in ("Default threshold (0.50)", "Approve everyone"):
    base = policy_df.loc[name, "cost_per_applicant"]
    print(f"Cost reduction vs '{name}': {(1 - chosen_cost / base):.1%}")

# Sensitivity: how does the best threshold move if the cost ratio changes?
sens_rows = []
for r in SENSITIVITY_RATIOS:
    c = cost_curve(y_val, p_val, r, 1.0)
    t = float(grid[np.argmin(c)])
    pred = (p_test >= t).astype(int)
    sens_rows.append({"FN_to_FP_cost_ratio": r, "theory_threshold": 1 / (1 + r),
                      "val_optimal_threshold": t, **describe(pred, y_test)})
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(REPORT_DIR / "day3_threshold_sensitivity.csv", index=False)
print("\nSensitivity to the cost ratio (test performance):")
print(sens_df.round(3).to_string(index=False))

# Figures: cost curve, confusion matrices
plt.figure(figsize=(7.5, 4.5))
plt.plot(grid, val_costs, label="Validation (used to choose)")
plt.plot(grid, test_costs, ls="--", label="Test (for reference)")
plt.axvline(THRESHOLD, color="red", lw=1, label=f"Chosen threshold = {THRESHOLD:.3f}")
plt.axvline(0.5, color="grey", lw=1, ls=":", label="Default 0.50")
plt.xlabel("Decision threshold (reject if predicted default probability >= threshold)")
plt.ylabel("Cost per applicant")
plt.title(f"Business cost by threshold (missed default = {COST_FN:g}x rejected good loan)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "12_cost_curve.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (title, pred) in zip(axes, [("Default threshold 0.50", (p_test >= 0.5).astype(int)),
                                    (f"Chosen threshold {THRESHOLD:.3f}", (p_test >= THRESHOLD).astype(int))]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Approve", "Reject"], yticklabels=["Repaid", "Defaulted"])
    d = describe(pred, y_test)
    ax.set_title(f"{title}\nrecall {d['recall_defaults']:.2f} | precision {d['precision_defaults']:.2f}")
    ax.set_xlabel("Model decision")
    ax.set_ylabel("What actually happened")
plt.tight_layout()
plt.savefig(FIG_DIR / "13_confusion_matrices.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# 5. Risk deciles
# ----------------------------------------------------------------------------
section("5. RISK DECILES (test set)")
dec = pd.DataFrame({"p": p_test, "y": y_test})
dec["decile"] = pd.qcut(dec["p"].rank(method="first"), 10, labels=range(1, 11))
dec_df = dec.groupby("decile", observed=True).agg(
    applicants=("y", "size"), avg_predicted=("p", "mean"), actual_default_rate=("y", "mean"),
    defaults=("y", "sum")).reset_index()
dec_df["share_of_all_defaults"] = dec_df["defaults"] / dec_df["defaults"].sum()
dec_df = dec_df.sort_values("decile", ascending=False).reset_index(drop=True)  # riskiest first
dec_df["cumulative_defaults_captured"] = dec_df["share_of_all_defaults"].cumsum()
dec_df.to_csv(REPORT_DIR / "day3_decile_table.csv", index=False)
print("Decile 10 = riskiest 10% of applicants")
print(dec_df.round(3).to_string(index=False))

plt.figure(figsize=(8, 4.5))
x = np.arange(len(dec_df))
plt.bar(x - 0.2, dec_df["avg_predicted"], 0.4, label="Predicted")
plt.bar(x + 0.2, dec_df["actual_default_rate"], 0.4, label="Actual")
plt.xticks(x, dec_df["decile"])
plt.xlabel("Risk decile (10 = riskiest)")
plt.ylabel("Default rate")
plt.title("Predicted vs actual default rate by risk decile (test)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "14_decile_chart.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# 6. Odds ratios with bootstrap confidence intervals
# ----------------------------------------------------------------------------
section("6. ODDS RATIOS (bootstrap confidence intervals)")
from preprocessing import CATEGORICAL_FEATURES, FLAG_FEATURES, NUMERIC_FEATURES  # noqa: E402

UNIT_EFFECTS = {   # feature: (size of the change in original units, plain-English label)
    "person_age": (1.0, "+1 year of age"),
    "person_emp_length": (1.0, "+1 year of employment"),
    "cb_person_cred_hist_length": (1.0, "+1 year of credit history"),
    "loan_int_rate": (1.0, "+1 percentage point of interest rate"),
    "loan_percent_income": (0.10, "+10 points of loan-to-income"),
    "log_income": (np.log(2), "income doubles"),
    "log_loan_amnt": (np.log(2), "loan amount doubles"),
}
# Turn each coefficient (per 1 standard deviation for numeric features) into the change
# in log-odds for a human-sized step.
scale, labels = [], []
for name in pre.feature_names_:
    if name in NUMERIC_FEATURES:
        size, label = UNIT_EFFECTS.get(name, (None, "+1 standard deviation"))
        scale.append(1.0 if size is None else size / float(pre.std_[name]))
        labels.append(label)
    elif name in FLAG_FEATURES:
        scale.append(1.0)
        labels.append("flag = 1 (vs 0)")
    else:
        col, val = name.split("=", 1)
        scale.append(1.0)
        labels.append(f"{val} (vs {pre.categories_[col][0]})")
scale = np.array(scale)

boot = []
for i in range(N_BOOTSTRAP):
    rg = np.random.default_rng(SEED + i)
    idx = rg.integers(0, len(y_train), len(y_train))
    Xb, yb = X_train[idx], y_train[idx]
    if config["resample"]:
        Xb, yb = resample(Xb, yb, config["resample"], SEED + i)
    m = LogisticRegressionScratch(penalty=config["penalty"], C=config["C"],
                                  class_weight=config["class_weight"],
                                  max_iter=3000, tol=1e-5).fit(Xb, yb, init=(model.coef_, model.intercept_))
    boot.append(m.coef_)
    if (i + 1) % 50 == 0:
        print(f"  bootstrap {i + 1}/{N_BOOTSTRAP}")
boot = np.array(boot)

point = np.exp(model.coef_ * scale)
boot_or = np.exp(boot * scale)
or_df = pd.DataFrame({
    "feature": pre.feature_names_,
    "effect": labels,
    "odds_ratio": point,
    "ci_low": np.percentile(boot_or, 2.5, axis=0),
    "ci_high": np.percentile(boot_or, 97.5, axis=0),
    "same_sign_in_bootstrap": np.maximum((boot > 0).mean(0), (boot < 0).mean(0)),
})
or_df["significant"] = (or_df["ci_low"] > 1) | (or_df["ci_high"] < 1)
or_df = or_df.sort_values("odds_ratio", ascending=False).reset_index(drop=True)
or_df.to_csv(REPORT_DIR / "day3_odds_ratios.csv", index=False)
print("\nOdds ratios (above 1 = higher default odds; the 95% interval excludes 1 when 'significant'):")
print(or_df.round(3).to_string(index=False))
print("\nCaution: each odds ratio holds the other features constant. Features that overlap")
print("(interest rate vs loan grade, loan amount vs loan-to-income) share credit, so read")
print("them together. Regularisation also shrinks coefficients slightly toward 0.")

plot_df = or_df.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(plot_df))))
# red = raises default odds, blue = lowers them, grey = interval includes 1 (no clear effect)
colors = np.where(~plot_df["significant"], "#999999",
                  np.where(plot_df["odds_ratio"] > 1, "#C44E52", "#4C72B0"))
ax.hlines(plot_df.index, plot_df["ci_low"], plot_df["ci_high"], color=colors, lw=2)
ax.scatter(plot_df["odds_ratio"], plot_df.index, color=colors, zorder=3)
ax.axvline(1, color="k", lw=1)
ax.set_xscale("log")
ticks = [t for t in [0.25, 0.5, 1, 2, 4, 8, 16] if plot_df["ci_low"].min() / 1.3 <= t <= plot_df["ci_high"].max() * 1.3]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks])
ax.minorticks_off()
ax.set_yticks(plot_df.index)
ax.set_yticklabels([f"{f}  [{e}]" for f, e in zip(plot_df["feature"], plot_df["effect"])], fontsize=8)
ax.set_xlabel("Odds ratio (log scale) with 95% bootstrap interval")
ax.set_title("What drives default odds? (grey = no clear effect)")
plt.tight_layout()
plt.savefig(FIG_DIR / "15_odds_ratios.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# 7. Save for the app
# ----------------------------------------------------------------------------
section("7. SAVE")
joblib.dump(platt, MODEL_DIR / "calibrator.joblib")
with open(MODEL_DIR / "decision_config.json", "w") as f:
    json.dump({
        "threshold": THRESHOLD,
        "cost_false_negative": COST_FN,
        "cost_false_positive": COST_FP,
        "platt_a": platt.a_,
        "platt_b": platt.b_,
        "test_metrics_calibrated": {k: float(v) for k, v in rows["Logistic (scratch), calibrated"].items()},
    }, f, indent=2)
print(f"Saved calibrator.joblib and decision_config.json to {MODEL_DIR}")
print("Reports saved to reports/ and figures 10-15 to reports/figures/")
print("\nEvaluation complete. Next: the Streamlit app and final write-up.")
