"""
Day 2: logistic regression from scratch, regularisation, and class imbalance.

Run from the project root (after data_cleaning.py):
    python src/modeling.py

Steps
  1. Preprocess (fit on train only) and check multicollinearity (VIF)
  2. Gradient check: prove the hand-written gradient is correct
  3. Train from scratch and validate against scikit-learn
  4. Tune L1 / L2 regularisation with 5-fold cross-validation (train only)
  5. Compare imbalance strategies: none, class weights, oversampling, undersampling
  6. Save the chosen model for Day 3

The TEST set is not touched today. It is saved for the final evaluation on Day 3.
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from logreg_scratch import LogisticRegressionScratch, gradient_check
from preprocessing import TARGET, Preprocessor, load_splits, vif_table

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
for d in (FIG_DIR, REPORT_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", module="sklearn")   # keep the console readable
SEED = 42
CHOSEN_STRATEGY = None   # set e.g. "Class weights (balanced)" to override the automatic pick
sns.set_theme(style="whitegrid")
SK_VERSION = tuple(int(x) for x in sklearn.__version__.split(".")[:2])


def make_sklearn_logreg(penalty, C=1.0):
    """
    scikit-learn reference model. Handles the penalty= deprecation in sklearn 1.8+.
    L1 uses the saga solver because liblinear also penalises the intercept, which would
    make it a slightly different problem from ours.
    """
    kw = dict(max_iter=10000, tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if SK_VERSION >= (1, 8):
            if penalty is None:
                return LogisticRegression(C=np.inf, **kw)
            if penalty == "l2":
                return LogisticRegression(C=C, **kw)
            return LogisticRegression(l1_ratio=1.0, C=C, solver="saga", **kw)
        if penalty is None:
            return LogisticRegression(penalty=None, **kw)
        if penalty == "l2":
            return LogisticRegression(penalty="l2", C=C, **kw)
        return LogisticRegression(penalty="l1", C=C, solver="saga", **kw)


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def evaluate(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
    }


# ----------------------------------------------------------------------------
# 1. Data and preprocessing
# ----------------------------------------------------------------------------
section("1. PREPROCESSING AND MULTICOLLINEARITY")
train_df, val_df, test_df = load_splits()
pre = Preprocessor().fit(train_df)
X_train, X_val = pre.transform(train_df), pre.transform(val_df)
y_train = train_df[TARGET].to_numpy(int)
y_val = val_df[TARGET].to_numpy(int)
print(f"X_train {X_train.shape} | X_val {X_val.shape} | features: {len(pre.feature_names_)}")
print(f"Default rate: train {y_train.mean():.2%} | val {y_val.mean():.2%}")

vif = vif_table(X_train, pre.feature_names_)
vif.to_csv(REPORT_DIR / "day2_vif.csv", index=False)
print("\nHighest VIFs (above ~5 means overlapping information):")
print(vif.head(8).round(2).to_string(index=False))

# ----------------------------------------------------------------------------
# 2. Gradient check
# ----------------------------------------------------------------------------
section("2. GRADIENT CHECK")
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_train), 500, replace=False)
for pen in ("l2", None):
    err = gradient_check(X_train[idx], y_train[idx], penalty=pen, C=1.0)
    print(f"penalty={pen}: max relative error analytic vs numeric = {err:.2e}")

# ----------------------------------------------------------------------------
# 3. From scratch vs scikit-learn (no regularisation)
# ----------------------------------------------------------------------------
section("3. FROM SCRATCH vs SCIKIT-LEARN (unregularised)")
t0 = time.time()
scratch = LogisticRegressionScratch(penalty=None, max_iter=20000, tol=1e-9).fit(X_train, y_train)
t_scratch = time.time() - t0
plain = LogisticRegressionScratch(penalty=None, accelerated=False, max_iter=3000, tol=1e-9)
plain.fit(X_train, y_train)

sk = make_sklearn_logreg(None).fit(X_train, y_train)

p_scratch = scratch.predict_proba(X_val)
p_sk = sk.predict_proba(X_val)[:, 1]
coef_diff = np.abs(scratch.coef_ - sk.coef_[0]).max()
prob_diff = np.abs(p_scratch - p_sk).max()
print(f"Accelerated GD: {scratch.n_iter_} iterations, {t_scratch:.1f}s (step size {scratch.lr_:.3f})")
print(f"Max |coefficient difference|   : {coef_diff:.2e}")
print(f"Max |predicted prob difference|: {prob_diff:.2e}")
print(f"Intercept: scratch {scratch.intercept_:.4f} | sklearn {sk.intercept_[0]:.4f}")
print(f"Validation ROC-AUC: scratch {roc_auc_score(y_val, p_scratch):.4f} | "
      f"sklearn {roc_auc_score(y_val, p_sk):.4f}")
print(f"\nFor comparison, plain GD after {plain.n_iter_} iterations: max |coef diff| = "
      f"{np.abs(plain.coef_ - sk.coef_[0]).max():.2e}, final loss {plain.loss_history_[-1]:.6f} "
      f"(accelerated: {scratch.loss_history_[-1]:.6f})")

# Figures: convergence + coefficient agreement
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
floor = min(scratch.loss_history_[-1], plain.loss_history_[-1])
axes[0].plot(np.array(plain.loss_history_) - floor + 1e-12, label="Plain gradient descent")
axes[0].plot(np.array(scratch.loss_history_) - floor + 1e-12, label="Accelerated (Nesterov)")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Training loss above the optimum")
axes[0].set_title("Convergence speed")
axes[0].legend()
lim = [min(sk.coef_[0].min(), scratch.coef_.min()), max(sk.coef_[0].max(), scratch.coef_.max())]
axes[1].scatter(sk.coef_[0], scratch.coef_, alpha=0.8)
axes[1].plot(lim, lim, "r--", lw=1)
axes[1].set_xlabel("scikit-learn coefficient")
axes[1].set_ylabel("From-scratch coefficient")
axes[1].set_title("Coefficients: scratch vs scikit-learn")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_scratch_vs_sklearn.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# 4. Regularisation: 5-fold CV on TRAIN, using the from-scratch model
# ----------------------------------------------------------------------------
section("4. REGULARISATION TUNING (5-fold CV, ROC-AUC)")
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rows = []
for penalty in ("l2", "l1"):
    for C in C_GRID:
        scores = []
        for tr, te in skf.split(X_train, y_train):
            m = LogisticRegressionScratch(penalty=penalty, C=C, max_iter=3000, tol=1e-6)
            m.fit(X_train[tr], y_train[tr])
            scores.append(roc_auc_score(y_train[te], m.predict_proba(X_train[te])))
        cv_rows.append({"penalty": penalty, "C": C,
                        "cv_auc_mean": np.mean(scores), "cv_auc_std": np.std(scores)})
        print(f"  {penalty}  C={C:<7} AUC = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(REPORT_DIR / "day2_cv_results.csv", index=False)

# Pick the SIMPLEST model within one standard error of the best ("1-SE rule"):
# among settings whose CV score is statistically close to the top, take the smallest C
# (strongest regularisation). Ties between L2 and L1 go to L2, which behaves better
# when features overlap.
def one_se_choice(grp):
    top = grp.loc[grp["cv_auc_mean"].idxmax()]
    se = top["cv_auc_std"] / np.sqrt(skf.get_n_splits())
    ok = grp[grp["cv_auc_mean"] >= top["cv_auc_mean"] - se]
    return top, ok.sort_values("C").iloc[0]


chosen = None
for pen in ("l2", "l1"):
    top, cand = one_se_choice(cv_df[cv_df["penalty"] == pen])
    print(f"  {pen}: best C={top['C']} (AUC {top['cv_auc_mean']:.4f}) -> 1-SE pick C={cand['C']} "
          f"(AUC {cand['cv_auc_mean']:.4f})")
    if chosen is None or cand["cv_auc_mean"] > chosen["cv_auc_mean"] + 1e-4:
        chosen = cand
BEST_PENALTY, BEST_C = chosen["penalty"], float(chosen["C"])
print(f"\nChosen: {BEST_PENALTY} with C={BEST_C}")

plt.figure(figsize=(7, 4.5))
for pen, grp in cv_df.groupby("penalty"):
    plt.errorbar(grp["C"], grp["cv_auc_mean"], yerr=grp["cv_auc_std"], marker="o",
                 capsize=3, label=pen.upper())
plt.xscale("log")
plt.xlabel("C (smaller = stronger regularisation)")
plt.ylabel("CV ROC-AUC")
plt.title("Regularisation tuning (from-scratch model)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "08_regularisation_cv.png", dpi=150)
plt.close()

# Confirm the chosen setting against scikit-learn too
sk_pen = make_sklearn_logreg(BEST_PENALTY, BEST_C).fit(X_train, y_train)
scr_pen = LogisticRegressionScratch(penalty=BEST_PENALTY, C=BEST_C,
                                    max_iter=20000, tol=1e-9).fit(X_train, y_train)
print(f"\nChosen setting, validation ROC-AUC: scratch "
      f"{roc_auc_score(y_val, scr_pen.predict_proba(X_val)):.4f} | sklearn "
      f"{roc_auc_score(y_val, sk_pen.predict_proba(X_val)[:, 1]):.4f}")
print(f"Max |coefficient difference| scratch vs sklearn: "
      f"{np.abs(scr_pen.coef_ - sk_pen.coef_[0]).max():.2e}")
if BEST_PENALTY == "l1":
    print(f"Coefficients shrunk to exactly zero (scratch): {(scr_pen.coef_ == 0).sum()} of {len(scr_pen.coef_)}")

# ----------------------------------------------------------------------------
# 5. Class imbalance strategies (validation set)
# ----------------------------------------------------------------------------
section("5. CLASS IMBALANCE STRATEGIES")


def resample(X, y, mode, seed=SEED):
    """Random over/under-sampling. Only ever applied to TRAINING data."""
    rg = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if mode == "oversample":
        pos = rg.choice(pos, size=len(neg), replace=True)
    elif mode == "undersample":
        neg = rg.choice(neg, size=len(pos), replace=False)
    keep = rg.permutation(np.concatenate([pos, neg]))
    return X[keep], y[keep]


strategies = {
    "No adjustment": dict(cw=None, resample=None),
    "Class weights (balanced)": dict(cw="balanced", resample=None),
    "Random oversampling": dict(cw=None, resample="oversample"),
    "Random undersampling": dict(cw=None, resample="undersample"),
}
imb_rows, imb_models = [], {}
for name, cfg in strategies.items():
    Xs, ys = X_train, y_train
    if cfg["resample"]:
        Xs, ys = resample(X_train, y_train, cfg["resample"])
    m = LogisticRegressionScratch(penalty=BEST_PENALTY, C=BEST_C, class_weight=cfg["cw"],
                                  max_iter=20000, tol=1e-9).fit(Xs, ys)
    proba = m.predict_proba(X_val)
    row = {"strategy": name, **evaluate(y_val, proba), "mean_pred_prob": proba.mean()}
    imb_rows.append(row)
    imb_models[name] = (m, cfg)

imb_df = pd.DataFrame(imb_rows)
imb_df.to_csv(REPORT_DIR / "day2_imbalance_results.csv", index=False)
print(f"True validation default rate: {y_val.mean():.3f}")
print(imb_df.round(4).to_string(index=False))
print("\nNote: 'mean_pred_prob' far above the true default rate means the probabilities are")
print("inflated by the rebalancing. Ranking (AUC) is fine, but calibration will need a fix on Day 3.")

plot_df = imb_df.melt(id_vars="strategy", value_vars=["roc_auc", "pr_auc", "recall", "precision"],
                      var_name="metric", value_name="score")
plt.figure(figsize=(10, 4.5))
sns.barplot(data=plot_df, x="metric", y="score", hue="strategy")
plt.title("Imbalance strategies on the validation set (threshold = 0.5)")
plt.ylim(0, 1)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "09_imbalance_comparison.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# 6. Choose and save the model
# ----------------------------------------------------------------------------
section("6. SAVE THE CHOSEN MODEL")
# The ranking metrics barely move between strategies, so we choose on validation PR-AUC
# (the more informative metric when defaults are the minority class).
winner = CHOSEN_STRATEGY or imb_df.loc[imb_df["pr_auc"].idxmax(), "strategy"]
final_model, final_cfg = imb_models[winner]
print(f"Chosen strategy: {winner}" + ("" if CHOSEN_STRATEGY else " (best validation PR-AUC)"))

coef_df = pd.DataFrame({
    "feature": pre.feature_names_,
    "coefficient": final_model.coef_,
    "odds_ratio_per_1sd_or_flag": np.exp(final_model.coef_),
}).sort_values("coefficient", key=abs, ascending=False)
coef_df.to_csv(REPORT_DIR / "day2_coefficients.csv", index=False)
print("\nTop 10 coefficients by size (numeric features are per 1 standard deviation):")
print(coef_df.head(10).round(3).to_string(index=False))

joblib.dump(pre, MODEL_DIR / "preprocessor.joblib")
joblib.dump(final_model, MODEL_DIR / "logreg_scratch.joblib")
with open(MODEL_DIR / "model_config.json", "w") as f:
    json.dump({
        "penalty": BEST_PENALTY,
        "C": BEST_C,
        "imbalance_strategy": winner,
        "class_weight": final_cfg["cw"],
        "resample": final_cfg["resample"],
        "seed": SEED,
        "features": pre.feature_names_,
        "validation": {k: float(v) for k, v in
                       imb_df[imb_df.strategy == winner].iloc[0].drop("strategy").items()},
    }, f, indent=2)

print(f"\nSaved model, preprocessor and config to {MODEL_DIR}")
print("Day 2 complete. Next: full evaluation, threshold selection and calibration on the test set (Day 3).")
