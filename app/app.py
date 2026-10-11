"""
Loan Default Risk Estimator: a Streamlit front end for the from-scratch logistic regression.

Run from the project root:
    streamlit run app/app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from scoring import contributions, load_artifacts, probability_curve, score_applicant

st.set_page_config(page_title="Loan Default Risk Estimator", page_icon="💳", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


art = get_artifacts()
cats = art["preprocessor"]["categories"]
threshold = art["decision"]["threshold"]
cost_fn, cost_fp = art["decision"]["cost_false_negative"], art["decision"]["cost_false_positive"]
typical = art["typical_ranges"]

# ----------------------------------------------------------------------------
# Friendly names for the explanation chart
# ----------------------------------------------------------------------------
FEATURE_LABELS = {
    "person_age": "Age",
    "log_income": "Income",
    "person_emp_length": "Employment length",
    "log_loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan as share of income",
    "cb_person_cred_hist_length": "Credit history length",
    "emp_length_to_age": "Employment length relative to age",
    "cred_hist_to_age": "Credit history relative to age",
    "interest_cost_to_income": "Interest cost relative to income",
    "high_burden": "Loan above 30% of income",
    "emp_length_missing": "Employment length unknown",
    "int_rate_missing": "Interest rate unknown",
}
CATEGORY_LABELS = {
    "person_home_ownership": "Home",
    "loan_intent": "Purpose",
    "loan_grade": "Loan grade",
    "cb_person_default_on_file": "Prior default",
    "age_group": "Age group",
}
LEVEL_LABELS = {"DEBTCONSOLIDATION": "Debt consolidation", "HOMEIMPROVEMENT": "Home improvement",
                "Y": "Yes", "N": "No"}


def friendly(name):
    if "=" in name:
        col, level = name.split("=", 1)
        return f"{CATEGORY_LABELS.get(col, col)}: {LEVEL_LABELS.get(level, level.title())}"
    return FEATURE_LABELS.get(name, name)


def level_label(level):
    return LEVEL_LABELS.get(level, level.title() if level.isupper() and len(level) > 1 else level)


# ----------------------------------------------------------------------------
# Example applicants (the widgets reset when the example changes)
# ----------------------------------------------------------------------------
PRESETS = {
    "Typical applicant": dict(age=28, income=55000, home="RENT", emp=4, intent="EDUCATION",
                              grade="B", amount=10000, rate=11.0, prior="N", hist=5),
    "Lower-risk applicant": dict(age=38, income=95000, home="MORTGAGE", emp=10, intent="HOMEIMPROVEMENT",
                                 grade="A", amount=8000, rate=7.5, prior="N", hist=12),
    "Higher-risk applicant": dict(age=24, income=32000, home="RENT", emp=1, intent="MEDICAL",
                                  grade="D", amount=15000, rate=15.5, prior="Y", hist=3),
}


def pick(options, wanted):
    return options.index(wanted) if wanted in options else 0


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("💳 Loan Default Risk Estimator")
st.caption(
    "Enter an applicant and a loan. A logistic regression model, written from scratch in NumPy, "
    "estimates the chance of default and shows what is driving it."
)

# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Applicant and loan")
    preset_name = st.selectbox("Start from an example", list(PRESETS))
    ps = PRESETS[preset_name]
    k = preset_name  # widget keys include the example name, so changing it resets the widgets

    st.subheader("About the applicant")
    age = st.number_input("Age", min_value=18, max_value=100, value=int(ps["age"]), step=1, key=f"age_{k}")
    income = st.number_input("Annual income", min_value=1000, max_value=10_000_000,
                             value=int(ps["income"]), step=1000, key=f"income_{k}")
    home_opts = cats["person_home_ownership"]
    home = st.selectbox("Home ownership", home_opts, index=pick(home_opts, ps["home"]),
                        format_func=level_label, key=f"home_{k}")
    emp_unknown = st.checkbox("Employment length unknown", value=False, key=f"empu_{k}")
    emp = st.number_input("Years employed", min_value=0.0, max_value=60.0, value=float(ps["emp"]),
                          step=1.0, disabled=emp_unknown, key=f"emp_{k}")
    hist = st.number_input("Credit history (years)", min_value=0, max_value=60, value=int(ps["hist"]),
                           step=1, key=f"hist_{k}")
    prior_opts = cats["cb_person_default_on_file"]
    prior = st.selectbox("Previous default on file?", prior_opts, index=pick(prior_opts, ps["prior"]),
                         format_func=level_label, key=f"prior_{k}")

    st.subheader("About the loan")
    intent_opts = cats["loan_intent"]
    intent = st.selectbox("Loan purpose", intent_opts, index=pick(intent_opts, ps["intent"]),
                          format_func=level_label, key=f"intent_{k}")
    grade_opts = cats["loan_grade"]
    grade = st.selectbox("Loan grade (A best, G worst)", grade_opts, index=pick(grade_opts, ps["grade"]),
                         key=f"grade_{k}")
    amount = st.number_input("Loan amount", min_value=500, max_value=1_000_000, value=int(ps["amount"]),
                             step=500, key=f"amount_{k}")
    rate_unknown = st.checkbox("Interest rate not set yet (estimate from grade)", value=False, key=f"rateu_{k}")
    rate = st.number_input("Interest rate (%)", min_value=0.0, max_value=40.0, value=float(ps["rate"]),
                           step=0.1, disabled=rate_unknown, key=f"rate_{k}")

raw = {
    "person_age": age,
    "person_income": income,
    "person_home_ownership": home,
    "person_emp_length": None if emp_unknown else emp,
    "loan_intent": intent,
    "loan_grade": grade,
    "loan_amnt": amount,
    "loan_int_rate": None if rate_unknown else rate,
    "cb_person_default_on_file": prior,
    "cb_person_cred_hist_length": hist,
}

# ----------------------------------------------------------------------------
# Sanity checks (the same rules used to clean the training data)
# ----------------------------------------------------------------------------
problems = []
if not emp_unknown and emp > age - 14:
    problems.append("Years employed can't exceed age minus 14.")
if hist > age - 14:
    problems.append("Credit history can't be longer than age minus 14.")
if problems:
    st.error("Please fix these inputs: " + " ".join(problems))
    st.stop()

# ----------------------------------------------------------------------------
# Score
# ----------------------------------------------------------------------------
prob, x_row, feats = score_applicant(art, raw)
decline = prob >= threshold

outside = []
checks = [("Age", age, "person_age"), ("Income", income, "person_income"),
          ("Loan amount", amount, "loan_amnt"), ("Loan as share of income", feats["loan_percent_income"],
                                                  "loan_percent_income"),
          ("Credit history", hist, "cb_person_cred_hist_length")]
if not emp_unknown:
    checks.append(("Years employed", emp, "person_emp_length"))
if not rate_unknown:
    checks.append(("Interest rate", rate, "loan_int_rate"))
for label, value, col in checks:
    lo, hi = typical[col]
    if value < lo or value > hi:
        outside.append(label)
if outside:
    st.warning(
        "These inputs are unusual compared with the data the model learned from "
        f"({', '.join(outside)}), so treat the estimate with extra caution."
    )

# ----------------------------------------------------------------------------
# Headline result
# ----------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Estimated chance of default", f"{prob:.1%}")
c2.metric("Decision cutoff", f"{threshold:.1%}",
          help=f"Chosen to minimise expected cost, assuming a missed default costs {cost_fn / cost_fp:.0f}x "
               "as much as turning away a good borrower.")
c3.metric("Loan as share of income", f"{feats['loan_percent_income']:.0%}")

if decline:
    st.error(f"**Refer for review / decline.** Estimated risk ({prob:.1%}) is above the {threshold:.1%} cutoff.")
else:
    st.success(f"**Approve.** Estimated risk ({prob:.1%}) is below the {threshold:.1%} cutoff.")

# Risk gauge
fig, ax = plt.subplots(figsize=(8, 1.1))
ax.barh([0], [1], color="#e8e8e8", height=0.5)
ax.barh([0], [prob], color="#C44E52" if decline else "#55A868", height=0.5)
ax.axvline(threshold, color="black", lw=2)
ax.text(threshold, 0.42, f" cutoff {threshold:.1%}", va="bottom", ha="left", fontsize=8)
ax.set_xlim(0, max(0.5, min(1.0, prob * 1.3, 1.0)))
ax.set_ylim(-0.5, 0.8)
ax.set_yticks([])
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ----------------------------------------------------------------------------
# Why this score
# ----------------------------------------------------------------------------
st.subheader("Why this score?")
contrib = contributions(art, x_row)
contrib = contrib[contrib.abs() > 1e-9]
top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)

left, right = st.columns([3, 2])
with left:
    fig, ax = plt.subplots(figsize=(7, 4))
    plot = top.iloc[::-1]
    ax.barh([friendly(n) for n in plot.index], plot.values,
            color=["#C44E52" if v > 0 else "#4C72B0" for v in plot.values])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Effect on the odds of default (log scale)  <- safer | riskier ->")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    raises = [friendly(n) for n, v in top.items() if v > 0][:3]
    lowers = [friendly(n) for n, v in top.items() if v < 0][:3]
    if raises:
        st.markdown("**Pushing risk up:** " + ", ".join(raises))
    if lowers:
        st.markdown("**Pulling risk down:** " + ", ".join(lowers))
    st.caption(
        "Each bar is that factor's contribution to the model's score. Numeric factors are compared with "
        "an average applicant; categories are compared with a baseline option (for example grade A or "
        "mortgage). Some factors overlap, such as interest rate and loan grade, so read them together."
    )

# ----------------------------------------------------------------------------
# What if
# ----------------------------------------------------------------------------
st.subheader("What if the loan were bigger or smaller?")
amounts = np.linspace(500, max(35000, amount * 2), 60)
curve = probability_curve(art, raw, "loan_amnt", amounts)
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(amounts, curve, color="#4C72B0", lw=2)
ax.axhline(threshold, color="black", ls="--", lw=1, label=f"Cutoff {threshold:.1%}")
ax.scatter([amount], [prob], color="#C44E52", zorder=3, label="Current loan")
ax.set_xlabel("Loan amount")
ax.set_ylabel("Estimated chance of default")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("Everything else is held fixed, including the interest rate. A larger loan raises the "
           "loan-to-income ratio, which is what moves the line.")

# ----------------------------------------------------------------------------
# About
# ----------------------------------------------------------------------------
with st.expander("How this works and what it can't do"):
    m = art["test_metrics"]
    info = art["model_info"]
    st.markdown(
        f"""
**The model.** Logistic regression implemented from scratch in NumPy
({info['penalty'].upper() if info['penalty'] else 'no'} regularisation, C = {info['C']}), trained on a public
Kaggle credit risk dataset. Class imbalance was handled with *{info['imbalance_strategy'].lower()}*, and the
probabilities were then calibrated so that "20%" means roughly 20 in 100.

**The cutoff.** {threshold:.1%}, chosen to minimise expected cost on validation data, assuming a missed
default costs {cost_fn / cost_fp:.0f}x as much as turning away a borrower who would have repaid. Those costs are
illustrative assumptions, not a real lender's figures.

**Held-out test performance.** ROC-AUC {m['roc_auc']:.3f}, Gini {m['gini']:.3f}, KS {m['ks']:.3f},
Brier score {m['brier']:.3f}.

**Limits.** The dataset appears synthetic or simplified, contains only approved loans, and has no time
dimension. This is a portfolio demonstration, **not** a tool for real lending decisions, and it has not been
tested for fairness.
        """
    )

st.caption("Educational project. Not financial advice.")
