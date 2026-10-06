# Credit Risk Prediction with Logistic Regression

Can we predict which loan applicants are likely to default, and can we explain *why* in a way a human (or a risk manager) would actually trust?

This is a project where I build a credit default model from the ground up. I clean the data, build logistic regression **from scratch in NumPy**, check it against scikit-learn, pick a decision threshold based on business cost instead of defaulting to 0.5, and wrap it all in a small app.

I picked logistic regression on purpose. In real credit risk work it's still the industry standard because it's fast, stable, and, most importantly, **explainable**. A model that says "approve" or "decline" is only useful if someone can ask "why?" and get a real answer.

---

## Project status

| Day | Focus | Status |
|-----|-------|--------|
| 1 | Data cleaning, EDA, feature engineering, train/val/test split | Done |
| 2 | Logistic regression from scratch, regularization, class imbalance | In progress |
| 3 | Evaluation, threshold selection, calibration, interpretation | Planned |
| 4 | Streamlit app, final write-up, cleanup | Planned |

I'm updating this README as each day wraps up, so the results below fill in over time.

---

## The dataset

I'm using the [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) from Kaggle: about 32,500 loan records with 12 columns and a binary target, `loan_status` (1 = default, 0 = repaid).

| Feature | What it means |
|---------|---------------|
| `person_age` | Applicant's age |
| `person_income` | Annual income |
| `person_home_ownership` | Rent, own, mortgage, or other |
| `person_emp_length` | Years of employment |
| `loan_intent` | Why they want the loan (education, medical, etc.) |
| `loan_grade` | Lender-assigned grade, A (best) to G (worst) |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate |
| `loan_percent_income` | Loan amount as a share of income |
| `cb_person_default_on_file` | Prior default on record (Y/N) |
| `cb_person_cred_hist_length` | Length of credit history in years |

A quick honesty note: this dataset looks synthetic or heavily simplified rather than real bank data. I'm treating it as a realistic practice problem, so the numbers here show the *method* working, not how a real lender's portfolio would behave.

The data isn't included in this repo. Download it from the Kaggle link above and put the CSV in a `dataset/` folder (details in [Getting started](#getting-started)).

---

## What have we done yet? 
## Cleaning and exploring the data

The raw data isn't ready to model. Here's what I found and what I did about it.

**Problems I found**
- Duplicate rows
- Missing values in `person_emp_length` and `loan_int_rate`
- Impossible values, like applicants over 100 years old or with 120+ years of employment
- Employment or credit history longer than the person could possibly have been an adult
- Heavily skewed income, with a few extreme outliers

**What I did**
- Removed exact duplicates
- Dropped records that are physically impossible (age over 100, employment over 60 years, or employment/credit history longer than `age - 14`)
- Created "was this value missing?" flags *before* filling anything in, because missingness can carry information on its own
- Split the data **70/15/15 into train/validation/test**, stratified so each split keeps the same default rate
- Filled missing interest rates with the **median for that loan grade**, since rate and grade are tightly linked. Missing employment length got the overall median. **All of these values come from the training set only**, so no information leaks from validation or test
- Ran all EDA on the training split only, to keep the validation and test sets untouched

**Features I added**

| Feature | Why |
|---------|-----|
| `log_income`, `log_loan_amnt` | Tame the heavy right skew so the model isn't dominated by outliers |
| `emp_length_to_age` | Employment stability relative to life stage |
| `cred_hist_to_age` | How established the applicant's credit history is |
| `high_burden` | Flag for loans above 30% of income |
| `interest_cost_to_income` | Rough measure of repayment pressure |
| `age_group` | Lets the model capture non-linear age effects |
| `emp_length_missing`, `int_rate_missing` | Missing-value flags |

**Key findings** *(fill in after running the script)*
- Overall default rate: **XX%**
- Rows after cleaning: **XX,XXX** (from 32,581 raw)
- Default rate climbs sharply as loan grade worsens (A: **XX%** to G: **XX%**)
- Biggest correlations with default: **[feature]**, **[feature]**, **[feature]**
- `loan_grade`, `loan_int_rate`, and `loan_percent_income` overlap a lot, so I'll check multicollinearity (VIF) in the modeling stage

All figures are saved in [`reports/figures/`](reports/figures/).


## Project structure

```
credit-risk-logistic-regression/
├── dataset/                      # Raw Kaggle CSV (not committed, see below)
├── data/
│   └── processed/                # train.csv, val.csv, test.csv, cleaning_params.json
├── reports/
│   └── figures/                  # EDA charts
├── src/
│   └── data_cleaning.py          # Day 1: cleaning, EDA, features, split
├── requirements.txt
└── README.md

## Getting started

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download the data**

Grab `credit_risk_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) and place it at:
```
dataset/credit_risk_dataset.csv
```

**3. Run the pipeline**
```bash
python src/data_cleaning.py
```

This prints a data quality audit to the console, saves the EDA charts to `reports/figures/`, and writes the cleaned train, validation, and test files to `data/processed/`.

---

## Decisions worth knowing about

A few choices I made deliberately, in case you're reading the code and wondering why:

- **Imputation is fit on train only.** Computing medians on the full dataset would quietly leak validation and test information into training.
- **EDA uses the training split only.** The test set stays untouched until the very end.
- **Encoding and scaling come later, also fit on train only.** Same reason.
- **Reproducibility.** Everything uses a fixed random seed (`42`), and every cleaning decision is saved in `cleaning_params.json`.

---

## Tech stack

Python, NumPy, pandas, scikit-learn, matplotlib, seaborn, statsmodels, and Streamlit.


## Acknowledgments

Dataset: [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) by Lao Tse on Kaggle. Please check the dataset page for its license terms.
