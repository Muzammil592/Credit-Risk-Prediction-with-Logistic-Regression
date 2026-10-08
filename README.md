# Credit Risk Prediction with Logistic Regression

Can we predict which loan applicants are likely to default, and can we explain *why* in a way a human (or a risk manager) would actually trust?

That's the question behind this project. I'm building a credit default model from the ground up: cleaning the data, writing logistic regression **from scratch in NumPy**, checking it against scikit-learn, and handling the practical headaches that show up in real lending problems, like imbalanced classes and overlapping features.

I picked logistic regression on purpose. In credit risk it's still the industry standard because it's fast, stable, and, most importantly, **explainable**. A model that says "approve" or "decline" is only useful if someone can ask "why?" and get a real answer.

---

## Where things stand

This project is still growing, so here's an honest snapshot.

**Done**
- Data cleaning, exploration, and feature engineering
- A clean, leak-free train/validation/test split
- Logistic regression written from scratch in NumPy, with L1 and L2 regularization and class weights
- Validation of my implementation against scikit-learn
- Regularization tuning with cross-validation
- A side-by-side comparison of ways to handle class imbalance

**Coming next**
- Full evaluation on the held-out test set (ROC-AUC, precision-recall, confusion matrix)
- Choosing a decision threshold based on business cost instead of the default 0.5
- Calibration checks, so a predicted 20% really means about 20%
- Plain-English interpretation of the coefficients as odds ratios
- A tree-based baseline, to see what extra complexity actually buys
- A small Streamlit app where you can enter applicant details and get a default probability

I update this README as pieces get finished.

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

A quick honesty note: this dataset looks synthetic or heavily simplified rather than real bank data. I'm treating it as a realistic practice problem, so this project shows the *method* at work, not how a real lender's portfolio would behave.

The data isn't included in this repo. Download it from the Kaggle link above and put the CSV in a `dataset/` folder (see [Getting started](#getting-started)).

---

## How it works

### 1. Cleaning and exploring the data

The raw data isn't ready to model. A few things needed attention:

- Duplicate rows
- Missing values in `person_emp_length` and `loan_int_rate`
- Impossible values, like applicants over 100 years old or with 120+ years of employment
- Employment or credit history longer than the person could possibly have been an adult
- Heavily skewed income, with a few extreme outliers

What I did about it:

- Removed exact duplicates
- Dropped records that are physically impossible (age over 100, employment over 60 years, or employment/credit history longer than `age - 14`)
- Created "was this value missing?" flags *before* filling anything in, since missingness can carry information on its own
- Split the data **70/15/15 into train/validation/test**, stratified so each split keeps the same default rate
- Filled missing interest rates with the **median for that loan grade** (rate and grade are tightly linked), and missing employment length with the overall median. **All of these values come from the training set only**, so nothing leaks in from validation or test
- Ran all exploratory analysis on the training split only, to keep the other splits untouched

I also engineered a handful of features:

| Feature | Why |
|---------|-----|
| `log_income`, `log_loan_amnt` | Tame the heavy right skew so outliers don't dominate |
| `emp_length_to_age`, `cred_hist_to_age` | Stability and maturity relative to life stage |
| `high_burden` | Flag for loans above 30% of income |
| `interest_cost_to_income` | Rough measure of repayment pressure |
| `age_group` | Lets the model capture non-linear age effects |
| `emp_length_missing`, `int_rate_missing` | Missing-value flags |

All charts from this stage are saved in [`reports/figures/`](reports/figures/).

### 2. Logistic regression, built by hand

Instead of just calling a library, I wrote the model myself in NumPy so I understand every moving part.

- **The math:** weighted log-loss with optional L1 or L2 regularization, scaled the same way as scikit-learn so the `C` parameter means the same thing in both
- **Gradient check:** I compare my hand-derived gradient against a numerical (finite-difference) one to prove the maths is right
- **Optimizer:** gradient descent with Nesterov momentum and adaptive restarts, plus an automatic step size. Several features here are strongly related (income, loan amount, and loan-to-income ratio), which makes plain gradient descent painfully slow, and momentum fixes that. The code lets you switch it off to see the difference
- **L1 regularization:** done with a proximal step (soft-thresholding), which is what pushes some coefficients to exactly zero
- **Class weights:** supported directly, using the same "balanced" formula as scikit-learn
- **Validation:** I fit scikit-learn's version on the same data and compare coefficients, predicted probabilities, and AUC

### 3. Tuning and imbalance

- **Multicollinearity:** a VIF check shows which features overlap (interest rate and loan grade are the usual suspects)
- **Regularization:** 5-fold cross-validation on the training set across a grid of `C` values for both L1 and L2. I pick the simplest model that's statistically as good as the best one (the "one standard error rule"), which favors stability over squeezing out the last fraction of a point
- **Class imbalance:** defaults are the minority class, so I compare four approaches on the validation set: no adjustment, class weights, random oversampling, and random undersampling. Rebalancing tends to distort predicted probabilities, which is one reason a calibration check is on the to-do list

The test set stays locked away until the final evaluation.

---

## Project structure

```
credit-risk-logistic-regression/
├── dataset/                      # Raw Kaggle CSV (not committed, see below)
├── data/
│   └── processed/                # train.csv, val.csv, test.csv, cleaning_params.json
├── models/                       # Saved model, preprocessor, and config
├── reports/
│   ├── figures/                  # EDA and modeling charts
│   └── *.csv                     # VIF, CV results, imbalance comparison, coefficients
├── src/
│   ├── data_cleaning.py          # Cleaning, EDA, feature engineering, split
│   ├── preprocessing.py          # Scaling and encoding (fit on train only)
│   ├── logreg_scratch.py         # Logistic regression from scratch in NumPy
│   └── modeling.py               # Validation, tuning, imbalance comparison, saving
├── requirements.txt
└── README.md
```

More scripts and the Streamlit app will land here as they're built.

---

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/credit-risk-logistic-regression.git
cd credit-risk-logistic-regression
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the data**

Grab `credit_risk_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) and place it at:
```
dataset/credit_risk_dataset.csv
```

**4. Clean the data and build the splits**
```bash
python src/data_cleaning.py
```
This prints a data quality audit, saves the charts to `reports/figures/`, and writes the cleaned train, validation, and test files to `data/processed/`.

**5. Train and compare the models**
```bash
python src/modeling.py
```
This runs the gradient check, compares my implementation with scikit-learn, tunes regularization, compares imbalance strategies, and saves the chosen model to `models/`. It takes a few minutes because the cross-validation fits the from-scratch model many times.

---

## Decisions worth knowing about

A few deliberate choices, in case you're reading the code and wondering why:

- **Anything learned from data is fit on train only.** Imputation values, scaling statistics, and category lists all come from the training split. Otherwise validation and test information leaks in.
- **Exploration uses the training split only.** The test set stays untouched until the very end.
- **Resampling touches training data only.** Over- and undersampling are never applied to validation or test.
- **Reproducibility.** Everything uses a fixed random seed (`42`), and the cleaning decisions are saved in `cleaning_params.json`.
- **Simple over fancy.** When two settings perform about the same, I go with the simpler, more regularized one.

---

## Tech stack

Python, NumPy, pandas, scikit-learn, matplotlib, seaborn, and Streamlit.

---

## Limitations

- The dataset appears to be synthetic or simplified, so this shouldn't be read as performance on real lending data.
- It has no time dimension, so I can't test how the model holds up as conditions change.
- It covers only loans that were issued, so there's no information on applicants who were rejected. In real credit modeling this selection bias is a significant issue.
- A real deployment would also need fairness testing and regulatory review, which are beyond the scope of this project.

---

## Acknowledgments

Dataset: [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) by Lao Tse on Kaggle. Please check the dataset page for its license terms.

---

## Contact

Questions or feedback? Open an issue or reach out on [LinkedIn](https://www.linkedin.com/in/muzammil-abbas-404b80266/) or by [email](mailto:muzammilabbas779677@gmail.com).