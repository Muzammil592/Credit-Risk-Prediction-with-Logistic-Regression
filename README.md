# Credit Risk Prediction with Logistic Regression

Can we predict which loan applicants are likely to default, and can we explain *why* in a way a human (or a risk manager) would actually trust?

That's the question behind this project. I built a credit default model from the ground up: cleaning the data, writing logistic regression **from scratch in NumPy**, checking it against scikit-learn, and handling the practical headaches that show up in real lending problems, like imbalanced classes, overlapping features, and decisions that cost real money. It ends with an interactive app where you can score an applicant and see what's driving the result.

I picked logistic regression on purpose. In credit risk it's still the industry standard because it's fast, stable, and, most importantly, **explainable**. A model that says "approve" or "decline" is only useful if someone can ask "why?" and get a real answer.

---

## Where things stand

**Done**
- Data cleaning, exploration, and feature engineering
- A clean, leak-free train/validation/test split
- Logistic regression written from scratch in NumPy, with L1 and L2 regularization and class weights
- Validation of my implementation against scikit-learn
- Regularization tuning with cross-validation
- A side-by-side comparison of ways to handle class imbalance
- Probability calibration, so a predicted 20% really means about 20%
- A decision threshold chosen by business cost instead of the default 0.5
- Evaluation on the held-out test set, including a gradient boosting baseline for comparison
- Plain-English interpretation of the model as odds ratios with confidence intervals
- An interactive Streamlit app that scores an applicant and explains the result

**Ideas for going further**
- Deploying the app as a public live demo
- Fairness checks across applicant groups
- Time-based validation, if a dataset with dates becomes available

---

## The app

The Streamlit app lets you describe an applicant and a loan, then shows:

- **The estimated chance of default**, as a calibrated probability
- **A decision** (approve, or refer for review) based on the cost-optimized cutoff
- **Why this score?**, a chart of which factors pushed the risk up or down, in plain language
- **A what-if chart** showing how the risk changes as the loan amount changes
- **Guardrails:** impossible inputs (like employment longer than the applicant's adult life) are blocked, and unusual inputs trigger a warning that the model hasn't seen many people like this

<!-- Add 2-3 screenshots here, for example:
![Higher-risk applicant with explanation chart](docs/app_screenshot.png)
-->

<!-- If you deploy it, add the link here:
**Live demo:** https://your-app-name.streamlit.app
-->

Run it yourself with `streamlit run app/app.py` (see [Getting started](#getting-started)).

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
- **Class imbalance:** defaults are the minority class, so I compare four approaches on the validation set: no adjustment, class weights, random oversampling, and random undersampling. Rebalancing tends to distort predicted probabilities, which is exactly why the next step exists

### 4. Evaluation, calibration, and decisions

A model that ranks applicants well isn't automatically a model you can make decisions with. This stage closes that gap.

- **Calibration:** rebalancing the training data inflates predicted probabilities. I fix this with Platt scaling, a tiny logistic regression fitted on the validation set that stretches and shifts the scores without changing the ranking of applicants. I also compare it against isotonic regression and check reliability curves
- **A baseline to beat:** a gradient boosting model trained on the same features, to see what extra complexity actually buys
- **Credit-risk metrics:** ROC-AUC, Gini, KS, and PR-AUC for ranking quality, plus Brier score and log-loss for probability quality
- **A threshold that reflects money, not habit:** instead of the default 0.5, I choose the cutoff that minimizes expected business cost, where approving a loan that defaults costs more than rejecting a good applicant. The threshold is picked on the validation set, then checked on the test set against simple policies like "approve everyone" and "use 0.5". A sensitivity table shows how the best threshold moves as the cost ratio changes
- **Risk deciles:** applicants are grouped by predicted risk to confirm that higher predicted risk really does mean more defaults
- **Odds ratios in plain English:** coefficients are converted into statements like "each extra percentage point of interest rate multiplies the odds of default by X", with 95% confidence intervals from bootstrapping. Features whose intervals include 1 are flagged as having no clear effect

The test set is used once, only to report how earlier choices perform. It never influences them.

### 5. Turning the model into an app

Getting a model into an app has a few traps, so I handled them deliberately:

- **No pickles in the app.** An export script writes the trained model, scaling statistics, calibration, and cutoff to a small JSON file. That file is safe to commit and doesn't depend on scikit-learn or on the training code, so the deployed app stays light.
- **The app is proven to match the pipeline.** The export script scores the real test set two ways, through the training pipeline and through the app's code, starting from raw typed-in fields, and checks that the probabilities agree. If they don't, it warns before you deploy.
- **The same cleaning rules apply.** Missing values are filled with the training medians, and impossible inputs are rejected, exactly as in the cleaning step.

---

## Project structure

```
credit-risk-logistic-regression/
├── app/
│   ├── app.py                    # Streamlit app
│   ├── scoring.py                # Scoring logic (numpy + pandas only)
│   ├── model_artifacts.json      # Exported model, written by export_artifacts.py
│   └── requirements.txt          # Light dependencies for deployment
├── dataset/                      # Raw Kaggle CSV (not committed, see below)
├── data/
│   └── processed/                # train.csv, val.csv, test.csv, cleaning_params.json
├── models/                       # Saved model, preprocessor, calibrator, and settings
├── reports/
│   ├── figures/                  # EDA, modeling, and evaluation charts
│   └── *.csv                     # VIF, CV results, imbalance comparison, test metrics,
│                                 # threshold analysis, deciles, odds ratios
├── src/
│   ├── data_cleaning.py          # Cleaning, EDA, feature engineering, split
│   ├── preprocessing.py          # Scaling and encoding (fit on train only)
│   ├── logreg_scratch.py         # Logistic regression from scratch in NumPy
│   ├── modeling.py               # Validation, tuning, imbalance comparison, saving
│   ├── calibration.py            # Platt scaling for trustworthy probabilities
│   ├── evaluation.py             # Test metrics, threshold, deciles, odds ratios
│   └── export_artifacts.py       # Exports the model for the app and verifies it
├── requirements.txt
└── README.md
```

---

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/Muzammil592/credit-risk-logistic-regression.git
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

**6. Evaluate and interpret**
```bash
python src/evaluation.py
```
This calibrates the probabilities, trains the gradient boosting baseline, scores everything on the test set, chooses the cost-based threshold, builds the decile table, and bootstraps the odds ratios. It takes a few minutes because of the bootstrapping.

**7. Export the model for the app**
```bash
python src/export_artifacts.py
```
This writes `app/model_artifacts.json` and checks that the app's scoring matches the training pipeline. You should see a PASS message at the end.

**8. Launch the app**
```bash
streamlit run app/app.py
```
It opens in your browser at `http://localhost:8501`.

---

## Decisions worth knowing about

A few deliberate choices, in case you're reading the code and wondering why:

- **Anything learned from data is fit on train only.** Imputation values, scaling statistics, and category lists all come from the training split. Otherwise validation and test information leaks in.
- **Exploration uses the training split only.** The test set stays untouched until the very end.
- **Resampling touches training data only.** Over- and undersampling are never applied to validation or test.
- **Calibration and the threshold are chosen on validation.** The test set only reports how those choices perform, so the final numbers aren't flattered by tuning.
- **The cost ratio is an assumption.** The relative cost of a missed default versus a rejected good loan is illustrative, not taken from a real lender, which is why the project includes a sensitivity table rather than a single answer.
- **The app runs on exported numbers, not pickles.** Pickled models can break between library versions, so the app reads a plain JSON file instead.
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
- The business costs behind the decision threshold are assumptions. A real lender would plug in its own loss and profit figures.
- Odds ratios and the app's explanation chart hold other features constant, and several features overlap (interest rate and loan grade, loan amount and loan-to-income), so individual effects should be read together rather than in isolation.
- The app is an educational demonstration. It has not been tested for fairness and is **not** meant for real lending decisions, which would also need regulatory review.

---

## Acknowledgments

Dataset: [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) by Lao Tse on Kaggle. Please check the dataset page for its license terms.

---

## Contact

Questions or feedback? Open an issue or reach out on [LinkedIn](https://www.linkedin.com/in/muzammil-abbas-404b80266/) or by [email](mailto:muzammilabbas779677@gmail.com).
