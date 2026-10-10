"""
Platt scaling: fixes probabilities that were distorted by class weights or resampling.

Idea: take the model's raw score z (the log-odds) and fit a tiny 1-feature logistic
regression  p = sigmoid(a * z + b)  on data the model was NOT trained on (the validation
set). Because a and b just stretch and shift the score, the ranking of applicants (and so
the AUC) is unchanged. Only the probabilities become trustworthy.

Lives in its own module (not inside a script) so the saved object can be loaded later,
for example by the Streamlit app.
"""

import numpy as np

from logreg_scratch import LogisticRegressionScratch, sigmoid


class PlattCalibrator:
    def fit(self, scores, y):
        scores = np.asarray(scores, dtype=float)
        mu, sd = scores.mean(), scores.std()
        z = ((scores - mu) / sd).reshape(-1, 1)          # standardise for stable optimisation
        m = LogisticRegressionScratch(penalty=None, max_iter=5000, tol=1e-10).fit(z, y)
        # convert back to the original score scale
        self.a_ = float(m.coef_[0] / sd)
        self.b_ = float(m.intercept_ - m.coef_[0] * mu / sd)
        return self

    def predict_proba(self, scores):
        return sigmoid(self.a_ * np.asarray(scores, dtype=float) + self.b_)
