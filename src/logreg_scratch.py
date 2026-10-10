"""
Logistic regression from scratch in NumPy.

Objective being minimised (same scaling as scikit-learn, divided by C * n):

    (1/n) * sum_i  w_i * [ log(1 + exp(z_i)) - y_i * z_i ]      <- weighted log-loss
    + 1 / (2 C n) * ||beta||_2^2                                 <- if penalty = "l2"
    + 1 / (C n)   * ||beta||_1                                   <- if penalty = "l1"

where z_i = x_i . beta + b, and w_i are sample weights (class weights included).
The intercept b is never penalised.

Because the scaling matches scikit-learn, the same `C` means the same thing in both
libraries, so you can compare coefficients directly.

Optimiser (full-batch, first-order, no external solvers):
  - none / L2: gradient descent
  - L1: proximal gradient descent (ISTA): a gradient step on the smooth loss followed by
        soft-thresholding, which is what drives coefficients to exactly 0.
  - accelerated=True (default) adds Nesterov momentum (FISTA) with adaptive restart.
    Features such as log_income, log_loan_amnt and loan_percent_income are strongly
    related, which makes plain gradient descent crawl along a narrow valley; momentum
    fixes most of that. Set accelerated=False to see the plain version.
  - The step size defaults to 1/L, where L is the largest curvature of the loss, so you
    don't have to hand-tune a learning rate.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: only ever exponentiates a non-positive number."""
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def log_loss_terms(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row  log(1 + exp(z)) - y*z, computed stably."""
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - y * z


class LogisticRegressionScratch:
    def __init__(
        self,
        penalty=None,          # None, "l2" or "l1"
        C: float = 1.0,        # inverse regularisation strength (like scikit-learn)
        class_weight=None,     # None, "balanced", or {0: w0, 1: w1}
        lr=None,               # None = automatic 1/L step size
        max_iter: int = 5000,
        tol: float = 1e-7,     # stop when the (proximal) gradient step is this small
        accelerated: bool = True,
    ):
        if penalty not in (None, "l1", "l2"):
            raise ValueError("penalty must be None, 'l1' or 'l2'")
        self.penalty = penalty
        self.C = C
        self.class_weight = class_weight
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.accelerated = accelerated

    # ------------------------------------------------------------------ helpers
    def _sample_weights(self, y, sample_weight):
        n = len(y)
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        cw = self.class_weight
        if cw is None:
            return sw
        if cw == "balanced":
            # same formula as scikit-learn: n / (n_classes * count_of_class)
            n1 = y.sum()
            n0 = n - n1
            cw = {0: n / (2.0 * n0), 1: n / (2.0 * n1)}
        return sw * np.where(y == 1, cw[1], cw[0])

    def _reg_strength(self, n):
        return 0.0 if self.penalty is None else 1.0 / (self.C * n)

    def _auto_lr(self, X, sw, lam):
        """1/L, where L bounds the curvature of the loss: 0.25 * max(w_i) * lambda_max(X'X/n)."""
        Xa = np.hstack([X, np.ones((X.shape[0], 1))])
        lmax = np.linalg.eigvalsh(Xa.T @ Xa / X.shape[0])[-1]
        L = 0.25 * lmax * sw.max() + (lam if self.penalty == "l2" else 0.0)
        return 1.0 / L

    def _objective(self, w, b, X, y, sw, lam):
        z = X @ w + b
        loss = np.mean(sw * log_loss_terms(z, y))
        if self.penalty == "l2":
            loss += 0.5 * lam * np.dot(w, w)
        elif self.penalty == "l1":
            loss += lam * np.abs(w).sum()
        return loss

    def _smooth_loss_and_grad(self, w, b, X, y, sw):
        """Weighted log-loss (+ L2 term if used) and its gradients."""
        n = X.shape[0]
        z = X @ w + b
        loss = np.mean(sw * log_loss_terms(z, y))
        resid = sw * (sigmoid(z) - y)
        grad_w = X.T @ resid / n
        grad_b = resid.sum() / n
        if self.penalty == "l2":
            lam = self._reg_strength(n)
            loss += 0.5 * lam * np.dot(w, w)
            grad_w = grad_w + lam * w
        return loss, grad_w, grad_b

    # --------------------------------------------------------------------- API
    def fit(self, X, y, sample_weight=None, init=None):
        """`init=(coef, intercept)` optionally warm-starts the optimiser (used for bootstrapping)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        sw = self._sample_weights(y, sample_weight)
        lam = self._reg_strength(n)
        lr = self.lr if self.lr is not None else self._auto_lr(X, sw, lam)

        if init is None:
            w, b = np.zeros(d), 0.0       # current iterate
        else:
            w, b = np.array(init[0], dtype=float), float(init[1])
        yw, yb = w.copy(), b              # point where the gradient is evaluated
        t = 1.0                           # momentum bookkeeping (FISTA)
        prev_loss = np.inf
        self.loss_history_ = []

        for it in range(1, self.max_iter + 1):
            _, gw, gb = self._smooth_loss_and_grad(yw, yb, X, y, sw)

            w_new = yw - lr * gw
            b_new = yb - lr * gb
            if self.penalty == "l1":
                # soft-thresholding: the proximal step for the L1 penalty
                w_new = np.sign(w_new) * np.maximum(np.abs(w_new) - lr * lam, 0.0)

            loss = self._objective(w_new, b_new, X, y, sw, lam)
            self.loss_history_.append(loss)
            step = max(np.max(np.abs(w_new - w)), abs(b_new - b)) / lr

            if self.accelerated:
                if loss > prev_loss:      # momentum overshot: restart it
                    t = 1.0
                    yw, yb = w_new.copy(), b_new
                else:
                    t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
                    mom = (t - 1.0) / t_new
                    yw = w_new + mom * (w_new - w)
                    yb = b_new + mom * (b_new - b)
                    t = t_new
            else:
                yw, yb = w_new, b_new

            w, b, prev_loss = w_new, b_new, loss
            if step < self.tol:
                break

        self.coef_ = w
        self.intercept_ = b
        self.n_iter_ = it
        self.lr_ = lr
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        """Probability of class 1 (default), as a 1-D array."""
        return sigmoid(self.decision_function(X))

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def gradient_check(X, y, penalty="l2", C=1.0, eps=1e-6, seed=0):
    """
    Compare the analytic gradient against a numerical (finite-difference) gradient.
    A max relative error around 1e-7 or smaller means the maths is implemented correctly.
    """
    rng = np.random.default_rng(seed)
    model = LogisticRegressionScratch(penalty=penalty, C=C)
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    sw = np.ones(len(y))
    w = rng.normal(scale=0.1, size=X.shape[1])
    b = 0.1

    _, gw, gb = model._smooth_loss_and_grad(w, b, X, y, sw)

    num_gw = np.zeros_like(w)
    for j in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[j] += eps
        wm[j] -= eps
        num_gw[j] = (model._smooth_loss_and_grad(wp, b, X, y, sw)[0]
                     - model._smooth_loss_and_grad(wm, b, X, y, sw)[0]) / (2 * eps)
    num_gb = (model._smooth_loss_and_grad(w, b + eps, X, y, sw)[0]
              - model._smooth_loss_and_grad(w, b - eps, X, y, sw)[0]) / (2 * eps)

    analytic = np.append(gw, gb)
    numeric = np.append(num_gw, num_gb)
    rel_err = np.abs(analytic - numeric) / np.maximum(1e-12, np.abs(analytic) + np.abs(numeric))
    return float(rel_err.max())
    