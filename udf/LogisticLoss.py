import admm
import numpy as np


class LogisticLoss(admm.UDFBase):
    r"""Logistic regression loss (negative log-likelihood).

    Mathematical definition:

        f(w) = Σᵢ log(1 + exp(−yᵢ · aᵢᵀw))

    where A is the data matrix (m × n), y ∈ {−1, +1}ᵐ are labels,
    and w ∈ ℝⁿ is the weight vector.

    Behavior:
        yᵢ·aᵢᵀw ≫ 0:   loss ≈ 0           (correct, confident)
        yᵢ·aᵢᵀw = 0:    loss = log(2)      (uncertain)
        yᵢ·aᵢᵀw ≪ 0:    loss ≈ −yᵢ·aᵢᵀw   (wrong, linear penalty)

    Gradient:

        ∇f(w) = −Aᵀ · (y ⊙ σ(−y ⊙ Aw))

    where σ(z) = 1/(1+exp(−z)) is the sigmoid function.

    Properties:
    - Convex (log-sum-exp composition)
    - Smooth (C∞)
    - Lipschitz gradient with L = ‖A‖₂² / 4
    - Fisher-consistent for the Bayes-optimal classifier

    The logistic loss is the standard loss for binary classification.
    Unlike the hinge loss (SVM), it provides calibrated probability
    estimates via P(y=1|x) = σ(wᵀx).

    Parameters
    ----------
    arg : admm.Var or expression
        The weight vector w.
    A : array_like, shape (m, n)
        Data matrix (each row is a data point).
    y : array_like, shape (m,)
        Labels in {−1, +1}.
    """

    def __init__(self, arg, A, y):
        self.arg = arg
        self.A = np.asarray(A, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        w = np.asarray(arglist[0], dtype=float).ravel()
        z = self.y * (self.A @ w)
        return float(np.sum(np.logaddexp(0, -z)))

    def grad(self, arglist):
        w = np.asarray(arglist[0], dtype=float).ravel()
        z = self.y * (self.A @ w)
        sig = 1.0 / (1.0 + np.exp(z))  # sigmoid(-z)
        g = -self.A.T @ (self.y * sig)
        return [g.reshape(arglist[0].shape)]
