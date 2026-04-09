import admm
import numpy as np


class BinaryCrossEntropyLoss(admm.UDFBase):
    r"""Binary cross-entropy (log loss) for classification.

    Mathematical definition:

        f(p) = −Σᵢ [tᵢ · log(pᵢ) + (1 − tᵢ) · log(1 − pᵢ)]

    where t ∈ [0,1]ⁿ are fixed target probabilities and
    p ∈ (0,1)ⁿ are the predicted probabilities (the variable).

    Behavior:
        pᵢ → tᵢ:   f(p) → 0            (perfect prediction)
        pᵢ → 0, tᵢ = 1:  f → +∞        (confident wrong prediction)
        pᵢ → 1, tᵢ = 0:  f → +∞        (confident wrong prediction)

    Gradient:

        ∇f(p)ᵢ = −tᵢ/pᵢ + (1 − tᵢ)/(1 − pᵢ)

    The binary cross-entropy is the standard loss for binary
    classification (logistic regression). It is strictly convex
    on (0,1) and measures the KL divergence between the Bernoulli
    distributions parameterized by t and p.

    Note: Variables should be constrained to (0, 1) for the loss
    to be well-defined. Use box constraints 0 < p < 1.

    Parameters
    ----------
    arg : admm.Var or expression
        The predicted probabilities p ∈ (0, 1).
    targets : array_like
        Target probabilities t ∈ [0, 1].
    """

    def __init__(self, arg, targets):
        self.arg = arg
        self.t = np.asarray(targets, dtype=float)

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        p = np.clip(np.asarray(arglist[0], dtype=float), 1e-15, 1 - 1e-15)
        return float(-np.sum(
            self.t * np.log(p) + (1 - self.t) * np.log(1 - p)
        ))

    def grad(self, arglist):
        p = np.clip(np.asarray(arglist[0], dtype=float), 1e-15, 1 - 1e-15)
        return [-self.t / p + (1 - self.t) / (1 - p)]
