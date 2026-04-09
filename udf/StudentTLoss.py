import admm
import numpy as np


class StudentTLoss(admm.UDFBase):
    r"""Student-t loss: heavy-tailed robust loss based on the t-distribution.

    Mathematical definition:

        f(r) = Σᵢ log(1 + rᵢ²/ν)

    where ν > 0 is the degrees-of-freedom parameter.

    Behavior:
        |r| ≪ √ν:  f(r) ≈ r²/ν                    (like scaled L2)
        |r| ≫ √ν:  f(r) ≈ 2·log(|r|/√ν)           (logarithmic, very slow growth)

    The Student-t loss is closely related to the Cauchy loss (which
    corresponds to ν = 1). Larger ν makes the loss closer to L2;
    smaller ν gives heavier tails and more robustness.

    Gradient:

        ∇f(r)ᵢ = 2rᵢ / (ν + rᵢ²)

    The gradient is redescending: it peaks at |r| = √ν and decays
    to zero as O(1/r) for large residuals.

    Parameters
    ----------
    arg : admm.Var or expression
        The input (typically a residual vector).
    v : float
        Degrees of freedom. Smaller v = heavier tails = more robust.
        v=1 gives the Cauchy loss.
    """

    def __init__(self, arg, v=1.0):
        self.arg = arg
        self.v = v

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return float(np.sum(np.log(1 + r ** 2 / self.v)))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return [2.0 * r / (self.v + r ** 2)]
