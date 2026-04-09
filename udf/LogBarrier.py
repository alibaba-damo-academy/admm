import admm
import numpy as np


class LogBarrier(admm.UDFBase):
    r"""Logarithmic barrier function for the positive orthant.

    Mathematical definition:

        f(x) = −Σᵢ log(xᵢ)

    Domain: x > 0 (componentwise).

    Behavior:
        xᵢ → 0⁺:   f → +∞              (infinite barrier at boundary)
        xᵢ → +∞:    f → −∞ (slowly)     (logarithmic decay)
        xᵢ = 1:     f(xᵢ) = 0           (each component)

    Gradient:

        ∇f(x)ᵢ = −1/xᵢ

    Hessian:

        ∇²f(x)ᵢᵢ = 1/xᵢ²    (diagonal, positive definite on x > 0)

    Properties:
    - Strictly convex on x > 0
    - Self-concordant: |f'''| ≤ 2·(f'')^(3/2)
    - The canonical barrier for the positive orthant in interior
      point methods

    The log barrier is a cornerstone of interior point optimization.
    Adding t·f(x) to an objective keeps iterates strictly positive,
    with t → 0 recovering the constrained optimum. Also used as a
    regularizer to prevent variables from reaching zero.

    Parameters
    ----------
    arg : admm.Var or expression
        The input (must be positive).
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        x = np.maximum(x, 1e-30)
        return float(-np.sum(np.log(x)))

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        x = np.maximum(x, 1e-30)
        return [-1.0 / x]
