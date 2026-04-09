import admm
import numpy as np


class DoubleWellPotential(admm.UDFBase):
    r"""Double-well potential: non-convex penalty with two minima.

    Mathematical definition:

        f(x) = Σᵢ (xᵢ² − 1)²

    Behavior:
        xᵢ = ±1:   f = 0                 (two global minima)
        xᵢ = 0:    f = 1                 (local maximum, barrier)
        |xᵢ| → ∞:  f → ∞                (quartic growth)

    Expanded form:

        f(x) = Σᵢ (xᵢ⁴ − 2xᵢ² + 1)

    Gradient:

        ∇f(x)ᵢ = 4xᵢ · (xᵢ² − 1)

    Gradient behavior:
        xᵢ = 0:    ∇f = 0   (unstable equilibrium)
        xᵢ = ±1:   ∇f = 0   (stable minima)
        |xᵢ| > 1:  pushes away from origin (quartic growth)
        |xᵢ| < 1:  pushes toward ±1

    Properties:
    - Non-convex (the key feature: two separated minima)
    - Smooth (C∞)
    - The prototypical bistable potential in physics
    - The Ginzburg-Landau free energy density

    Used in:
    - Phase-field modeling (material science)
    - Binary signal recovery (encourages ±1 values)
    - Bistable systems in physics
    - Non-convex regularization (promotes discrete values)
    - Testing optimization algorithms on non-convex landscapes

    Parameters
    ----------
    arg : admm.Var or expression
        The input vector.
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.sum((x ** 2 - 1) ** 2))

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return [4.0 * x * (x ** 2 - 1)]
