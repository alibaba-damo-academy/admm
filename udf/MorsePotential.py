import admm
import numpy as np


class MorsePotential(admm.UDFBase):
    r"""Morse potential: a bounded smooth potential from molecular physics.

    Mathematical definition:

        f(x) = D · Σᵢ (1 − exp(−a·(xᵢ − r₀)))²

    where D > 0 is the well depth, a > 0 controls the well width,
    and r₀ is the equilibrium position.

    Behavior:
        xᵢ = r₀:    f = 0                 (minimum, equilibrium)
        xᵢ → ±∞:    f → D                 (bounded, saturates)
        |xᵢ − r₀| ≪ 1/a:  f ≈ D·a²·(xᵢ−r₀)²  (parabolic near minimum)

    Gradient:

        ∇f(x)ᵢ = 2Da · (1 − exp(−a·(xᵢ−r₀))) · exp(−a·(xᵢ−r₀))

    Properties:
    - Smooth (C∞) everywhere
    - Bounded above by n·D
    - Non-convex globally, but locally quadratic near r₀
    - The gradient is redescending: peaks and decays to zero

    Originally from quantum chemistry (diatomic molecule vibrations),
    the Morse potential is useful as a robust loss that:
    - Behaves like L2 near the equilibrium
    - Saturates for large deviations (outlier rejection)
    - Has a tunable shape via the (D, a, r₀) parameters

    Parameters
    ----------
    arg : admm.Var or expression
        The input.
    D : float
        Well depth (maximum loss per element).
    a : float
        Controls the width of the well (larger a = narrower well).
    r0 : float
        Equilibrium position (location of the minimum).
    """

    def __init__(self, arg, D=1.0, a=1.0, r0=0.0):
        self.arg = arg
        self.D = D
        self.a = a
        self.r0 = r0

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        t = np.clip(-self.a * (x - self.r0), -500, 500)
        e = np.exp(t)
        return float(self.D * np.sum((1 - e) ** 2))

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        t = np.clip(-self.a * (x - self.r0), -500, 500)
        e = np.exp(t)
        return [2.0 * self.D * self.a * (1 - e) * e]
