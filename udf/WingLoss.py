import admm
import numpy as np


class WingLoss(admm.UDFBase):
    """Wing loss: precision-focused loss from face landmark localization.

    Function:

        f(r) = sum(w * ln(1 + sqrt(r_i^2 + delta^2) / eps))

    Behavior:
        - Near zero: gradient ≈ w * r / (delta * (eps + delta)) — steep
        - Large |r|: grows as w * ln(|r| / eps) — logarithmic

    Gradient:

        grad_i = w * r_i / (sqrt(r_i^2 + delta^2) * (eps + sqrt(r_i^2 + delta^2)))

    The steep gradient near zero makes the optimizer "pay more attention"
    to reducing small errors, unlike L2 whose gradient vanishes at zero.

    Parameters
    ----------
    arg : admm.Var or expression
        The input (typically a residual vector).
    w : float
        Loss magnitude. Default 10.0.
    eps : float
        Transition scale. Default 2.0.
    delta : float
        Smoothing at zero. Default 0.01.

    Reference
    ---------
    Feng et al., "Wing Loss for Robust Facial Landmark Localisation with
    Convolutional Neural Networks", CVPR 2018.
    """

    def __init__(self, arg, w=10.0, eps=2.0, delta=0.01):
        self.arg = arg
        self.w = w
        self.eps = eps
        self.delta = delta

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        s = np.sqrt(r ** 2 + self.delta ** 2)
        return float(np.sum(self.w * np.log(1 + s / self.eps)))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        s = np.sqrt(r ** 2 + self.delta ** 2)
        return [self.w * (r / s) / (self.eps + s)]
