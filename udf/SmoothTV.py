import admm
import numpy as np


class SmoothTV(admm.UDFBase):
    """Smooth total variation penalty for edge-preserving denoising.

    Function:

        f(x) = sum(sqrt((x_{i+1} - x_i)^2 + eps))

    This is a differentiable approximation to sum(|x_{i+1} - x_i|),
    the standard total variation penalty used in signal and image processing.

    Gradient (involves neighboring elements):

        g_1     = -(d_1 / sqrt(d_1^2 + eps))
        g_i     = (d_{i-1} / sqrt(d_{i-1}^2 + eps)) - (d_i / sqrt(d_i^2 + eps))
        g_n     = (d_{n-1} / sqrt(d_{n-1}^2 + eps))

    where d_i = x_{i+1} - x_i.

    Parameters
    ----------
    arg : admm.Var
        Signal variable (1-D vector).
    eps : float
        Smoothing parameter. Smaller eps = closer to true TV.
    """

    def __init__(self, arg, eps=1e-4):
        self.arg = arg
        self.eps = eps

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float).ravel()
        d = np.diff(x)
        return float(np.sum(np.sqrt(d ** 2 + self.eps)))

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float).ravel()
        d = np.diff(x)
        dd = d / np.sqrt(d ** 2 + self.eps)
        g = np.zeros_like(x)
        g[:-1] -= dd
        g[1:] += dd
        return [g.reshape(arglist[0].shape)]
