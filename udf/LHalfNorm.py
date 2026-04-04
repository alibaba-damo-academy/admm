import admm
import numpy as np


class LHalfNorm(admm.UDFBase):
    """L_{1/2} penalty: sum of square roots of absolute values.

    Function:

        f(x) = ||x||_{1/2} = sum_i sqrt(|x_i|)

    The L_{1/2} penalty promotes sparsity more aggressively than L1
    while remaining continuous. It is nonconvex.

    Proximal operator (coordinatewise; active when |v_i| > 1.5 * lam^(2/3)):

        prox_{lam * f}(v)_i = (2/3) * |v_i| * (1 + cos(2*pi/3 - 2*phi_i/3)) * sgn(v_i)

        where  phi_i = arccos(3 * sqrt(3) * lam / (4 * |v_i|^(3/2)))

    Parameters
    ----------
    arg : admm.Var
        The optimization variable (vector).
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, tensorlist):
        vector = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.sqrt(np.abs(vector))))

    def argmin(self, lamb, tensorlist):
        vector = np.asarray(tensorlist[0], dtype=float)
        abs_vector = np.abs(vector)
        threshold = 1.5 * (lamb ** (2.0 / 3.0))
        prox = np.zeros_like(vector)

        active = abs_vector > threshold
        if np.any(active):
            phi = np.arccos(
                np.clip(
                    (3.0 * np.sqrt(3.0) * lamb) / (4.0 * np.power(abs_vector[active], 1.5)),
                    -1.0,
                    1.0,
                )
            )
            prox_abs = (2.0 * abs_vector[active] / 3.0) * (
                1.0 + np.cos((2.0 * np.pi / 3.0) - (2.0 * phi / 3.0))
            )
            prox[active] = np.sign(vector[active]) * prox_abs

        return [prox.tolist()]
