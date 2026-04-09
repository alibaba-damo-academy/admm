import admm
import numpy as np


class GammaRegressionLoss(admm.UDFBase):
    """Gamma regression loss with log link (GLM deviance).

    Function:

        f(mu) = sum(y_i * exp(-mu_i) + mu_i)

    where mu_i = log(E[y_i]) is the log-scale parameter.
    The minimum is at mu_i = log(y_i).

    Gradient:

        grad_i = -y_i * exp(-mu_i) + 1

    Applications: insurance claims, survival analysis, environmental
    data — any positive, right-skewed response variable.

    Parameters
    ----------
    arg : admm.Var
        The log-scale parameter mu (same length as y).
    y : array
        Observed positive responses.
    """

    def __init__(self, arg, y):
        self.arg = arg
        self.y = np.asarray(y, dtype=float)

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        mu = np.asarray(arglist[0], dtype=float).ravel()
        neg_mu = np.clip(-mu, -500, 500)
        return float(np.sum(self.y * np.exp(neg_mu) + mu))

    def grad(self, arglist):
        mu = np.asarray(arglist[0], dtype=float).ravel()
        neg_mu = np.clip(-mu, -500, 500)
        return [(-self.y * np.exp(neg_mu) + 1.0).reshape(arglist[0].shape)]
