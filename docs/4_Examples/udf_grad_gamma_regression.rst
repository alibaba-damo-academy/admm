.. include:: ../definition.hrst

.. _udf-grad-example-gamma-regression:

Gamma Regression (GLM with Log Link)
=====================================

This example shows how to package a **Gamma regression loss** — a generalized linear model (GLM)
for positive-valued responses — as a gradient-based UDF. The full model is

.. math::

   \begin{array}{ll}
   \min\limits_\mu & \displaystyle\sum_{i=1}^{n} \bigl(y_i e^{-\mu_i} + \mu_i\bigr)
     \;+\; \tfrac{\lambda}{2}\|\mu\|_2^2 \\[6pt]
   \text{s.t.} & -5 \le \mu_i \le 10.
   \end{array}

Here :math:`\mu_i = \log(\mathbb{E}[y_i])` is the log-scale parameter (log link), and
:math:`y_i > 0` is the observed positive response. The Gamma deviance
:math:`y e^{-\mu} + \mu` has no closed-form proximal operator, but its gradient is trivial —
making it a natural fit for the ``grad`` UDF path.

The Gamma distribution is a standard model for positive, right-skewed data:

- **Insurance**: claim amounts
- **Survival analysis**: waiting times, durations
- **Environmental science**: rainfall, pollutant concentrations

The log link :math:`\mathbb{E}[y] = e^\mu` ensures positivity of the predicted mean without
explicit positivity constraints.

The function value returned by :py:meth:`UDFBase.eval` sums the Gamma deviance over all
observations:

.. math::

   f(\mu) = \sum_i \bigl(y_i \, e^{-\mu_i} + \mu_i\bigr).

This function is convex, with a unique minimum at :math:`\mu_i = \log(y_i)` (the sample
log-mean). So ``eval`` computes:

.. code-block:: python

    def eval(self, arglist):
        mu = np.asarray(arglist[0], dtype=float).ravel()
        return float(np.sum(self.y * np.exp(-mu) + mu))

The gradient returned by :py:meth:`UDFBase.grad` is

.. math::

   \nabla f(\mu)_i = -y_i \, e^{-\mu_i} + 1,

which is zero exactly at :math:`\mu_i = \log(y_i)`. The implementation:

.. code-block:: python

    def grad(self, arglist):
        mu = np.asarray(arglist[0], dtype=float).ravel()
        return [(-self.y * np.exp(-mu) + 1.0).reshape(arglist[0].shape)]

Note the ``.reshape(arglist[0].shape)`` at the end — this ensures the gradient has the same
shape as the input, which is important when the solver passes 2-D arrays.

The :py:meth:`UDFBase.arguments` method returns the single optimization variable. The observed
data :math:`y` is a fixed parameter stored as an instance attribute:

.. code-block:: python

    def arguments(self):
        return [self.arg]

This pattern — storing fixed data in ``__init__`` while listing only the optimization variable
in ``arguments`` — is the standard way to embed problem data in a UDF.

Complete runnable example:

.. code-block:: python

    import admm
    import numpy as np

    class GammaRegressionLoss(admm.UDFBase):
        """Gamma regression loss (log link): f(mu) = sum(y*exp(-mu) + mu).

        The minimum is at mu_i = log(y_i).
        Gradient: -y*exp(-mu) + 1.

        Parameters
        ----------
        arg : admm.Var
            The log-scale parameter mu.
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
            return float(np.sum(self.y * np.exp(-mu) + mu))

        def grad(self, arglist):
            mu = np.asarray(arglist[0], dtype=float).ravel()
            return [(-self.y * np.exp(-mu) + 1.0).reshape(arglist[0].shape)]

    # Generate Gamma-distributed data
    np.random.seed(314)
    n = 30
    mu_true = np.random.uniform(0.5, 3.0, size=n)
    k = 5.0  # shape parameter
    y = np.random.gamma(shape=k, scale=np.exp(mu_true) / k, size=n)

    model = admm.Model()
    mu = admm.Var("mu", n)
    model.setObjective(GammaRegressionLoss(mu, y) + 0.025 * admm.sum(admm.square(mu)))
    model.addConstr(mu >= -5)
    model.addConstr(mu <= 10)
    model.optimize()

    mu_val = np.asarray(mu.X).ravel()
    print(" * status:", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(f" * RMSE: {np.sqrt(np.mean((mu_val - mu_true) ** 2)):.4f}")  # Expected: ≈ 0.42
    print(f" * Correlation: {np.corrcoef(np.exp(mu_val), np.exp(mu_true))[0,1]:.4f}")


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_grad_gamma_regression.py


This example demonstrates how |ADMM|'s constraint handling composes naturally with custom
smooth losses through the ``grad`` path. The box constraints :math:`-5 \le \mu \le 10`
prevent numerical instability in :math:`e^{-\mu}`, while the L2 regularizer shrinks the
estimates toward zero. The Gamma regression UDF recovers the log-mean parameters with an
RMSE comparable to the unregularized MLE (:math:`\hat\mu_i = \log y_i`).
