.. include:: ../definition.hrst

.. _udf-grad-example-smooth-quantile:

Smooth Quantile Regression
==========================

This example shows how to implement a **smooth quantile loss** as a gradient-based UDF. The full
model is

.. math::

   \begin{array}{ll}
   \min\limits_x & \displaystyle\sum_{i=1}^{n} \Bigl[
     \tau \, u_i + \tfrac{1}{\beta} \log\!\bigl(1 + e^{-\beta u_i}\bigr)
   \Bigr],
   \qquad u_i = a_i^\top x - b_i.
   \end{array}

This is a smooth approximation to the classical pinball (check) loss
:math:`\rho_\tau(u) = u \cdot (\tau - \mathbf{1}_{u<0})` used in quantile regression.
The parameter :math:`\tau \in (0,1)` selects which conditional quantile to estimate (e.g.,
:math:`\tau = 0.5` gives the median, :math:`\tau = 0.1` gives the 10th percentile).
As :math:`\beta \to \infty` the smooth version converges to the exact pinball loss.

Ordinary regression estimates the conditional *mean*. Quantile regression estimates the
conditional *quantile*, which is useful for prediction intervals, asymmetric risk modeling,
and detecting heteroscedasticity.

The function value returned by :py:meth:`UDFBase.eval` combines a linear term with a softplus:

.. math::

   f(u) = \sum_i \Bigl[\tau \, u_i + \tfrac{1}{\beta}\,\mathrm{softplus}(-\beta u_i)\Bigr],

where :math:`\mathrm{softplus}(z) = \log(1 + e^z)`. For numerical stability, the softplus is
evaluated using the identity :math:`\mathrm{softplus}(z) \approx z` for large :math:`z`:

.. code-block:: python

    def _softplus(self, z):
        return np.where(z > 20, z, np.log(1 + np.exp(np.clip(z, -500, 20))))

    def eval(self, arglist):
        u = np.asarray(arglist[0], dtype=float)
        return float(np.sum(self.tau * u + (1.0 / self.beta) * self._softplus(-self.beta * u)))

The gradient returned by :py:meth:`UDFBase.grad` is

.. math::

   \nabla f(u)_i = \tau - \bigl(1 - \sigma(\beta u_i)\bigr) = \tau - \frac{1}{1 + e^{\beta u_i}},

where :math:`\sigma` is the logistic sigmoid. For :math:`u_i \gg 0` (prediction above target),
the gradient approaches :math:`\tau`; for :math:`u_i \ll 0` (prediction below target), it
approaches :math:`\tau - 1`. This asymmetry is what makes quantile regression estimate quantiles
rather than means.

The sigmoid is also evaluated with a numerically stable formula:

.. code-block:: python

    def _sigmoid(self, z):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    def grad(self, arglist):
        u = np.asarray(arglist[0], dtype=float)
        return [self.tau - (1.0 - self._sigmoid(self.beta * u))]

The :py:meth:`UDFBase.arguments` method returns the single expression this UDF depends on:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Note that ``tau`` and ``beta`` are hyperparameters stored as instance attributes — they are
not optimization variables and do not appear in ``arguments``.

Complete runnable example:

.. code-block:: python

    import admm
    import numpy as np

    class SmoothQuantileLoss(admm.UDFBase):
        """Smooth pinball loss for quantile regression.

        f(u) = sum(tau*u + (1/beta)*softplus(-beta*u))
        grad = tau - (1 - sigmoid(beta*u))

        Parameters
        ----------
        arg : admm.Var or expression
            The residual u = prediction - target.
        tau : float
            Quantile level in (0, 1). Default 0.5 (median).
        beta : float
            Smoothing parameter. Larger = closer to exact pinball.
        """
        def __init__(self, arg, tau=0.5, beta=20.0):
            self.arg = arg
            self.tau = tau
            self.beta = beta

        def arguments(self):
            return [self.arg]

        def _softplus(self, z):
            return np.where(z > 20, z, np.log(1 + np.exp(np.clip(z, -500, 20))))

        def _sigmoid(self, z):
            return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                            np.exp(z) / (1.0 + np.exp(z)))

        def eval(self, arglist):
            u = np.asarray(arglist[0], dtype=float)
            return float(np.sum(self.tau * u + (1.0 / self.beta) * self._softplus(-self.beta * u)))

        def grad(self, arglist):
            u = np.asarray(arglist[0], dtype=float)
            return [self.tau - (1.0 - self._sigmoid(self.beta * u))]

    # Heteroscedastic data: noise variance depends on x[:,0]
    np.random.seed(123)
    n, p = 100, 3
    A = np.random.randn(n, p)
    x_true = np.array([2.0, -1.0, 0.5])
    noise_scale = 0.5 + 2.0 * np.abs(A[:, 0])
    b = A @ x_true + noise_scale * np.random.randn(n)

    # Fit three quantiles to get a prediction interval
    for tau in [0.1, 0.5, 0.9]:
        model = admm.Model()
        x = admm.Var("x", p)
        model.setObjective(SmoothQuantileLoss(A @ x - b, tau=tau))
        model.optimize()
        print(f" * tau={tau}: x = {np.asarray(x.X).round(4)}")

    # Expected: tau=0.1 and tau=0.9 bracket the median, showing the prediction interval


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_grad_smooth_quantile.py


In this example, the noise variance depends on the first covariate (heteroscedastic design).
By fitting :math:`\tau \in \{0.1, 0.5, 0.9\}`, we obtain a prediction interval. The coefficient
:math:`x_0` shows the largest spread across quantiles (about 0.74), confirming that the first
covariate drives the conditional variance. The median regression (:math:`\tau = 0.5`) is also
more robust than ordinary least squares — it is not pulled by outliers the way the mean is.
