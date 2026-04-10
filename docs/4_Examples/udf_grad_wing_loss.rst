.. include:: ../definition.hrst

.. _udf-grad-example-wing-loss:

Wing Loss for Precise Regression
================================

This example shows how to package the **Wing loss** — a loss function from face landmark
localization [Feng2018]_ — as a gradient-based UDF. The full model is

.. math::

   \begin{array}{ll}
   \min\limits_\beta & \displaystyle\sum_{i=1}^{n}
     w \cdot \ln\!\Bigl(1 + \frac{\sqrt{(a_i^\top \beta - b_i)^2 + \delta^2}}{\varepsilon}\Bigr).
   \end{array}

Here :math:`w` controls the overall loss magnitude, :math:`\varepsilon` sets the
linear-to-logarithmic transition scale, and :math:`\delta > 0` smooths the loss at the origin.
Wing loss is designed to amplify attention to small errors while handling large errors gracefully
— useful when precision at *every* data point matters.

The function value returned by :py:meth:`UDFBase.eval` sums the Wing loss over all residuals.
To keep the function smooth at :math:`r = 0`, the squared residual is regularized by
:math:`\delta^2` inside the square root:

.. math::

   f(r) = \sum_i w \cdot \ln\!\Bigl(1 + \frac{\sqrt{r_i^2 + \delta^2}}{\varepsilon}\Bigr).

For small residuals :math:`|r| \ll \varepsilon`, this behaves like
:math:`\frac{w}{\varepsilon}|r|` (steep gradient), and for large residuals
:math:`|r| \gg \varepsilon`, it grows as :math:`w \cdot \ln(|r|/\varepsilon)` (logarithmic,
slower than L2). So ``eval`` computes:

.. code-block:: python

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        s = np.sqrt(r ** 2 + self.delta ** 2)
        return float(np.sum(self.w * np.log(1 + s / self.eps)))

The gradient returned by :py:meth:`UDFBase.grad` uses the chain rule through the square root
and the logarithm:

.. math::

   \nabla f(r)_i
   = w \cdot \frac{r_i}{\sqrt{r_i^2 + \delta^2}}
     \cdot \frac{1}{\varepsilon + \sqrt{r_i^2 + \delta^2}}.

Near zero, the first factor :math:`r/\sqrt{r^2+\delta^2}` is approximately :math:`r/\delta`
(linear), and the second factor is approximately :math:`1/(\varepsilon + \delta)` (constant),
so the gradient is proportional to :math:`r` with a steep coefficient :math:`w/(\delta(\varepsilon+\delta))`.
This steep gradient near zero is what makes Wing loss "pay more attention" to reducing small errors.
The implementation:

.. code-block:: python

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        s = np.sqrt(r ** 2 + self.delta ** 2)
        return [self.w * (r / s) / (self.eps + s)]

The :py:meth:`UDFBase.arguments` method returns the single expression this UDF depends on.
The hyperparameters :math:`w`, :math:`\varepsilon`, and :math:`\delta` are stored as instance
attributes and do not appear in ``arguments``:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import admm
    import numpy as np

    class WingLoss(admm.UDFBase):
        """Wing loss: f(r) = sum(w * ln(1 + sqrt(r^2 + delta^2) / eps)).

        Properties:
            - Near zero: gradient ≈ w*r / (delta*(eps+delta)) — steep
            - Large |r|: grows as w*ln(|r|/eps) — logarithmic
            - Everywhere smooth (delta > 0)

        Parameters
        ----------
        arg : admm.Var or expression
            The residual vector.
        w : float
            Loss magnitude.
        eps : float
            Transition scale.
        delta : float
            Smoothing at zero.
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

    # Regression data with moderate outliers
    np.random.seed(2026)
    n, p = 60, 5
    A = np.random.randn(n, p)
    beta_true = np.array([1.5, -0.8, 2.0, 0.3, -1.2])
    b = A @ beta_true + 0.3 * np.random.randn(n)
    outlier_idx = np.random.choice(n, size=6, replace=False)
    b[outlier_idx] += np.random.choice([-1, 1], size=6) * np.random.uniform(3, 6, size=6)

    model = admm.Model()
    beta = admm.Var("beta", p)
    model.setObjective(WingLoss(A @ beta - b, w=10.0, eps=2.0))
    model.setOption(admm.Options.admm_max_iteration, 5000)
    model.optimize()

    print(" * status:", model.StatusString)            # Expected: SOLVE_OPT_SUCCESS
    print(" * beta:", np.asarray(beta.X))              # Expected: ≈ [1.5, -0.8, 2, 0.3, -1.2]
    print(" * Wing error:", np.linalg.norm(np.asarray(beta.X) - beta_true))  # Expected: ≈ 0.13


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_grad_wing_loss.py


In this concrete example, Wing loss achieves
:math:`\|\beta_{\text{Wing}} - \beta_{\text{true}}\| \approx 0.13` compared to
:math:`\|\beta_{\text{OLS}} - \beta_{\text{true}}\| \approx 0.63` (OLS), a 5x improvement.
The median absolute residual is also halved (0.20 vs 0.43), confirming that Wing loss
prioritizes reducing residuals at *every* observation rather than just the largest ones.


.. [Feng2018] Z.-H. Feng, J. Kittler, M. Awais, P. Huber, X.-J. Wu.
   "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks."
   *CVPR*, 2018.
