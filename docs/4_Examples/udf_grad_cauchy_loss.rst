.. include:: ../definition.hrst

.. _udf-grad-example-cauchy:

Cauchy Loss Robust Regression
=============================

This example shows how to package the **Cauchy (Lorentzian) loss** as a gradient-based UDF. The
full model is

.. math::

   \begin{array}{ll}
   \min\limits_x & \displaystyle\sum_{i=1}^{n} \log\!\Bigl(1 + \bigl(\tfrac{a_i^\top x - b_i}{c}\bigr)^{\!2}\Bigr)
                   \;+\; \tfrac{\lambda}{2}\|x\|_2^2.
   \end{array}

Here :math:`c > 0` is a scale parameter controlling the transition from quadratic to logarithmic
behavior. The Cauchy loss is a heavy-tailed loss function from robust statistics whose influence
function is *bounded and redescending* — as residuals grow larger, their effect on the fit actually
*decreases*. This makes it exceptionally resistant to gross outliers.

The function value returned by :py:meth:`UDFBase.eval` is the sum of the Cauchy loss over all
residuals:

.. math::

   f(r) = \sum_i \log\!\bigl(1 + (r_i / c)^2\bigr).

Near zero the Cauchy loss behaves like :math:`r^2/c^2` (quadratic), but for large :math:`|r|` it
grows only logarithmically — much slower than the quadratic growth of L2.
So ``eval`` computes the sum:

.. code-block:: python

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return float(np.sum(np.log(1 + (r / self.c) ** 2)))

The gradient returned by :py:meth:`UDFBase.grad` is

.. math::

   \nabla f(r)_i = \frac{2 r_i}{c^2 + r_i^2},

which tends to **zero** as :math:`|r_i| \to \infty`. This is the redescending property: very large
residuals are *downweighted to zero* — the outlier effectively removes itself from the fit. Compare
this with L2 (gradient :math:`2r`, unbounded), Huber (gradient clipped to :math:`\pm 1`, bounded
but constant), or even log-cosh (gradient :math:`\tanh(r)`, bounded but not redescending).

The implementation is a single expression:

.. code-block:: python

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return [2.0 * r / (self.c ** 2 + r ** 2)]

The :py:meth:`UDFBase.arguments` method tells |ADMM| which symbolic object this UDF depends on:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Note that the scale parameter :math:`c` is stored as an instance attribute during ``__init__``
and does **not** appear in ``arguments`` — only the optimization variable (or expression) does.

Complete runnable example:

.. code-block:: python

    import admm
    import numpy as np

    class CauchyLoss(admm.UDFBase):
        """Cauchy/Lorentzian loss: f(r) = sum(log(1 + (r_i/c)^2)).

        Properties:
            - Quadratic near zero: ≈ r^2/c^2
            - Logarithmic for large |r|: ≈ 2*log(|r|/c)
            - Redescending gradient: 2r/(c^2+r^2) → 0 as |r| → ∞

        Parameters
        ----------
        arg : admm.Var or expression
            The residual vector.
        c : float
            Scale parameter. Smaller c = more aggressive outlier rejection.
        """
        def __init__(self, arg, c=1.0):
            self.arg = arg
            self.c = c

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            r = np.asarray(arglist[0], dtype=float)
            return float(np.sum(np.log(1 + (r / self.c) ** 2)))

        def grad(self, arglist):
            r = np.asarray(arglist[0], dtype=float)
            return [2.0 * r / (self.c ** 2 + r ** 2)]

    # Data with 25% gross outliers (magnitude 20–50)
    np.random.seed(7)
    n, p = 80, 5
    A = np.random.randn(n, p)
    x_true = np.array([3.0, -1.0, 2.0, 0.5, -0.5])
    b = A @ x_true + 0.2 * np.random.randn(n)
    outlier_idx = np.random.choice(n, size=20, replace=False)
    b[outlier_idx] += np.random.choice([-1, 1], size=20) * np.random.uniform(20, 50, size=20)

    model = admm.Model()
    x = admm.Var("x", p)
    residual = A @ x - b
    model.setObjective(CauchyLoss(residual, c=2.0) + 0.005 * admm.sum(admm.square(x)))
    model.optimize()

    print(" * status:", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * x:", np.asarray(x.X))          # Expected: ≈ [3, -1, 2, 0.5, -0.5]
    print(" * Cauchy error:", np.linalg.norm(np.asarray(x.X) - x_true))  # Expected: ≈ 0.07


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_grad_cauchy_loss.py


In this concrete example, 25% of observations are corrupted by outliers of magnitude 20–50
(far exceeding the signal level). Ordinary least squares gives
:math:`\|x_{\text{OLS}} - x_{\text{true}}\| \approx 4.1` because each outlier exerts force
proportional to its size. The Cauchy loss recovers the true coefficients to
:math:`\|x_{\text{Cauchy}} - x_{\text{true}}\| \approx 0.07` — a **59x improvement** —
because the redescending gradient
:math:`2r/(c^2 + r^2)` automatically suppresses the influence of any residual much
larger than the scale :math:`c`.
