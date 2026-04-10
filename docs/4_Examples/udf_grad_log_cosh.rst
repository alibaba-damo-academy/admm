.. include:: ../definition.hrst

.. _udf-grad-example-log-cosh:

Log-Cosh Robust Regression
==========================

This example shows how to package the **log-cosh loss** — a smooth, robust alternative to least
squares — as a gradient-based UDF. The full model is

.. math::

   \begin{array}{ll}
   \min\limits_x & \displaystyle\sum_{i=1}^{n} \log\bigl(\cosh(a_i^\top x - b_i)\bigr)
                   \;+\; \tfrac{\lambda}{2}\|x\|_2^2.
   \end{array}

Here :math:`\log(\cosh(r))` is applied elementwise to the residual vector :math:`r = Ax - b`.
The log-cosh function is a smooth approximation to the absolute value, so the objective trades off
a robust fit to the data against an L2 regularizer. Because the loss is everywhere smooth, we can
use the ``grad`` UDF path — no proximal formula needed.

The function value returned by :py:meth:`UDFBase.eval` is the sum of log-cosh over all residuals:

.. math::

   f(r) = \sum_i \log(\cosh(r_i)).

The log-cosh function interpolates between L2 and L1:

.. math::

   \log(\cosh(r))
   \;\approx\;
   \begin{cases}
     \tfrac{1}{2} r^2, & |r| \ll 1 \quad (\text{like L2 — quadratic for small residuals}), \\[4pt]
     |r| - \log 2,     & |r| \gg 1 \quad (\text{like L1 — linear for large residuals}).
   \end{cases}

So ``eval`` sums up the elementwise log-cosh values:

.. code-block:: python

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return float(np.sum(np.log(np.cosh(r))))

The gradient returned by :py:meth:`UDFBase.grad` is

.. math::

   \nabla f(r)_i = \tanh(r_i),

which is bounded in :math:`[-1, 1]`. This is the key to robustness: large residuals (outliers)
contribute a gradient of at most :math:`\pm 1`, while in ordinary least squares the gradient
:math:`2r_i` grows without bound. The implementation is a single line:

.. code-block:: python

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return [np.tanh(r)]

The :py:meth:`UDFBase.arguments` method tells |ADMM| which symbolic object this UDF depends on.
In this case the custom loss is a function of one expression (the residual vector), so
``arguments`` returns a one-element list:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import admm
    import numpy as np

    class LogCoshLoss(admm.UDFBase):
        """Log-cosh loss: f(r) = sum(log(cosh(r_i))).

        A smooth approximation to L1:
            log(cosh(r)) ≈ r^2/2 for small |r|
            log(cosh(r)) ≈ |r| - log(2) for large |r|

        Gradient: tanh(r), bounded in [-1, 1].
        """
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            r = np.asarray(arglist[0], dtype=float)
            return float(np.sum(np.log(np.cosh(r))))

        def grad(self, arglist):
            r = np.asarray(arglist[0], dtype=float)
            return [np.tanh(r)]

    # Data with 20% outliers
    np.random.seed(42)
    n, p = 50, 5
    A = np.random.randn(n, p)
    x_true = np.array([1.0, -2.0, 0.5, 0.0, 1.5])
    b = A @ x_true + 0.1 * np.random.randn(n)
    outlier_idx = np.random.choice(n, size=10, replace=False)
    b[outlier_idx] += np.random.choice([-1, 1], size=10) * np.random.uniform(8, 15, size=10)

    lam = 0.1
    model = admm.Model()
    x = admm.Var("x", p)
    residual = A @ x - b
    model.setObjective(LogCoshLoss(residual) + (lam / 2) * admm.sum(admm.square(x)))
    model.optimize()

    print(" * status:", model.StatusString)       # Expected: SOLVE_OPT_SUCCESS
    print(" * x:", np.asarray(x.X))               # Expected: ≈ [1, -2, 0.5, 0, 1.5]
    print(" * ||x - x_true||:", np.linalg.norm(np.asarray(x.X) - x_true))  # Expected: ≈ 0.26


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_grad_log_cosh.py


In this concrete example, 20% of the measurements are corrupted by outliers with magnitude
8–15. Ordinary least squares gives
:math:`\|x_{\text{OLS}} - x_{\text{true}}\|_2 \approx 2.07`
because the unbounded gradient :math:`2r_i` lets outliers dominate the fit.
Log-cosh regression recovers the true coefficients to
:math:`\|x_{\text{log-cosh}} - x_{\text{true}}\|_2 \approx 0.26`
— an 8x improvement — because the bounded gradient :math:`\tanh(r_i)` limits
each outlier's influence to at most :math:`\pm 1`.
