.. include:: ../definition.hrst

.. _udf-grad-example-smooth-tv:

Smooth Total Variation Denoising
================================

This example shows how to package a **smooth total variation** (TV) penalty as a gradient-based
UDF. The full model is

.. math::

   \begin{array}{ll}
   \min\limits_x & \tfrac{1}{2}\|x - y\|_2^2
     \;+\; \lambda \displaystyle\sum_{i=1}^{n-1} \sqrt{(x_{i+1} - x_i)^2 + \varepsilon}.
   \end{array}

Here the first term keeps :math:`x` close to the noisy observation :math:`y`, and the second term
is a smooth approximation to the standard TV penalty :math:`\sum_i |x_{i+1} - x_i|`. The
smoothing parameter :math:`\varepsilon > 0` replaces the non-differentiable absolute value with a
differentiable function, making this a natural fit for the ``grad`` UDF path.

Standard TV is one of the most widely used regularizers in signal and image processing for
edge-preserving denoising: it penalizes total variation (sum of absolute differences) while
allowing sharp transitions. The smooth variant preserves this behavior while enabling gradient
computation.

The function value returned by :py:meth:`UDFBase.eval` computes the smooth TV over consecutive
differences:

.. math::

   f(x) = \sum_{i=1}^{n-1} \sqrt{d_i^2 + \varepsilon}, \qquad d_i = x_{i+1} - x_i.

So ``eval`` uses ``np.diff`` to compute the finite differences and then sums the smoothed
absolute values:

.. code-block:: python

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float).ravel()
        d = np.diff(x)
        return float(np.sum(np.sqrt(d ** 2 + self.eps)))

The gradient returned by :py:meth:`UDFBase.grad` involves *neighboring elements* — this is not
an elementwise operation. Using :math:`\tilde{d}_i = d_i / \sqrt{d_i^2 + \varepsilon}`, the
gradient at position :math:`i` is:

.. math::

   g_i =
   \begin{cases}
     -\tilde{d}_1, & i = 1, \\
     \tilde{d}_{i-1} - \tilde{d}_i, & 2 \le i \le n{-}1, \\
     \tilde{d}_{n-1}, & i = n.
   \end{cases}

Each interior element receives a contribution from the difference to its left (:math:`+\tilde{d}_{i-1}`)
and the difference to its right (:math:`-\tilde{d}_i`). The implementation uses array slicing:

.. code-block:: python

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float).ravel()
        d = np.diff(x)
        dd = d / np.sqrt(d ** 2 + self.eps)
        g = np.zeros_like(x)
        g[:-1] -= dd   # contribution from d_i to x_i
        g[1:]  += dd   # contribution from d_i to x_{i+1}
        return [g.reshape(arglist[0].shape)]

This demonstrates that ``grad``-based UDFs are **not limited to coordinatewise operations** —
they can encode any differentiable function of the full variable vector, including structural
relationships between neighboring elements.

The :py:meth:`UDFBase.arguments` method returns the single variable this UDF depends on:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import admm
    import numpy as np

    class SmoothTV(admm.UDFBase):
        """Smooth total variation: f(x) = sum(sqrt((x_{i+1}-x_i)^2 + eps)).

        A differentiable approximation to sum(|x_{i+1} - x_i|).
        The gradient involves finite differences between neighboring elements.

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
            g[1:]  += dd
            return [g.reshape(arglist[0].shape)]

    # Piecewise-constant signal with noise
    np.random.seed(99)
    n = 100
    signal_true = np.zeros(n)
    signal_true[0:25] = 1.0
    signal_true[25:50] = 3.0
    signal_true[50:75] = 0.5
    signal_true[75:100] = 2.0
    y = signal_true + 0.5 * np.random.randn(n)

    model = admm.Model()
    x = admm.Var("x", n)
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + 2.0 * SmoothTV(x))
    model.optimize()

    x_val = np.asarray(x.X).ravel()
    mse_noisy = np.mean((y - signal_true) ** 2)
    mse_denoised = np.mean((x_val - signal_true) ** 2)
    print(" * status:", model.StatusString)                    # Expected: SOLVE_OPT_SUCCESS
    print(f" * MSE (noisy):    {mse_noisy:.6f}")               # Expected: ≈ 0.25
    print(f" * MSE (denoised): {mse_denoised:.6f}")            # Expected: ≈ 0.03
    print(f" * MSE reduction:  {mse_noisy / mse_denoised:.1f}x")  # Expected: ≈ 8x


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_grad_smooth_tv.py


In this concrete example, the true signal has 4 constant segments at levels :math:`[1, 3, 0.5, 2]`
corrupted by Gaussian noise (:math:`\sigma = 0.5`). The smooth TV penalty reduces the mean squared
error by approximately **8x** (from 0.25 to 0.03). At the center of each constant segment, the
denoised signal recovers the true level to within 0.01–0.05, demonstrating effective
edge-preserving denoising.
