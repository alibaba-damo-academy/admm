.. include:: ../definition.hrst

.. _udf-example-simplex:

The Simplex Indicator
=====================

Constraining a vector to lie in a simplex is common in probability modeling, assignment relaxations,
and nonnegative mixture weights.
This example shows how to encode that feasible set with an indicator UDF:

.. math::

   \min\limits_x \; \tfrac{1}{2}\|x-y\|_2^2 + \delta_{\Delta_r}(x),

where

.. math::

   \Delta_r = \{x \in \mathbb{R}^n : x_i \ge 0,\; \sum_i x_i = r\}

is the simplex of radius :math:`r`.

This is a convex set and could also be modeled with standard DCP tools, but it serves as a useful
projection-style UDF example.

The value returned by :py:meth:`UDFBase.eval` is the simplex indicator:

.. math::

   f(x) = \delta_{\Delta_r}(x)
   =
   \begin{cases}
      0, & x_i \ge 0 \text{ for all } i \text{ and } \sum_i x_i = r, \\
      +\infty, & \text{otherwise}.
   \end{cases}

So ``eval`` checks nonnegativity and the sum constraint:

.. code-block:: python

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        if np.min(x) >= -1e-9 and abs(np.sum(x) - self.radius) <= 1e-9:
            return 0.0
        return float("inf")

The proximal operator returned by :py:meth:`UDFBase.argmin` is the Euclidean projection onto the simplex:

.. math::

   \operatorname{prox}_{\delta_{\Delta_r}}(v)
   =
   \Pi_{\Delta_r}(v),
   \qquad
   \bigl(\Pi_{\Delta_r}(v)\bigr)_i = \max(v_i - \theta, 0),

where :math:`\theta` is chosen so that :math:`\sum_i \max(v_i - \theta, 0) = r`.

The code computes this projection with the standard sorting-based algorithm:

.. code-block:: python

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        sorted_v = np.sort(v)[::-1]
        cumulative = np.cumsum(sorted_v) - self.radius
        indices = np.arange(1, len(v) + 1)
        rho = np.nonzero(sorted_v - cumulative / indices > 0)[0][-1]
        theta = cumulative[rho] / (rho + 1)
        prox = np.maximum(v - theta, 0.0)
        return [prox.tolist()]

The :py:meth:`UDFBase.arguments` method says that this indicator depends on one symbolic vector:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    class SimplexIndicator(admm.UDFBase):
        def __init__(self, arg, radius=1.0):
            self.arg = arg
            self.radius = radius

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            if np.min(x) >= -1e-9 and abs(np.sum(x) - self.radius) <= 1e-9:
                return 0.0
            return float("inf")

        def argmin(self, lamb, arglist):
            v = np.asarray(arglist[0], dtype=float)
            sorted_v = np.sort(v)[::-1]
            cumulative = np.cumsum(sorted_v) - self.radius
            indices = np.arange(1, len(v) + 1)
            rho = np.nonzero(sorted_v - cumulative / indices > 0)[0][-1]
            theta = cumulative[rho] / (rho + 1)
            prox = np.maximum(v - theta, 0.0)
            return [prox.tolist()]

    y = np.array([0.2, -0.1, 0.7])

    model = admm.Model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + SimplexIndicator(x, 1.0))
    model.optimize()

    print(" * x: ", np.asarray(x.X))  # Expected: ≈ [0.25, 0, 0.75]
    print(" * model.ObjVal: ", model.ObjVal)  # Expected: ≈ 0.0075


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_simplex.py


In this concrete example, the projection shifts all coordinates by a common offset :math:`\theta` and then clips
negative values to zero. Solving the simplex-balance equation gives

.. math::

   \theta = -0.05,
   \qquad
   \sum_i \max(y_i - \theta, 0) = 1.

Therefore

.. math::

   \begin{aligned}
   x^\star
   &= \max([0.2,\; -0.1,\; 0.7] - (-0.05), 0) \\
   &= \max([0.25,\; -0.05,\; 0.75], 0) \\
   &= [0.25,\; 0,\; 0.75].
   \end{aligned}

This point is feasible because it is componentwise nonnegative and its entries sum to one:

.. math::

   x^\star \in \Delta_1,
   \qquad
   \delta_{\Delta_1}(x^\star) = 0.

Projection onto the simplex reduces to a common shift plus clipping to zero.