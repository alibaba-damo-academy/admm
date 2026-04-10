.. include:: ../definition.hrst

.. _udf-example-l0-ball:

L0 Ball Indicator
=================

This example shows how to model a hard sparsity budget with an indicator UDF. The full problem is

.. math::

   \min\limits_x \; \tfrac{1}{2}\|x-y\|_2^2 + \delta_{\{\|x\|_0 \le k\}}(x),

where :math:`\delta_{\{\|x\|_0 \le k\}}` is the indicator of the cardinality-constrained set
:math:`\{x : \|x\|_0 \le k\}`.

Unlike the :math:`L_0` penalty, this UDF does not assign a gradual sparsity cost. It says the iterate is either
feasible, with at most :math:`k` nonzeros, or infeasible. This is a nonconvex constraint, so the solver acts as a
practical local method and the result should be interpreted as a locally optimal solution or stationary point.

The value returned by :py:meth:`UDFBase.eval` is the indicator itself:

.. math::

   f(x) = \delta_{\{\|x\|_0 \le k\}}(x)
   =
   \begin{cases}
      0, & \|x\|_0 \le k, \\
      +\infty, & \|x\|_0 > k.
   \end{cases}

So ``eval`` only checks whether the current point satisfies the sparsity budget:

.. code-block:: python

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return 0.0 if np.count_nonzero(np.abs(x) > 1e-12) <= self.k else float("inf")

The proximal step returned by :py:meth:`UDFBase.argmin` is Euclidean projection onto the :math:`L_0` ball:

.. math::

   \operatorname{prox}_{\delta_{\{\|x\|_0 \le k\}}}(v)
   = \Pi_{\{\|x\|_0 \le k\}}(v),

which keeps the :math:`k` entries of largest magnitude and sets the rest to zero. That is exactly what the code does:

.. code-block:: python

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        prox = np.zeros_like(v)
        keep_count = min(max(self.k, 0), v.size)
        if keep_count > 0:
            keep_idx = np.argpartition(np.abs(v), -keep_count)[-keep_count:]
            prox[keep_idx] = v[keep_idx]
        return [prox.tolist()]

The :py:meth:`UDFBase.arguments` method tells |ADMM| that this UDF depends on one symbolic vector:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    class L0BallIndicator(admm.UDFBase):
        def __init__(self, arg, k=2):
            self.arg = arg
            self.k = k

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            return 0.0 if np.count_nonzero(np.abs(x) > 1e-12) <= self.k else float("inf")

        def argmin(self, lamb, arglist):
            v = np.asarray(arglist[0], dtype=float)
            prox = np.zeros_like(v)
            keep_count = min(max(self.k, 0), v.size)
            if keep_count > 0:
                keep_idx = np.argpartition(np.abs(v), -keep_count)[-keep_count:]
                prox[keep_idx] = v[keep_idx]
            return [prox.tolist()]

    y = np.array([0.2, -1.5, 0.7, 3.0])

    model = admm.Model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + L0BallIndicator(x, k=2))
    model.optimize()

    print(" * x: ", np.asarray(x.X))  # Expected: ≈ [0, -1.5, 0, 3]
    print(" * model.ObjVal: ", model.ObjVal)  # Expected: ≈ 0.265


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_l0_ball.py


In this concrete example, :math:`k = 2`, so the projection keeps only the two largest entries in magnitude.
First compute the magnitudes:

.. math::

   |y| = [0.2,\; 1.5,\; 0.7,\; 3.0].

Hence the selected support is

.. math::

   S = \{i : |y_i| \text{ is among the two largest entries of } |y|\} = \{2,4\}.

The projection onto the :math:`L_0` ball keeps :math:`y_i` on that support and zeros out the complement:

.. math::

   x_i^\star =
   \begin{cases}
      y_i, & i \in S, \\
      0, & i \notin S,
   \end{cases}
   \qquad
   x^\star = [0,\; -1.5,\; 0,\; 3].

The two largest-magnitude coordinates are kept; the rest are zeroed out.