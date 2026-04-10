.. include:: ../definition.hrst

.. _udf-example-rank-r:

Rank-r Indicator
================

This example shows how to enforce a hard rank cap with an indicator UDF. The model is

.. math::

   \min\limits_X \; \tfrac{1}{2}\|X-Y\|_F^2 + \delta_{\{\operatorname{rank}(X)\le r\}}(X),

where :math:`\delta_{\{\operatorname{rank}(X)\le r\}}` is the indicator of the set of matrices with
rank at most :math:`r`.

This UDF does not softly penalize rank; it says the iterate is either feasible with rank at most :math:`r`
or infeasible. The constraint is nonconvex, so the solver acts as a practical local method and the result
should be interpreted as a locally optimal solution or stationary point.

The value returned by :py:meth:`UDFBase.eval` is the indicator itself:

.. math::

   f(X) = \delta_{\{\operatorname{rank}(X)\le r\}}(X)
   =
   \begin{cases}
      0, & \operatorname{rank}(X) \le r, \\
      +\infty, & \operatorname{rank}(X) > r.
   \end{cases}

So ``eval`` computes the singular values and checks whether too many are nonzero:

.. code-block:: python

    def eval(self, arglist):
        X = np.asarray(arglist[0], dtype=float)
        singular_v = np.linalg.svd(X, compute_uv=False)
        return 0.0 if np.sum(singular_v > 1e-10) <= self.rank_bound else float("inf")

The proximal operator returned by :py:meth:`UDFBase.argmin` is projection onto the set
:math:`\{\operatorname{rank}(X)\le r\}`. This is the truncated singular value decomposition:

.. math::

   \operatorname{prox}_{\delta_{\{\operatorname{rank}(X)\le r\}}}(Z)
   = \Pi_{\{\operatorname{rank}(X)\le r\}}(Z)
   = U \operatorname{diag}(\sigma_1,\ldots,\sigma_r,0,\ldots,0) V^\top.

So the code keeps the largest :math:`r` singular values and sets the rest to zero:

.. code-block:: python

    def argmin(self, lamb, arglist):
        Z = np.asarray(arglist[0], dtype=float)
        u, singular_v, vt = np.linalg.svd(Z, full_matrices=False)
        singular_v[min(self.rank_bound, len(singular_v)):] = 0.0
        prox = (u * singular_v) @ vt
        return [prox.tolist()]

The :py:meth:`UDFBase.arguments` method says that this UDF depends on one symbolic matrix:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    class RankRIndicator(admm.UDFBase):
        def __init__(self, arg, rank_bound=1):
            self.arg = arg
            self.rank_bound = rank_bound

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            X = np.asarray(arglist[0], dtype=float)
            singular_v = np.linalg.svd(X, compute_uv=False)
            return 0.0 if np.sum(singular_v > 1e-10) <= self.rank_bound else float("inf")

        def argmin(self, lamb, arglist):
            Z = np.asarray(arglist[0], dtype=float)
            u, singular_v, vt = np.linalg.svd(Z, full_matrices=False)
            singular_v[min(self.rank_bound, len(singular_v)) :] = 0.0
            prox = (u * singular_v) @ vt
            return [prox.tolist()]

    Y = np.array([[3.0, 0.0], [0.0, 1.0]])

    model = admm.Model()
    X = admm.Var("X", 2, 2)
    model.setObjective(0.5 * admm.sum(admm.square(X - Y)) + RankRIndicator(X, 1))
    model.optimize()

    print(" * X: ", np.asarray(X.X))  # Expected: ≈ [[3, 0], [0, 0]]
    print(" * model.ObjVal: ", model.ObjVal)  # Expected: ≈ 0.5


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_rank_r.py


In this concrete example, :math:`r = 1`, so the truncated SVD keeps only the largest singular value. Since
:math:`Y = \operatorname{diag}(3, 1)` already has singular values :math:`3` and :math:`1`, the projection
onto the rank-one set is

.. math::

   X^\star =
   \operatorname{diag}(3, 0),
   \qquad
   \operatorname{rank}(X^\star) = 1 \le r.

The smaller singular direction is truncated, leaving a rank-one matrix.