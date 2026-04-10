.. include:: ../definition.hrst

.. _udf-example-rank:

Matrix Rank Function
====================

This example shows how to package the nonconvex matrix rank function as a UDF. The model is

.. math::

   \min\limits_X \; \tfrac{1}{2}\|X-Y\|_F^2 + \lambda \,\operatorname{rank}(X),

where :math:`\|\cdot\|_F` is the Frobenius norm.
The rank penalty promotes low-rank structure, but it is nonconvex; the solver therefore acts as a practical
local method and the result should be interpreted as a locally optimal solution or stationary point.

The value returned by :py:meth:`UDFBase.eval` is the matrix rank itself:

.. math::

   f(X) = \operatorname{rank}(X) = \#\{i : \sigma_i(X) \neq 0\},

where :math:`\sigma_i(X)` are the singular values of :math:`X`. So ``eval`` computes the singular values and
counts the nonzero ones:

.. code-block:: python

    def eval(self, arglist):
        X = np.asarray(arglist[0], dtype=float)
        singular_v = np.linalg.svd(X, compute_uv=False)
        return float(np.sum(singular_v > 1e-10))

If :math:`Z = U \Sigma V^\top` is a singular value decomposition with
:math:`\Sigma = \operatorname{diag}(\sigma_i)`, then the proximal step returned by
:py:meth:`UDFBase.argmin` is

.. math::

   \operatorname{prox}_{\lambda \operatorname{rank}}(Z)
   =
   U \operatorname{diag}(\sigma_i') V^\top,
   \qquad
   \sigma_i' =
   \begin{cases}
   \sigma_i, & \sigma_i > \sqrt{2\lambda}, \\
   0, & \sigma_i \le \sqrt{2\lambda}.
   \end{cases}

So the proximal operator performs hard thresholding on singular values:

.. code-block:: python

    def argmin(self, lamb, arglist):
        Z = np.asarray(arglist[0], dtype=float)
        u, singular_v, vt = np.linalg.svd(Z, full_matrices=False)
        threshold = np.sqrt(2.0 * lamb)
        singular_v = np.where(singular_v <= threshold, 0.0, singular_v)
        prox = (u * singular_v) @ vt
        return [prox.tolist()]

The :py:meth:`UDFBase.arguments` method says that this custom function depends on one symbolic matrix:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    class RankPenalty(admm.UDFBase):
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            X = np.asarray(arglist[0], dtype=float)
            singular_v = np.linalg.svd(X, compute_uv=False)
            return float(np.sum(singular_v > 1e-10))

        def argmin(self, lamb, arglist):
            Z = np.asarray(arglist[0], dtype=float)
            u, singular_v, vt = np.linalg.svd(Z, full_matrices=False)
            threshold = np.sqrt(2.0 * lamb)
            singular_v = np.where(singular_v <= threshold, 0.0, singular_v)
            prox = (u * singular_v) @ vt
            return [prox.tolist()]

    Y = np.array([[2.0, 0.0], [0.0, 0.5]])
    lam = 0.5

    model = admm.Model()
    X = admm.Var("X", 2, 2)
    model.setObjective(0.5 * admm.sum(admm.square(X - Y)) + lam * RankPenalty(X))
    model.optimize()

    print(" * X: ", np.asarray(X.X))  # Expected: ≈ [[2, 0], [0, 0]]
    print(" * model.ObjVal: ", model.ObjVal)  # Expected: ≈ 0.625


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_matrix_rank.py


In this concrete example, :math:`\lambda = 0.5`, so the singular-value threshold is

.. math::

   \sqrt{2\lambda} = \sqrt{1} = 1.

The matrix :math:`Y = \operatorname{diag}(2, 0.5)` has singular values :math:`2` and :math:`0.5`. Applying the
hard threshold keeps the first singular value and discards the second, so

.. math::

   X^\star =
   \operatorname{diag}(2, 0),
   \qquad
   \operatorname{rank}(X^\star) = 1.

The smaller singular direction is discarded, leaving a rank-one matrix.