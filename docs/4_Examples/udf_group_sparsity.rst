.. include:: ../definition.hrst

.. _udf-example-group-sparsity:

Group Sparsity
==============

This example implements a simple group sparsity penalty with a matrix-valued UDF. If the columns of
:math:`X` are treated as groups, the model is

.. math::

   \min\limits_X \; \tfrac{1}{2}\|X-Y\|_F^2 + \lambda \sum_j \mathbf{1}_{\{\|X_{:,j}\|_2 \neq 0\}}.

This is a column-group analogue of scalar :math:`\ell_0` regularization and is nonconvex.

The value returned by :py:meth:`UDFBase.eval` counts how many column-groups are active:

.. math::

   G(X) = \sum_j \mathbf{1}_{\{\|X_{:,j}\|_2 \neq 0\}}.

So ``eval`` computes the Euclidean norm of each column and counts the nonzero ones:

.. code-block:: python

    def eval(self, arglist):
        X = np.asarray(arglist[0], dtype=float)
        column_norms = np.linalg.norm(X, axis=0)
        return float(np.count_nonzero(column_norms > 1e-12))

The proximal operator returned by :py:meth:`UDFBase.argmin` acts groupwise:

.. math::

   \bigl(\operatorname{prox}_{\lambda G}(V)\bigr)_{:,j}
   =
   \begin{cases}
   0, & \|V_{:,j}\|_2^2 \le 2\lambda, \\
   V_{:,j}, & \|V_{:,j}\|_2^2 > 2\lambda,
   \end{cases}

where :math:`G(X) = \lambda \sum_j \mathbf{1}_{\{\|X_{:,j}\|_2 \neq 0\}}`.

So each whole column-group is either dropped or kept. This is a nonconvex model; the solver acts as a
practical local method and should not be expected to certify global optimality.

The code mirrors that group hard-thresholding rule:

.. code-block:: python

    def argmin(self, lamb, arglist):
        Z = np.asarray(arglist[0], dtype=float)
        column_norm_sq = np.sum(Z * Z, axis=0)
        keep_mask = column_norm_sq > 2.0 * lamb
        prox = Z * keep_mask[np.newaxis, :]
        return [prox.tolist()]

The :py:meth:`UDFBase.arguments` method simply binds the UDF to the symbolic matrix variable:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    class GroupSparsityPenalty(admm.UDFBase):
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            X = np.asarray(arglist[0], dtype=float)
            column_norms = np.linalg.norm(X, axis=0)
            return float(np.count_nonzero(column_norms > 1e-12))

        def argmin(self, lamb, arglist):
            Z = np.asarray(arglist[0], dtype=float)
            column_norm_sq = np.sum(Z * Z, axis=0)
            keep_mask = column_norm_sq > 2.0 * lamb
            prox = Z * keep_mask[np.newaxis, :]
            return [prox.tolist()]

    Y = np.array([[0.2, 2.0, 0.3], [0.1, 1.0, 0.4]])
    lam = 1.0

    model = admm.Model()
    X = admm.Var("X", 2, 3)
    model.setObjective(0.5 * admm.sum(admm.square(X - Y)) + lam * GroupSparsityPenalty(X))
    model.optimize()

    print(" * X: ", np.asarray(X.X))  # Expected: ≈ [[0, 2, 0], [0, 1, 0]]
    print(" * model.ObjVal: ", model.ObjVal)  # Expected: ≈ 1.15


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_group_sparsity.py


In this concrete example, the squared column norms of :math:`Y` are

.. math::

   \begin{aligned}
   \|Y_{:,1}\|_2^2 &= 0.2^2 + 0.1^2 = 0.05, \\
   \|Y_{:,2}\|_2^2 &= 2^2 + 1^2 = 5, \\
   \|Y_{:,3}\|_2^2 &= 0.3^2 + 0.4^2 = 0.25.
   \end{aligned}

With :math:`\lambda = 1`, the group threshold is :math:`2\lambda = 2`, so only the middle column remains active.
The projected matrix is therefore

.. math::

   X^\star =
   \begin{bmatrix}
   0 & 2 & 0 \\
   0 & 1 & 0
   \end{bmatrix}.

Only the middle column survives; the two smaller-norm groups are zeroed as whole blocks.