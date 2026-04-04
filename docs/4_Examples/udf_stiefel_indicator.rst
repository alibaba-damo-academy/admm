.. include:: ../definition.hrst

.. _udf-example-stiefel:

The Stiefel-Manifold Indicator
==============================

This example shows how to encode orthonormal-column structure with an indicator UDF. The model is

.. math::

   \min\limits_X \; \tfrac{1}{2}\|X-Y\|_F^2 + \delta_{\mathrm{St}(m,n)}(X),

where

.. math::

   \mathrm{St}(m,n) = \{X \in \mathbb{R}^{m \times n} : X^\top X = I_n\}, \qquad m \ge n.

This is the set of matrices with orthonormal columns, which is nonconvex. The solver acts as a practical
local method; the result should be interpreted as a locally optimal solution or stationary point.
When :math:`m = n`, the Stiefel manifold reduces to the orthogonal group
:math:`\mathcal{O}_n = \{X \in \mathbb{R}^{n \times n} : X^\top X = I\}`, so the square orthogonal-matrix
case is already covered as a special case of this example.

The value returned by :py:meth:`UDFBase.eval` is the indicator of the manifold:

.. math::

   f(X) = \delta_{\mathrm{St}(m,n)}(X)
   =
   \begin{cases}
      0, & X^\top X = I_n, \\
      +\infty, & \text{otherwise}.
   \end{cases}

So ``eval`` checks whether the columns are orthonormal:

.. code-block:: python

    def eval(self, arglist):
        X = np.asarray(arglist[0], dtype=float)
        identity = np.eye(X.shape[1])
        return 0.0 if np.linalg.norm(X.T @ X - identity) <= 1e-9 else float("inf")

The proximal operator returned by :py:meth:`UDFBase.argmin` is given by the polar factor: if
:math:`Z = U \Sigma V^\top`, then

.. math::

   \operatorname{prox}_{\delta_{\mathrm{St}(m,n)}}(Z) = U V^\top.

That is exactly what the code computes:

.. code-block:: python

    def argmin(self, lamb, arglist):
        Z = np.asarray(arglist[0], dtype=float)
        u, _, vt = np.linalg.svd(Z, full_matrices=False)
        prox = u @ vt
        return [prox.tolist()]

The :py:meth:`UDFBase.arguments` method again just binds the UDF to one symbolic matrix:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    class StiefelIndicator(admm.UDFBase):
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            X = np.asarray(arglist[0], dtype=float)
            identity = np.eye(X.shape[1])
            return 0.0 if np.linalg.norm(X.T @ X - identity) <= 1e-9 else float("inf")

        def argmin(self, lamb, arglist):
            Z = np.asarray(arglist[0], dtype=float)
            u, _, vt = np.linalg.svd(Z, full_matrices=False)
            prox = u @ vt
            return [prox.tolist()]

    Y = np.array([[2.0, 0.0], [0.0, 0.5], [0.0, 0.0]])

    model = admm.Model()
    X = admm.Var("X", 3, 2)
    model.setObjective(0.5 * admm.sum(admm.square(X - Y)) + StiefelIndicator(X))
    model.optimize()

    print(" * X: ", np.asarray(X.X))  # Expected: ≈ [[1, 0], [0, 1], [0, 0]]
    print(" * model.ObjVal: ", model.ObjVal)  # Expected: ≈ 0.624999


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_stiefel.py


In this concrete example, the data matrix already points mostly along the first two coordinate directions, so
the polar-factor projection returns the obvious orthonormal-column matrix

.. math::

   X^\star =
   \begin{bmatrix}
   1 & 0 \\
   0 & 1 \\
   0 & 0
   \end{bmatrix}.

It is easy to verify that this point lies on the Stiefel manifold:

.. math::

   (X^\star)^\top X^\star = I_2,
   \qquad
   \delta_{\mathrm{St}(3,2)}(X^\star) = 0.

The polar factor of the SVD produces the nearest matrix with orthonormal columns.