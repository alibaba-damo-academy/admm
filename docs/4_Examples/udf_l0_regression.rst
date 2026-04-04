.. include:: ../definition.hrst

.. _udf-example-l0-regression:

L0-Regularized Regression
=========================

This example demonstrates combining a UDF with linear constraints — the most common
real-world UDF pattern. Unlike the denoising examples above, the data-fitting term here
involves a sensing matrix :math:`A`, showing that UDFs work naturally alongside standard
affine structure.

Consider the sparse regression problem

.. math::

   \begin{array}{ll}
   \min\limits_x & \tfrac{1}{2}\|Ax - b\|_2^2 + \lambda \|x\|_0 \\
   \text{s.t.} & x \ge 0.
   \end{array}

Here :math:`A \in \mathbb{R}^{m \times n}` is a sensing matrix, :math:`b \in \mathbb{R}^m` is an
observation vector, and :math:`\|x\|_0` counts nonzero entries. The L0 penalty promotes sparse solutions
while the nonnegativity constraint restricts the feasible set.

This is a nonconvex problem. The solver acts as a practical local method, so the result should be
interpreted as a locally optimal solution or stationary point.

The custom UDF still represents the :math:`L_0` count:

.. math::

   f(x) = \|x\|_0 = \#\{i : x_i \neq 0\}.

So :py:meth:`UDFBase.eval` is identical to the basic ``L0`` example:

.. code-block:: python

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.count_nonzero(np.abs(x) > 1e-12))

The proximal step returned by :py:meth:`UDFBase.argmin` is also the same hard-thresholding rule:

.. math::

   \operatorname{prox}_{\lambda \|\cdot\|_0}(v)
   = \operatorname*{argmin}_x \; \lambda \|x\|_0 + \tfrac{1}{2}\|x-v\|_2^2,
   \qquad
   x_i^\star =
   \begin{cases}
      0, & |v_i| \le \sqrt{2\lambda}, \\
      v_i, & |v_i| > \sqrt{2\lambda}.
   \end{cases}

So the code for ``argmin`` is the same as in the denoising case:

.. code-block:: python

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        threshold = np.sqrt(2.0 * lamb)
        prox = np.where(np.abs(v) <= threshold, 0.0, v)
        return [prox.tolist()]

The only bookkeeping method is :py:meth:`UDFBase.arguments`, which tells |ADMM| that the custom term depends on
the symbolic vector :math:`x`:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    class L0Norm(admm.UDFBase):
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            return float(np.count_nonzero(np.abs(x) > 1e-12))

        def argmin(self, lamb, arglist):
            v = np.asarray(arglist[0], dtype=float)
            threshold = np.sqrt(2.0 * lamb)
            prox = np.where(np.abs(v) <= threshold, 0.0, v)
            return [prox.tolist()]

    np.random.seed(42)
    n, m, k = 20, 30, 3
    x_true = np.zeros(n)
    x_true[np.random.choice(n, k, replace=False)] = np.random.rand(k) * 2 + 0.5
    A = np.random.randn(m, n)
    b = A @ x_true + 0.01 * np.random.randn(m)
    lam = 0.5

    model = admm.Model()
    x = admm.Var("x", n)
    model.setObjective(0.5 * admm.sum(admm.square(A @ x - b)) + lam * L0Norm(x))
    model.addConstr(x >= 0)
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: ≈ 1.50
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * nnz(x): ", np.count_nonzero(np.abs(np.asarray(x.X)) > 1e-6))  # Expected: 3


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_l0_regression.py


The key teaching point is that only the sparsity term is custom. The least-squares term
:math:`\tfrac{1}{2}\|Ax-b\|_2^2` and the nonnegativity constraint :math:`x \ge 0` are standard built-in pieces.
This is often the best way to use UDFs in practice: keep the global model structure in ordinary |ADMM| atoms,
and only implement the one missing proximal block yourself.

The solver recovers a nonnegative vector with exactly three active coordinates, matching
the planted sparsity level.