.. include:: ../definition.hrst

.. _example-ridge-regression:

Ridge Regression
================

Ridge regression starts from least squares and then adds an
:math:`\ell_2` penalty to discourage large coefficients.

The standard model is

.. math::

   \begin{array}{ll}
   \min\limits_\beta & \|X \beta - y\|_2^2 + \lambda \|\beta\|_2^2.
   \end{array}

Here :math:`\beta \in \mathbb{R}^n` is the coefficient vector,
:math:`X \in \mathbb{R}^{m \times n}` is the data matrix,
:math:`y \in \mathbb{R}^m` is the response vector, and :math:`\lambda \ge 0`
controls how strongly we shrink the coefficients toward zero.

The objective has two distinct pieces:

- The data-fit term :math:`\|X \beta - y\|_2^2` tries to match the observations.
- The shrinkage term :math:`\lambda \|\beta\|_2^2` penalizes large coefficient values.

**Step 1: Generate synthetic regression data**

We build a random feature matrix ``X``, a hidden coefficient vector
``beta_true``, and a noisy response vector ``y``. We also choose a regularization
weight ``lam``.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 100
    n = 25
    X = np.random.randn(m, n)
    beta_true = np.random.randn(n)
    y = X @ beta_true + 0.5 * np.random.randn(m)
    lam = 1.0

**Step 2: Create the model and coefficient variable**

The unknown quantity is the coefficient vector :math:`\beta`.

.. code-block:: python

    model = admm.Model()
    beta = admm.Var("beta", n)

**Step 3: Write the objective terms**

The prediction is ``X @ beta`` and the residual is ``X @ beta - y``. Squaring and
summing those residual entries gives the least-squares part of the objective. We
then add the ridge penalty, which depends directly on the size of ``beta``.

.. code-block:: python

    data_fit = admm.sum(admm.square(X @ beta - y))
    shrinkage = lam * admm.sum(admm.square(beta))
    model.setObjective(data_fit + shrinkage)

The first term rewards explaining the observed data well, while the second
penalizes large coefficients. That separation is the main modeling idea in ridge
regression.

**Step 4: Add constraints**

This ridge-regression example has no explicit constraints, so there are no
``model.addConstr(...)`` calls. The optimization model is determined entirely by
the objective above.

**Step 5: Solve and inspect the result**

Now we solve the model and print the standard solver outputs.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 37.12175719889099
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 100
    n = 25
    X = np.random.randn(m, n)
    beta_true = np.random.randn(n)
    y = X @ beta_true + 0.5 * np.random.randn(m)
    lam = 1.0

    model = admm.Model()
    beta = admm.Var("beta", n)
    model.setObjective(admm.sum(admm.square(X @ beta - y)) + lam * admm.sum(admm.square(beta)))
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 37.12175719889099
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/ridge_regression.py


The :math:`\ell_2` penalty shrinks coefficients toward zero without eliminating them,
yielding a more stable solution than plain least squares — especially when columns
of :math:`X` are correlated.