.. include:: ../definition.hrst

.. _example-huber-regression:

Huber Regression
================

Huber regression is designed for data fitting when we want to stay sensitive to
small errors but reduce the influence of large outliers.

The standard model is

.. math::

   \begin{array}{ll}
   \min\limits_\beta & \sum_{i=1}^{m} \phi\!\left(x_i^\top \beta - y_i\right),
   \end{array}

where the Huber loss is defined piecewise by

.. math::

   \phi(u)=
   \begin{cases}
   \tfrac12 u^2, & |u| \le M, \\
   M|u| - \tfrac12 M^2, & |u| > M.
   \end{cases}

Here :math:`\beta \in \mathbb{R}^n` is the regression vector,
:math:`x_i^\top` is the :math:`i`-th row of the data matrix :math:`X`,
:math:`y_i` is the corresponding observation, and :math:`M > 0` is the Huber
threshold that marks where the loss changes from quadratic behavior to linear
growth.

The piecewise definition is the main idea:

- For small residuals, the loss is quadratic, just like least squares.
- For large residuals, the loss becomes linear, so extreme outliers do not grow as aggressively.

**Step 1: Generate data with outliers**

We first create a standard linear regression problem, then deliberately corrupt the
first 8 observations with much larger noise. That gives Huber regression a reason
to differ from plain least squares.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 80
    n = 20
    X = np.random.randn(m, n)
    beta_true = np.random.randn(n)
    y = X @ beta_true + 0.1 * np.random.randn(m)
    y[:8] += 8.0 * np.random.randn(8)

**Step 2: Create the model and coefficient variable**

The unknown quantity is again a coefficient vector ``beta``.

.. code-block:: python

    model = admm.Model()
    beta = admm.Var("beta", n)

**Step 3: Write the residual vector and Huber objective**

As in ordinary regression, the residual compares prediction and observation. In
the symbolic API, ``admm.huber(residual, 1.0)`` means "apply the Huber function
with threshold :math:`M = 1.0` to each component of ``residual``." Then
``admm.sum(...)`` adds those per-observation losses together.

.. code-block:: python

    residual = X @ beta - y
    model.setObjective(admm.sum(admm.huber(residual, 1.0)))

So the code directly encodes the piecewise formula above, but in vectorized form.

**Step 4: Add constraints**

This Huber-regression example has no explicit constraints, so there are no
``model.addConstr(...)`` calls. Once the objective is set, the model is complete.

**Step 5: Solve and inspect the result**

After solving, we print the standard solver outputs.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 36.570113991744364
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 80
    n = 20
    X = np.random.randn(m, n)
    beta_true = np.random.randn(n)
    y = X @ beta_true + 0.1 * np.random.randn(m)
    y[:8] += 8.0 * np.random.randn(8)

    model = admm.Model()
    beta = admm.Var("beta", n)
    residual = X @ beta - y
    model.setObjective(admm.sum(admm.huber(residual, 1.0)))
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 36.570113991744364
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/huber_regression.py


The Huber loss is quadratic for small residuals and linear for large ones, so outliers
no longer dominate the fit. ADMM handles this mixed-curvature loss natively via
:py:func:`admm.huber`.