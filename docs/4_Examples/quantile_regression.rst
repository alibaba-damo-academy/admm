.. include:: ../definition.hrst

.. _example-quantile-regression:

Quantile Regression
===================

Quantile regression fits a chosen conditional quantile of the response rather than the
conditional mean.

For quantile level :math:`\tau \in (0, 1)`, a convenient convex form is

.. math::

   \begin{array}{ll}
   \min\limits_w & \dfrac{1}{2}\|Xw - y\|_1 + \left(\dfrac{1}{2} - \tau\right)\mathbf{1}^\top (Xw - y).
   \end{array}

If we write the residual as :math:`r = Xw - y`, then this objective is the usual
quantile, or pinball, loss written as a symmetric absolute-value term plus an asymmetric
linear correction. When :math:`\tau = 0.5`, the model reduces to median regression.

For :math:`\tau = 0.9`, under-prediction is penalized more strongly than over-prediction,
so the fitted line is pushed upward toward the 90th conditional percentile.

**Step 1: Generate noisy regression data and choose a quantile**

We create a hidden vector ``beta`` only to synthesize data, then form a noisy response
vector ``y``. The quantile level ``tau = 0.9`` tells the model which part of the
conditional distribution we want to estimate.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 10
    m = 200
    beta = np.random.randn(n)
    X = np.random.randn(m, n)
    y = X @ beta + 0.5 * np.random.randn(m)
    tau = 0.9

**Step 2: Create the model, tune solver settings, and define the variable**

The fitted parameter vector is ``w``. We also raise ``admm_max_iteration`` and tighten
both termination thresholds to ``1e-5``. Those solver settings are part of the reference
example because this noisy quantile-regression instance otherwise tends to make the
iteration limit the story, while the tutorial is trying to show a standard successful
solve path.

.. code-block:: python

    model = admm.Model()
    model.setOption(admm.Options.admm_max_iteration, 10000)
    model.setOption(admm.Options.termination_absolute_error_threshold, 1e-5)
    model.setOption(admm.Options.termination_relative_error_threshold, 1e-5)
    w = admm.Var("w", n)

**Step 3: Write the residual expression and quantile objective**

The residual vector is ``X @ w - y``. The first objective term,
``0.5 * admm.norm(residual, ord=1)``, gives the symmetric absolute-value part. The second
term, ``(0.5 - tau) * admm.sum(residual)``, tilts that loss so over- and under-prediction
are not treated equally.

.. code-block:: python

    residual = X @ w - y
    model.setObjective(0.5 * admm.norm(residual, ord=1) + (0.5 - tau) * admm.sum(residual))

That combination is what turns ordinary absolute-deviation fitting into quantile
regression.

**Step 4: Add constraints**

This quantile-regression example has no explicit constraints, so there are no
``model.addConstr(...)`` calls. The optimization model is fully specified by the
objective.

**Step 5: Solve and inspect the result**

Now we optimize and print the standard solver outputs.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 35.76437758914047
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 10
    m = 200
    beta = np.random.randn(n)
    X = np.random.randn(m, n)
    y = X @ beta + 0.5 * np.random.randn(m)
    tau = 0.9

    model = admm.Model()
    model.setOption(admm.Options.admm_max_iteration, 10000)
    model.setOption(admm.Options.termination_absolute_error_threshold, 1e-5)
    model.setOption(admm.Options.termination_relative_error_threshold, 1e-5)
    w = admm.Var("w", n)
    residual = X @ w - y
    model.setObjective(0.5 * admm.norm(residual, ord=1) + (0.5 - tau) * admm.sum(residual))
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 35.76437758914047
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/quantile_regression.py


Unlike ordinary regression, quantile regression fits the :math:`\tau`-th conditional
quantile rather than the mean. The asymmetric pinball loss
:math:`\rho_\tau(r) = \tfrac{1}{2}\|r\|_1 + (\tfrac{1}{2}-\tau)\,\mathbf{1}^\top r`
is assembled from :py:func:`admm.norm` and :py:func:`admm.sum`.