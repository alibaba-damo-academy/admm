.. include:: ../definition.hrst

.. _example-portfolio-optimization:

Portfolio Optimization
======================

Portfolio optimization asks how to divide capital across assets.

In the mean-variance model, each asset has an expected return, the assets share
a covariance matrix that measures joint risk, and a risk-aversion parameter
controls how much volatility we are willing to accept in exchange for return.

.. math::

   \begin{array}{ll}
   \min\limits_w & -\mu^\top w + \gamma w^\top \Sigma w \\
   \text{s.t.} & \mathbf{1}^\top w = 1, \\
               & w \ge 0.
   \end{array}

Here :math:`w` contains the portfolio weights, :math:`\mu` is the vector of
expected returns, :math:`\Sigma` is the covariance matrix of asset returns, and
:math:`\gamma > 0` is the risk-aversion parameter.

The first term, :math:`-\mu^\top w`, rewards portfolios with high expected
return. The second term, :math:`\gamma w^\top \Sigma w`, penalizes risky
portfolios. The constraints say the whole budget must be invested and that
short-selling is not allowed.

**Step 1: Generate expected returns, a covariance matrix, and a risk parameter**

The vector ``mu`` plays the role of expected returns. To build a positive
semidefinite covariance matrix, we first draw a random factor matrix ``F`` and
then form ``Sigma = F.T @ F + 0.1 * np.eye(n)``. The small diagonal shift makes
the matrix safely positive definite.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 50
    mu = np.abs(np.random.randn(n))
    F = np.random.randn(n + 5, n)
    Sigma = F.T @ F + 0.1 * np.eye(n)
    gamma = 0.5

**Step 2: Create the model and the portfolio-weight variable**

The variable ``w`` has one entry per asset. Its entries will become the
portfolio weights.

.. code-block:: python

    model = admm.Model()
    w = admm.Var("w", n)

**Step 3: Write the return term and the risk term**

The expression ``mu.T @ w`` is the portfolio's expected return, while
``w.T @ Sigma @ w`` is its quadratic risk. Since the API uses minimization, we
subtract expected return and add the risk penalty weighted by ``gamma``.

.. code-block:: python

    expected_return = mu.T @ w
    risk = w.T @ Sigma @ w
    model.setObjective(-expected_return + gamma * risk)

**Step 4: Add the budget and long-only constraints**

The equality constraint makes the portfolio fully invested, and the inequality
constraint prevents negative weights.

.. code-block:: python

    model.addConstr(admm.sum(w) == 1)
    model.addConstr(w >= 0)

**Step 5: Solve and inspect the result**

After solving, ``model.ObjVal`` gives the optimal value of the
return-versus-risk tradeoff, and ``model.StatusString`` reports solver success.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -0.9808918614054916
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 50
    mu = np.abs(np.random.randn(n))
    F = np.random.randn(n + 5, n)
    Sigma = F.T @ F + 0.1 * np.eye(n)
    gamma = 0.5

    model = admm.Model()
    w = admm.Var("w", n)
    expected_return = mu.T @ w
    risk = w.T @ Sigma @ w
    model.setObjective(-expected_return + gamma * risk)
    model.addConstr(admm.sum(w) == 1)
    model.addConstr(w >= 0)
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -0.9808918614054916
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/portfolio_optimization.py


The solution :math:`w` minimizes :math:`-\mu^\top w + \gamma\, w^\top \Sigma w` over the
simplex :math:`\{w \ge 0,\; \mathbf{1}^\top w = 1\}` — the classic mean-variance
trade-off, written in three lines of ADMM code.