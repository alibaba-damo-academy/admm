.. include:: ../definition.hrst

.. _example-svm-with-l1-regularization:

SVM with L1 Regularization
==========================

An :math:`\ell_1`-regularized support vector machine is a large-margin classifier that
also encourages sparsity in the fitted coefficients.

The optimization model is

.. math::

   \begin{array}{ll}
   \min\limits_{\beta,v} & \dfrac{1}{m}\sum_{i=1}^{m}
   \max\!\left(0, 1 - y_i(x_i^\top \beta - v)\right) + \lambda \|\beta\|_1.
   \end{array}

Here :math:`x_i \in \mathbb{R}^n` is the feature vector for observation :math:`i`,
:math:`y_i \in \{-1,1\}` is its class label, :math:`\beta \in \mathbb{R}^n` is the
coefficient vector, and :math:`v` is the intercept.

This objective again has two pieces:

- The hinge loss :math:`\max(0, 1 - y_i(x_i^\top \beta - v))` enforces a margin-based
  classification rule.
- The :math:`\ell_1` term :math:`\lambda \|\beta\|_1` promotes a sparse classifier.

**Step 1: Generate a sparse linear classification problem**

We first create a hidden sparse vector ``beta_true`` and then use it to generate labels.
The line ``beta_true[10:] = 0`` makes the last 15 entries zero, so the synthetic problem
really does have only a subset of informative features.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 120
    n = 25
    beta_true = np.random.randn(n)
    beta_true[10:] = 0
    X = np.random.randn(m, n)
    y = np.sign(X @ beta_true + 0.5 * np.random.randn(m))
    y[y == 0] = 1
    lam = 0.1

**Step 2: Create the model, increase the iteration budget, and define variables**

The fitted classifier uses a coefficient vector ``beta`` and an intercept ``v``. We also
set ``admm_max_iteration`` to ``10000``. That larger iteration budget is part of the
reference example so this bundled dataset reaches the usual successful termination path
instead of making the default iteration cap the main outcome.

.. code-block:: python

    model = admm.Model()
    model.setOption(admm.Options.admm_max_iteration, 10000)
    beta = admm.Var("beta", n)
    v = admm.Var("v")

**Step 3: Write the hinge-loss term and the L1 penalty**

The signed margin in this example is ``y * (X @ beta - v)``. If that signed margin is at
least ``1``, the hinge loss is zero. Otherwise the expression pays a linear penalty for
being inside the margin or on the wrong side of the separating hyperplane.

.. code-block:: python

    margin_loss = admm.sum(admm.maximum(1 - y * (X @ beta - v), 0))
    model.setObjective(margin_loss / m + lam * admm.norm(beta, ord=1))

The first term averages the hinge losses over the training set. The second term is the
sparsity regularizer that can remove weak features by driving some coefficients to zero.

**Step 4: Add constraints**

This SVM example has no explicit constraints, so there are no ``model.addConstr(...)``
calls. The entire model is encoded through the objective.

**Step 5: Solve and inspect the result**

Now we solve the model and print the standard solver outputs.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 0.5810343957323443
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 120
    n = 25
    beta_true = np.random.randn(n)
    beta_true[10:] = 0
    X = np.random.randn(m, n)
    y = np.sign(X @ beta_true + 0.5 * np.random.randn(m))
    y[y == 0] = 1
    lam = 0.1

    model = admm.Model()
    model.setOption(admm.Options.admm_max_iteration, 10000)
    beta = admm.Var("beta", n)
    v = admm.Var("v")
    margin_loss = admm.sum(admm.maximum(1 - y * (X @ beta - v), 0))
    model.setObjective(margin_loss / m + lam * admm.norm(beta, ord=1))
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 0.5810343957323443
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/svm_with_l1.py


The hinge loss penalizes margin violations directly, and the :math:`\ell_1` term performs
feature selection by zeroing weak coefficients — the same sparsity mechanism as in
Lasso, applied here to a classification model.