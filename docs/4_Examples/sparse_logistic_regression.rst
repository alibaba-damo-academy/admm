.. include:: ../definition.hrst

.. _example-sparse-logistic-regression:

Sparse Logistic Regression
==========================

Sparse logistic regression is a standard convex model for binary classification when we
also want automatic feature selection.

The optimization model is

.. math::

   \begin{array}{ll}
   \min\limits_{w,v} & \dfrac{1}{m}\sum_{i=1}^{m} \log\!\left(1 + \exp(-y_i(x_i^\top w + v))\right)
   + \lambda \|w\|_1.
   \end{array}

Here :math:`x_i \in \mathbb{R}^n` is the feature vector for observation :math:`i`,
:math:`y_i \in \{-1,1\}` is its class label, :math:`w \in \mathbb{R}^n` is the fitted
coefficient vector, and :math:`v` is the intercept.

This objective has two pieces:

- The logistic loss penalizes wrong or uncertain classifications smoothly.
- The :math:`\ell_1` term :math:`\lambda \|w\|_1` encourages many entries of :math:`w`
  to become exactly zero.

That second effect is why sparse logistic regression performs feature selection: if a
coefficient is driven to zero, that feature drops out of the classifier.

**Step 1: Generate labeled feature data**

We build a random feature matrix ``X`` and a hidden coefficient vector ``beta`` that is
used only to synthesize labels. The response vector ``y`` contains class labels in
``{-1, 1}``, which is the convention used by the margin expression in the loss.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 20
    m = 60
    beta = np.random.randn(n)
    X = np.random.randn(m, n)
    y = np.sign(X @ beta + 0.5 * np.random.randn(m))
    y[y == 0] = 1
    lam = 0.1

**Step 2: Create the model and decision variables**

The fitted classifier has a weight vector ``w`` and an intercept ``v``.

.. code-block:: python

    model = admm.Model()
    w = admm.Var("w", n)
    v = admm.Var("v")

**Step 3: Build the margin, logistic loss, and sparsity term**

The raw linear score is ``X @ w + v``. Multiplying by ``y`` turns that into a signed
margin: correctly classified points with comfortable separation have large positive signed
margins. The code stores the negative signed margin,
``margin = -y * (X @ w + v)``, because the loss formula contains
:math:`-y_i(x_i^\top w + v)` inside ``log(1 + exp(...))``.

.. code-block:: python

    margin = -y * (X @ w + v)
    logistic_loss = admm.sum(admm.logistic(margin, 1)) / m
    sparsity_penalty = lam * admm.norm(w, ord=1)
    model.setObjective(logistic_loss + sparsity_penalty)

``admm.logistic(margin, 1)`` encodes the smooth logistic loss term, while
``lam * admm.norm(w, ord=1)`` is the :math:`\ell_1` regularizer that can set some
coefficients exactly to zero.

**Step 4: Add constraints**

This sparse logistic regression example has no explicit constraints, so there are no
``model.addConstr(...)`` calls. The classification behavior is determined entirely by the
objective above.

**Step 5: Solve and inspect the result**

After optimizing, we print the standard solver outputs.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 0.6458330573436739
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 20
    m = 60
    beta = np.random.randn(n)
    X = np.random.randn(m, n)
    y = np.sign(X @ beta + 0.5 * np.random.randn(m))
    y[y == 0] = 1
    lam = 0.1

    model = admm.Model()
    w = admm.Var("w", n)
    v = admm.Var("v")
    margin = -y * (X @ w + v)
    model.setObjective(admm.sum(admm.logistic(margin, 1)) / m + lam * admm.norm(w, ord=1))
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 0.6458330573436739
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/sparse_logistic_regression.py


The :math:`\ell_1` penalty drives weak coefficients in :math:`w` to exactly zero, performing
automatic feature selection. Coefficients that survive are the features most relevant for
classification under the logistic loss.