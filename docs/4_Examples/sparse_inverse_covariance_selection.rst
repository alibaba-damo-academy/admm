.. include:: ../definition.hrst

.. _example-sparse-inverse-covariance-selection:

Sparse Inverse Covariance Selection
===================================

Sparse inverse covariance selection estimates a *precision matrix*, meaning the inverse
covariance matrix of a Gaussian model.

If :math:`S` is the sample covariance matrix computed from data, then we estimate a
matrix :math:`\Theta` by solving

.. math::

   \min\limits_{\Theta \succ 0} -\log \det(\Theta) + \mathrm{tr}(S\Theta) + \lambda \|\mathrm{vec}(\Theta)\|_1.

Each term has an important modeling role:

- :math:`S` summarizes the observed covariance structure in the samples
- :math:`\Theta` is the unknown precision matrix we want to estimate
- :math:`-\log \det(\Theta)` keeps :math:`\Theta` in the positive-definite region and is
  part of the Gaussian log-likelihood
- :math:`\mathrm{tr}(S\Theta)` measures how the candidate precision matrix interacts with
  the empirical covariance
- :math:`\|\mathrm{vec}(\Theta)\|_1` penalizes the absolute values of the entries of
  :math:`\Theta`, encouraging many of them to become small or zero

In statistics, sparsity in :math:`\Theta` is useful because zeros in a precision matrix
are related to conditional-independence structure.

**Step 1: Generate samples and compute the sample covariance**

We first create a synthetic positive-definite precision matrix ``true_precision``. After
inverting it, we use the resulting covariance matrix to generate Gaussian samples. The
matrix ``S`` is then the sample covariance estimated from those observations.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 30
    sample_num = 60
    A = np.random.randn(n, n)
    true_precision = A.T @ A + 0.5 * np.eye(n)
    samples = np.random.multivariate_normal(
        mean=np.zeros(n),
        cov=np.linalg.inv(true_precision),
        size=sample_num,
    )
    S = np.cov(samples, rowvar=False)
    lam = 0.05

**Step 2: Create the model and the precision-matrix variable**

The decision variable is ``Theta``, an :math:`n \times n` matrix. We declare it with
``PSD=True`` because a precision matrix must be symmetric and live in the positive
semidefinite cone. In combination with the log-determinant term, the optimizer is pushed
toward a positive-definite solution.

.. code-block:: python

    model = admm.Model()
    Theta = admm.Var("Theta", n, n, PSD=True)

**Step 3: Write the likelihood and sparsity terms**

The full objective is written directly in code:

.. code-block:: python

    model.setObjective(-admm.log_det(Theta) + admm.trace(S @ Theta) + lam * admm.sum(admm.abs(Theta)))

This one line mirrors the math almost term for term:

- ``-admm.log_det(Theta)`` is :math:`-\log \det(\Theta)`
- ``admm.trace(S @ Theta)`` is :math:`\mathrm{tr}(S\Theta)`
- ``lam * admm.sum(admm.abs(Theta))`` is the entrywise sparsity penalty

**Step 4: Add constraints**

This example has no separate ``model.addConstr(...)`` calls. The matrix-domain structure
is already encoded in the variable declaration ``PSD=True``, and the log-determinant term
handles the interior positive-definite behavior required by the likelihood model.

**Step 5: Solve and inspect the result**

After optimization, the standard outputs tell us whether the solve succeeded and what the
final objective value was. The estimated precision matrix itself is available in
``Theta.X``.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -15.134257007715702
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 30
    sample_num = 60
    A = np.random.randn(n, n)
    true_precision = A.T @ A + 0.5 * np.eye(n)
    samples = np.random.multivariate_normal(
        mean=np.zeros(n),
        cov=np.linalg.inv(true_precision),
        size=sample_num,
    )
    S = np.cov(samples, rowvar=False)
    lam = 0.05

    model = admm.Model()
    Theta = admm.Var("Theta", n, n, PSD=True)
    model.setObjective(-admm.log_det(Theta) + admm.trace(S @ Theta) + lam * admm.sum(admm.abs(Theta)))
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -15.134257007715702
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/sparse_inverse_covariance.py


The solution ``Theta`` maximizes :math:`\log\det\Theta - \operatorname{tr}(S\Theta)` subject
to an :math:`\ell_1` sparsity penalty. Zero entries in ``Theta`` indicate conditionally
independent variable pairs — the sparse structure reveals the underlying graphical model.