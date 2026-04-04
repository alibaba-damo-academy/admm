.. include:: ../definition.hrst

.. _example-robust-pca:

Robust PCA
==========

Robust PCA starts from a simple decomposition idea:

.. math::

   M = L + S.

Here :math:`M` is the observed matrix, :math:`L` is the structured low-rank part we want
to recover, and :math:`S` is a correction matrix that is encouraged to be sparse. The
optimization model is

.. math::

   \begin{array}{ll}
   \min\limits_{L,S} & \|L\|_* + \lambda \|\mathrm{vec}(S)\|_1 \\
   \text{s.t.} & L + S = M.
   \end{array}

The three main pieces play different roles:

- :math:`\|L\|_*` is the nuclear norm, so it promotes a low-rank matrix :math:`L`
- :math:`\|\mathrm{vec}(S)\|_1` is the sum of absolute values of all entries of :math:`S`,
  so it promotes sparse corruption or sparse corrections
- :math:`L + S = M` couples the two matrices so they must exactly reconstruct the observed
  matrix

This is a good example of a structured matrix model: one variable is encouraged to have a
spectral pattern, the other is encouraged to have an entrywise sparsity pattern, and a
linear matrix equation ties them together.

**Step 1: Generate an observed matrix with dominant low-rank structure**

We first build a matrix from a rank-:math:`r` factorization and then add a small random
perturbation. The optimization model will try to explain the observed matrix ``M`` as the
sum of a low-rank matrix ``L`` and a correction matrix ``S``.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 50
    r = 10
    n = 40
    M = np.random.randn(m, r) @ np.random.randn(r, n)
    M = M + 0.1 * np.random.randn(m, n)
    lam = 1.0 / np.sqrt(max(m, n))

**Step 2: Create the model and the two matrix variables**

The model has two unknown matrices, both with the same shape as ``M``:

- ``L`` will represent the low-rank part
- ``S`` will represent the sparse correction part

.. code-block:: python

    model = admm.Model()
    L = admm.Var("L", m, n)
    S = admm.Var("S", m, n)

**Step 3: Write the objective term by term**

The low-rank promotion term is written as ``admm.norm(L, ord="nuc")``. This is the
code form of the nuclear norm :math:`\|L\|_*`, which is the standard convex surrogate
for rank.

The sparse-correction term is written as ``admm.sum(admm.abs(S))``. That sums the
absolute values of every entry of ``S``, which is exactly the matrix :math:`\ell_1`
penalty. Multiplying by ``lam`` controls how strongly sparsity is encouraged.

.. code-block:: python

    model.setObjective(admm.norm(L, ord="nuc") + lam * admm.sum(admm.abs(S)))

**Step 4: Add the affine coupling constraint**

The reconstruction rule :math:`M = L + S` becomes one linear matrix equality:

.. code-block:: python

    model.addConstr(L + S == M)

There are no other explicit constraints in this example. The structure comes from the
objective and from this single coupling equation.

**Step 5: Solve and inspect the result**

After solving, ``model.ObjVal`` reports the optimal value of the low-rank-plus-sparse
objective, and ``model.StatusString`` reports whether the solver succeeded. If you want
to inspect the recovered matrices themselves, they are available afterward in ``L.X`` and
``S.X``.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 422.49613807195703
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 50
    r = 10
    n = 40
    M = np.random.randn(m, r) @ np.random.randn(r, n)
    M = M + 0.1 * np.random.randn(m, n)
    lam = 1.0 / np.sqrt(max(m, n))

    model = admm.Model()
    L = admm.Var("L", m, n)
    S = admm.Var("S", m, n)
    model.setObjective(admm.norm(L, ord="nuc") + lam * admm.sum(admm.abs(S)))
    model.addConstr(L + S == M)
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 422.49613807195703
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/robust_pca.py


Robust PCA decomposes :math:`M = L + S` where :math:`\|L\|_*` promotes low rank and
:math:`\|S\|_1` promotes entrywise sparsity. Each variable gets its own regularizer —
ADMM splits this naturally into two proximal subproblems.