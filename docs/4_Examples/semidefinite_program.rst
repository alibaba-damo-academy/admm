.. include:: ../definition.hrst

.. _example-semidefinite-program:

Semidefinite Program
====================

Semidefinite programming extends linear optimization from vectors to symmetric matrix
variables.

A typical form is

.. math::

   \begin{array}{ll}
   \min\limits_X & \mathrm{tr}(CX) \\
   \text{s.t.} & \mathrm{tr}(A_i X) = b_i,\quad i=1,\ldots,p, \\
               & X \succeq 0.
   \end{array}

Here :math:`X \in \mathbb{S}^n` is a symmetric matrix decision variable,
:math:`C \in \mathbb{S}^n` and :math:`A_i \in \mathbb{S}^n` are symmetric data
matrices, and :math:`X \succeq 0` means that :math:`X` must be positive
semidefinite. In other words, :math:`X` must lie in the PSD cone rather than merely
satisfy entrywise inequalities.

The trace operator :math:`\mathrm{tr}(CX)` plays the same role here that a dot
product plays in vector optimization: it turns two matrices into one scalar.

**Step 1: Generate symmetric matrix data**

We again fix the random seed so the same instance is reproduced every run. The
matrix ``C`` is built as ``R.T @ R``, which guarantees that ``C`` is symmetric and
positive semidefinite. Each ``A[i]`` is explicitly symmetrized with
``0.5 * (Ai + Ai.T)`` so it matches the symmetric matrix variable.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 4
    p = 3
    R = np.random.randn(n, n)
    C = R.T @ R                  # PSD: tr(C X) >= 0 for all X >> 0
    A = []
    b = []
    for _ in range(p):
        Ai = np.random.randn(n, n)
        Ai = 0.5 * (Ai + Ai.T)
        A.append(Ai)
        b.append(np.random.randn())
    A = np.array(A)
    b = np.array(b)

**Step 2: Create the model and PSD matrix variable**

This is the key modeling step for an SDP. The call
``admm.Var("X", n, n, PSD=True)`` creates an :math:`n \times n` matrix variable and
declares that it must be positive semidefinite. That one flag is what tells the
symbolic API to model :math:`X` as a PSD matrix variable.

.. code-block:: python

    model = admm.Model()
    X = admm.Var("X", n, n, PSD=True)

**Step 3: Write the trace objective**

The mathematical objective is :math:`\mathrm{tr}(CX)`. In code, we express that as
``admm.trace(C @ X)``. The matrix product ``C @ X`` happens first, and then
``admm.trace(...)`` extracts the scalar trace.

.. code-block:: python

    model.setObjective(admm.trace(C @ X))

**Step 4: Add the trace constraints one by one**

Each scalar condition :math:`\mathrm{tr}(A_i X) = b_i` becomes one call to
``addConstr(...)``. We use a loop because the model has one such equality for each
index :math:`i = 1, \ldots, p`.

.. code-block:: python

    for i in range(p):
        model.addConstr(admm.trace(A[i] @ X) == b[i])

**Step 5: Solve and inspect the result**

After solving, ``model.ObjVal`` reports the optimal trace objective and
``model.StatusString`` reports the solver status. If you want to inspect the actual
matrix solution, it is available afterward in ``X.X``.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 0.4295347451953324
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 4
    p = 3
    R = np.random.randn(n, n)
    C = R.T @ R                  # PSD: tr(C X) >= 0 for all X >> 0
    A = []
    b = []
    for _ in range(p):
        Ai = np.random.randn(n, n)
        Ai = 0.5 * (Ai + Ai.T)
        A.append(Ai)
        b.append(np.random.randn())
    A = np.array(A)
    b = np.array(b)

    model = admm.Model()
    X = admm.Var("X", n, n, PSD=True)
    model.setObjective(admm.trace(C @ X))
    for i in range(p):
        model.addConstr(admm.trace(A[i] @ X) == b[i])
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 0.4295347451953324
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/semidefinite_program.py


The decision variable is a PSD matrix declared with ``PSD=True``. The trace equalities
:math:`\operatorname{tr}(A_i X) = b_i` are linear in :math:`X`, so the whole SDP is
written directly — no vectorization or LMI reformulation required.