.. include:: ../definition.hrst

.. _example-quadratic-program:

Quadratic Program
=================

Quadratic programming adds curvature to the objective while keeping the constraints
affine.

The standard form is

.. math::

   \begin{array}{ll}
   \min\limits_x & \frac{1}{2} x^\top P x + q^\top x \\
   \text{s.t.} & Gx \le h, \\
               & Ax = b.
   \end{array}

Here :math:`x \in \mathbb{R}^n` is the decision vector, :math:`P \in \mathbb{S}_+^n`
is a positive semidefinite matrix that defines the quadratic curvature,
:math:`q \in \mathbb{R}^n` is the linear term, :math:`Gx \le h` collects the
inequalities, and :math:`Ax = b` collects the affine equalities.

The matrix :math:`P` is what makes this problem different from a linear program. If
:math:`P` is positive semidefinite, the objective stays convex, so we still have a
convex optimization problem.

**Step 1: Generate a convex quadratic instance**

We use ``np.random.seed(1)`` so the same synthetic problem is produced every time.
The key construction is ``P = P.T @ P``. That guarantees :math:`P \succeq 0`, which
makes the quadratic term convex.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 15
    n = 10
    p = 5
    P = np.random.randn(n, n)
    P = P.T @ P
    q = np.random.randn(n)
    G = np.random.randn(m, n)
    h = G @ np.random.randn(n)
    A = np.random.randn(p, n)
    b = np.random.randn(p)

**Step 2: Create the model and variable**

This example needs one vector decision variable :math:`x \in \mathbb{R}^{10}`.

.. code-block:: python

    model = admm.Model()
    x = admm.Var("x", n)

**Step 3: Write the objective**

The symbolic API follows the math almost verbatim:

- ``0.5 * x.T @ P @ x`` is the quadratic term :math:`\tfrac{1}{2} x^\top P x`.
- ``q.T @ x`` is the linear term :math:`q^\top x`.

Putting those two pieces together gives the full objective.

.. code-block:: python

    model.setObjective(0.5 * x.T @ P @ x + q.T @ x)

**Step 4: Add the constraints**

The two constraint families also map directly into symbolic expressions:

- :math:`Gx \le h` becomes ``model.addConstr(G @ x <= h)``.
- :math:`Ax = b` becomes ``model.addConstr(A @ x == b)``.

.. code-block:: python

    model.addConstr(G @ x <= h)
    model.addConstr(A @ x == b)

**Step 5: Solve and inspect the result**

Once the model is complete, we solve it and print the standard solver outputs.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 86.89077551539528
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 15
    n = 10
    p = 5
    P = np.random.randn(n, n)
    P = P.T @ P
    q = np.random.randn(n)
    G = np.random.randn(m, n)
    h = G @ np.random.randn(n)
    A = np.random.randn(p, n)
    b = np.random.randn(p)

    model = admm.Model()
    x = admm.Var("x", n)
    model.setObjective(0.5 * x.T @ P @ x + q.T @ x)
    model.addConstr(G @ x <= h)
    model.addConstr(A @ x == b)
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 86.89077551539528
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/quadratic_program.py


The solution minimizes :math:`\tfrac{1}{2}x^\top P x + q^\top x` over the polyhedron
:math:`\{x : Gx \le h,\; Ax = b\}`. Notice that the ADMM code is a line-by-line
transcription of that formulation — no manual dualization or reformulation required.