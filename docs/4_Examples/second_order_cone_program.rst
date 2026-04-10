.. include:: ../definition.hrst

.. _example-second-order-cone-program:

Second-Order Cone Program
=========================

Second-order cone programming is the standard convex formulation for Euclidean norm
constraints of the form

.. math::

   \|A_i x + b_i\|_2 \le c_i^\top x + d_i.

Together with affine equalities, this gives the standard SOCP template

.. math::

   \begin{array}{ll}
   \min\limits_x & f^\top x \\
   \text{s.t.} & \|A_i x + b_i\|_2 \le c_i^\top x + d_i,\quad i=1,\ldots,m, \\
               & Fx = g.
   \end{array}

Here :math:`x \in \mathbb{R}^n` is the decision variable,
:math:`f \in \mathbb{R}^n` defines the linear objective :math:`f^\top x`,
:math:`A_i \in \mathbb{R}^{n_i \times n}`, :math:`b_i \in \mathbb{R}^{n_i}`,
:math:`c_i \in \mathbb{R}^n`, :math:`d_i \in \mathbb{R}`, and
:math:`F \in \mathbb{R}^{p \times n}`, :math:`g \in \mathbb{R}^p` define the affine
equality system :math:`F x = g`. The left-hand side of each cone constraint is a
Euclidean norm, while the right-hand side is affine.

SOCPs appear whenever "stay inside a norm cone" is the natural modeling idea. They
are common in robust optimization, control, and geometric design problems.

**Step 1: Generate a feasible cone instance**

This example is built from a known feasible point :math:`x_0`. For each cone
constraint, we choose
:math:`d_i = \|A_i x_0 + b_i\|_2 - c_i^\top x_0`, so the inequality is satisfied at
:math:`x_0` with equality. We also set ``g = F @ x0`` so the affine equalities hold
at the same reference point.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 3
    n = 10
    p = 5
    n_i = 5

    f = np.random.randn(n)
    A = []
    b = []
    c = []
    d = []
    x0 = np.random.randn(n)
    for i in range(m):
        A.append(np.random.randn(n_i, n))
        b.append(np.random.randn(n_i))
        c.append(np.random.randn(n))
        d.append(np.linalg.norm(A[i] @ x0 + b[i], 2) - c[i].T @ x0)
    F = np.random.randn(p, n)
    g = F @ x0

**Step 2: Create the model and variable**

This problem has one vector decision variable :math:`x \in \mathbb{R}^{10}`.

.. code-block:: python

    model = admm.Model()
    x = admm.Var("x", n)

**Step 3: Write the objective**

The objective is the linear form :math:`f^\top x`, which maps directly to
``f.T @ x`` in the symbolic API.

.. code-block:: python

    model.setObjective(f.T @ x)

**Step 4: Add the cone constraints one by one**

This is the central modeling idea in an SOCP. For each index ``i``, the code

``admm.norm(A[i] @ x + b[i], ord=2) <= c[i].T @ x + d[i]``

is a literal translation of
:math:`\|A_i x + b_i\|_2 \le c_i^\top x + d_i`. We add those constraints in a loop
because the problem has several cone inequalities, not just one. After that, we add
the affine equality block :math:`Fx = g`.

.. code-block:: python

    for i in range(m):
        model.addConstr(admm.norm(A[i] @ x + b[i], ord=2) <= c[i].T @ x + d[i])
    model.addConstr(F @ x == g)

**Step 5: Solve and inspect the result**

Once the cone and equality constraints are in place, we solve and print the standard
solver outputs.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 2.06815161777782
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 3
    n = 10
    p = 5
    n_i = 5

    f = np.random.randn(n)
    A = []
    b = []
    c = []
    d = []
    x0 = np.random.randn(n)
    for i in range(m):
        A.append(np.random.randn(n_i, n))
        b.append(np.random.randn(n_i))
        c.append(np.random.randn(n))
        d.append(np.linalg.norm(A[i] @ x0 + b[i], 2) - c[i].T @ x0)
    F = np.random.randn(p, n)
    g = F @ x0

    model = admm.Model()
    x = admm.Var("x", n)
    model.setObjective(f.T @ x)
    for i in range(m):
        model.addConstr(admm.norm(A[i] @ x + b[i], ord=2) <= c[i].T @ x + d[i])
    model.addConstr(F @ x == g)
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 2.06815161777782
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/second_order_cone_program.py


Each second-order cone constraint :math:`\|A_i x + b_i\|_2 \le c_i^\top x + d_i` maps to
a single ``addConstr(...)`` call. No auxiliary variables or epigraph reformulations are
needed — ADMM accepts norm constraints directly.