.. include:: ../definition.hrst

.. _example-linear-program:

Linear Program
==============

A linear program is the simplest convex template: the objective is linear, and every
constraint is affine.

The standard form is

.. math::

   \begin{array}{ll}
   \min\limits_x & c^\top x \\
   \text{s.t.} & A x \le b, \\
               & x \ge 0.
   \end{array}

Here :math:`x \in \mathbb{R}^n` is the decision vector, :math:`c \in \mathbb{R}^n`
contains the linear costs, :math:`A \in \mathbb{R}^{m \times n}` and
:math:`b \in \mathbb{R}^m` define the inequality system, and :math:`x \ge 0`
keeps every component of the decision vector nonnegative.

This form is the basic model for allocation and planning problems. The objective
chooses a direction in which we want to move, and the inequalities carve out the
feasible polyhedron in which the solution must live.

**Step 1: Generate data from a feasible construction**

We build this random instance from a point that is already feasible. First we draw
a nonnegative vector :math:`x_0`. Then we set
:math:`b = A x_0 + s_0` with :math:`s_0 \ge 0`, so :math:`x_0` automatically
satisfies :math:`A x \le b`. This is a simple way to create synthetic data without
accidentally generating an inconsistent constraint system.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 15
    n = 10
    s0 = np.random.randn(m)
    lamb0 = np.maximum(-s0, 0)
    s0 = np.maximum(s0, 0)
    x0 = np.maximum(np.random.randn(n), 0)
    A = np.random.randn(m, n)
    b = A @ x0 + s0
    c = -A.T @ lamb0

The vector :math:`\lambda_0 \ge 0` is only used to define the cost vector
:math:`c = -A^\top \lambda_0` for this example.

**Step 2: Create the model and variable**

Now we create a model and one length-:math:`n` decision variable. In the symbolic
API, ``admm.Var("x", n)`` means "create a vector variable named ``x`` with
``n`` entries."

.. code-block:: python

    model = admm.Model()
    x = admm.Var("x", n)

**Step 3: Write the objective**

The mathematical objective is :math:`c^\top x`. In code, that same linear form is
written as ``c.T @ x``.

.. code-block:: python

    model.setObjective(c.T @ x)

**Step 4: Add the constraints**

The inequality system :math:`A x \le b` becomes one vectorized constraint, and the
nonnegativity condition :math:`x \ge 0` becomes a second constraint. These two lines
mirror the two rows of the display-math model above.

.. code-block:: python

    model.addConstr(A @ x <= b)
    model.addConstr(x >= 0)

**Step 5: Solve and inspect the result**

After ``model.optimize()``, the solver fills in the objective value and the solver
status string.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -7.629241267164004
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 15
    n = 10
    s0 = np.random.randn(m)
    lamb0 = np.maximum(-s0, 0)
    s0 = np.maximum(s0, 0)
    x0 = np.maximum(np.random.randn(n), 0)
    A = np.random.randn(m, n)
    b = A @ x0 + s0
    c = -A.T @ lamb0

    model = admm.Model()
    x = admm.Var("x", n)
    model.setObjective(c.T @ x)
    model.addConstr(A @ x <= b)
    model.addConstr(x >= 0)
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -7.629241267164004
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/linear_program.py


The solution is a vertex of the polyhedron :math:`\{x \ge 0 : Ax \le b\}` that minimizes
:math:`c^\top x`. The ADMM formulation maps directly: one ``setObjective``, one
``addConstr`` per constraint family.