.. include:: ../definition.hrst

.. _user-guide-constraints:

Constraints
===========

Constraints are added with :py:meth:`Model.addConstr`. In practice, models use a mix
of affine equalities and inequalities, norm-ball constraints, and cone constraints.
Use explicit constraints for relationships between expressions.

For intrinsic structure such as nonnegativity, symmetry, or PSD / NSD form, it is often
cleaner to declare the property directly on the variable as described in
:ref:`Variables <user-guide-variables>`.

How to Choose the Direct Form
-----------------------------

When you translate math into code, this teaching rule keeps models readable:

1. If a property belongs to one variable by design, prefer a variable attribute.
2. If it relates multiple expressions, add it with :py:meth:`Model.addConstr`.
3. Write the constraint in the same form you would say out loud.

For example, prefer ``admm.norm(x, ord=2) <= r`` over a more opaque algebraic rewrite.
The library can lower several equivalent forms, but the direct expression is usually
easier to read, review, and debug.

.. list-table:: Common constraint patterns
   :widths: 28 72
   :header-rows: 1
   :class: longtable

   * - Pattern
     - Example
   * - affine equality or inequality
     - ``model.addConstr(A @ x <= b)``
   * - aggregate linear constraint
     - ``model.addConstr(admm.sum(x) == 1)``
   * - elementwise bounds
     - ``model.addConstr(X >= 0)``
   * - L2 / SOC norm ball
     - ``model.addConstr(admm.norm(x, ord=2) <= s)``
   * - squared L2 norm ball
     - ``model.addConstr(admm.norm(x, ord=2) ** 2 <= r)``
   * - other norm balls
     - ``model.addConstr(admm.norm(X, ord='fro') <= r)``
   * - semidefinite cone
     - ``model.addConstr(X >> 0)``
   * - negative semidefinite cone
     - ``model.addConstr(X << 0)``

Walkthrough: Projecting Onto a Feasible Set
-------------------------------------------

The example below shows several common constraint families working together in one model.
The idea is simple: choose a vector ``x`` and a symmetric matrix ``X`` that stay as close
as possible to chosen targets, while also satisfying a collection of feasibility rules.

Read the model in three parts:

- the objective measures distance to ``x_target`` and ``X_target``
- the vector constraints keep ``x`` nonnegative and inside L1 and L2 norm balls
- the matrix constraints keep ``X`` inside nuclear-norm, Frobenius-norm, and PSD cones

Because the targets are not automatically feasible, solving the model effectively projects
them back onto the feasible set described by the constraints.

Complete runnable example:

The complete model can be written as

.. math::

   \begin{aligned}
   \min_{x, X} \quad & \|x - x_{\text{target}}\|_2^2
   + \|X - X_{\text{target}}\|_F^2 \\
   \text{s.t.} \quad & x \ge 0, \\
   & \|x\|_1 \le 1.2, \\
   & \|x\|_2 \le 1.0, \\
   & \|X\|_* \le 1.1, \\
   & \|X\|_F \le 1.0, \\
   & X \succeq 0
   \end{aligned}

where :math:`x` is the vector variable, :math:`X` is the symmetric matrix variable,
and :math:`x_{\text{target}}`, :math:`X_{\text{target}}` are the target values we would
like to match as closely as feasibility allows.

.. code-block:: python

    import admm
    import numpy as np

    x_target = np.array([2.0, 1.0])
    X_target = np.array([[2.0, 0.0], [0.0, 1.0]])

    model = admm.Model()
    x = admm.Var("x", 2)
    X = admm.Var("X", 2, 2, symmetric=True)

    model.setObjective(
        admm.sum(admm.square(x - x_target))
        + admm.sum(admm.square(X - X_target))
    )
    model.addConstr(x >= 0)
    model.addConstr(admm.norm(x, ord=1) <= 1.2)
    model.addConstr(admm.norm(x, ord=2) <= 1.0)
    model.addConstr(admm.norm(X, ord='nuc') <= 1.1)
    model.addConstr(admm.norm(X, ord='fro') <= 1.0)
    model.addConstr(X >> 0)

    model.optimize()
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * model.ObjVal: ", round(model.ObjVal, 6))   # Expected: around 3.462854
    print(" * x.X: ", np.round(np.asarray(x.X), 6))      # Expected: [0.974155 0.225845]
    print(" * X.X: ", np.round(np.asarray(X.X), 6))      # Expected: [[0.99441 0.     ] [0.      0.10559]]


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/constraints_projection.py


This is a projection problem: find the point nearest to
:math:`(x_{\text{target}}, X_{\text{target}})` inside the feasible set defined by all
constraints simultaneously. The solution generally differs from the targets because
multiple constraints are active at once.

In practice, write constraints in their natural form — ``admm.norm(x, ord=2) <= r``,
``PSD=True`` — so the code mirrors the math directly.

For the broader formulation boundary, see :ref:`Supported Problem Structure <user-guide-supported-problem-structure>`.
