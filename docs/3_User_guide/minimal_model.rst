.. include:: ../definition.hrst

.. _user-guide-minimal-model:

Minimal Model
=============

This page shows the smallest complete ADMM modeling workflow from start to finish.
The model is intentionally tiny, so you can focus on the pattern: define variables,
define parameters, build the model, solve it, and then read the solution.

.. math::

   \begin{array}{ll}
   \min\limits_{x_1, x_2} & x_1 + x_2 \\
   \text{s.t.} & x_1 \ge p, \\
               & x_2 \ge 0,
   \end{array}

Here :math:`x_1` and :math:`x_2` are scalar decision variables, and :math:`p`
is a scalar parameter that we do not bind until solve time.

**Step 1: Create the variables and parameter**

We begin by declaring the unknowns and the parameter. Since this example is scalar,
each object is created without a shape argument.

.. code-block:: python

    import admm

    x1 = admm.Var("x1")
    x2 = admm.Var("x2")
    p = admm.Param("p")

The two ``Var`` objects are the values the solver may change, while ``Param("p")``
represents solve-time data that we will supply later.

**Step 2: Create the model**

Now we create the optimization model that will hold the objective and constraints.

.. code-block:: python

    model = admm.Model()

**Step 3: Add the objective**

The mathematical objective is :math:`x_1 + x_2`, so in code we write the same
symbolic expression and pass it to :py:meth:`Model.setObjective`.

.. code-block:: python

    model.setObjective(x1 + x2)

Because the objective is linear, the solver will try to make both variables as
small as the constraints allow.

**Step 4: Add the constraints**

The first constraint, :math:`x_1 \ge p`, says that :math:`x_1` must stay above the
parameter value chosen at solve time. The second constraint, :math:`x_2 \ge 0`,
makes the second variable nonnegative.

.. code-block:: python

    model.addConstr(x1 >= p)
    model.addConstr(x2 >= 0)

These two lines mirror the two rows in the display-math model above.

**Step 5: Solve with a concrete parameter value**

Before solving, we set a simple iteration limit for this reference example. Then we
bind :math:`p = 2` by passing a dictionary into :py:meth:`Model.optimize`.

.. code-block:: python

    model.setOption(admm.Options.admm_max_iteration, 1000)
    model.optimize({"p": 2})

    print(" * model.ObjVal: ", model.ObjVal)              # Expected: 2.0
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * x1.X: ", x1.X)                              # Expected: 2.0
    print(" * x2.X: ", x2.X)                              # Expected: 0.0

The status string tells us whether the solve succeeded, the objective value reports
the final cost, and ``x1.X`` / ``x2.X`` expose the numerical solution values.

Complete runnable example:

.. code-block:: python

    import admm

    x1 = admm.Var("x1")
    x2 = admm.Var("x2")
    p = admm.Param("p")

    model = admm.Model()
    model.setObjective(x1 + x2)
    model.addConstr(x1 >= p)
    model.addConstr(x2 >= 0)
    model.setOption(admm.Options.admm_max_iteration, 1000)
    model.optimize({"p": 2})

    print(" * model.ObjVal: ", model.ObjVal)              # Expected: 2.0
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * x1.X: ", x1.X)                              # Expected: 2.0
    print(" * x2.X: ", x2.X)                              # Expected: 0.0


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/minimal_model.py


Why does the solution land here? The constraint :math:`x_1 \ge 2` forces the first
variable to be at least :math:`2`, and :math:`x_2 \ge 0` forces the second variable
to be at least :math:`0`. Since the objective tries to minimize their sum, the best
choice is to sit exactly on those lower bounds: :math:`x_1^\star = 2`,
:math:`x_2^\star = 0`.

The same pattern scales naturally from scalar models to vector and matrix models.
For the general sequence behind this example, see :ref:`Modeling Workflow <user-guide-modeling-workflow>`.
For more on solve-time data, see :ref:`Parameters <user-guide-parameters>`.