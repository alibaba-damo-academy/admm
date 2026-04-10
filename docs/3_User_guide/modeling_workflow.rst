.. include:: ../definition.hrst

.. _user-guide-modeling-workflow:

Modeling Workflow
=================

Most formulations follow the same sequence from blank model to interpreted result.
When you are learning the API, this is the safest default walkthrough to follow
before tuning details.

.. list-table:: Standard workflow
   :widths: 7 43 50
   :header-rows: 1
   :class: longtable

   * - Step
     - Action
     - Main API entry points
   * - 1
     - create decision variables and data placeholders
     - :py:class:`Var`, :py:class:`Param`
   * - 2
     - build expressions from operators and atoms
     - operators, global functions
   * - 3
     - set one scalar objective
     - :py:meth:`Model.setObjective`
   * - 4
     - add feasibility conditions
     - :py:meth:`Model.addConstr`
   * - 5
     - tune options if needed
     - :py:meth:`Model.setOption`
   * - 6
     - solve the formulation
     - :py:meth:`Model.optimize`
   * - 7
     - inspect results
     - :py:attr:`Model.ObjVal`, :py:attr:`Model.StatusString`, :py:attr:`Var.X`

Three practical habits make this workflow smoother:

- keep the objective scalar and delay option tuning until the model structure is already correct
- use variable attributes for intrinsic structure, and explicit constraints for relationships between expressions
- check solver status before interpreting the objective value or the primal solution

.. rst-class:: landing-page-section-heading

.. rubric:: Step 1: Declare variables and parameters

Start by naming the unknowns the solver should choose and the data you may want to bind
later. Use :py:class:`Var` for decision variables and :py:class:`Param` for values that
can change between solves without changing the symbolic model.

.. code-block:: python

    x = admm.Var("x", n)
    alpha = admm.Param("alpha")

.. rst-class:: landing-page-section-heading

.. rubric:: Step 2: Build expressions from operators and atoms

Next, turn the mathematics into symbolic expressions. It often helps to name the
important pieces before combining them into one final objective.

.. code-block:: python

    residual = A @ x - b
    loss = 0.5 * admm.sum(admm.square(residual))
    regularizer = alpha * admm.norm(x, ord=1)

.. rst-class:: landing-page-section-heading

.. rubric:: Step 3: Set one scalar objective

The argument to :py:meth:`Model.setObjective` must be one final scalar expression.
If the math reads as "fit plus penalty", write that same combination in code.

.. code-block:: python

    model.setObjective(loss + regularizer)

.. rst-class:: landing-page-section-heading

.. rubric:: Step 4: Add feasibility conditions

After the objective is in place, add the relationships that must hold in every feasible
solution. In this example we require the coefficient vector to stay nonnegative.

.. code-block:: python

    model.addConstr(x >= 0)

.. rst-class:: landing-page-section-heading

.. rubric:: Step 5: Tune options only after the model structure is correct

Most beginner issues come from model structure, not from low-level solver settings.
Set options once the objective and constraints already match the intended math.

.. code-block:: python

    model.setOption(admm.Options.admm_max_iteration, 5000)

.. rst-class:: landing-page-section-heading

.. rubric:: Step 6: Solve with concrete parameter data

If the model contains parameters, bind them when calling :py:meth:`Model.optimize`.
That lets you keep the symbolic model and swap only the numeric values.

.. code-block:: python

    model.optimize({"alpha": 0.1})

.. rst-class:: landing-page-section-heading

.. rubric:: Step 7: Read the results in order

After the solve, check the status first, then the objective value, and then the variable
values. That habit prevents you from over-interpreting a failed or limited solve.

.. rst-class:: landing-page-section-heading

.. rubric:: Complete Example

The following example runs through the full workflow on a small nonnegative Lasso-style
problem. It is a good chapter-level reference because it includes variables, a parameter,
constraints, solver options, and post-solve inspection in one place.
Depending on the backend and verbosity settings, solver logs may appear before the
labeled prints below.

Mathematically, the model is

.. math::

   \begin{aligned}
   \min_x \quad & \frac{1}{2}\|Ax - b\|_2^2 + \alpha \|x\|_1 \\
   \text{s.t.} \quad & x \ge 0
   \end{aligned}

where:

- :math:`x` is the coefficient vector we want the solver to choose.
- :math:`A` and :math:`b` are the data matrix and observation vector.
- :math:`\alpha` is the regularization weight that we bind at solve time through the
  parameter ``alpha`` in the code.

.. code-block:: python

    import admm
    import numpy as np

    np.random.seed(1)
    m = 30
    n = 10
    A = np.random.randn(m, n)  # Data matrix
    b = np.random.randn(m)     # Observation vector

    model = admm.Model()  # Create the model
    x = admm.Var("x", n)  # Optimization variable
    alpha = admm.Param("alpha")  # Regularization parameter set before optimization

    model.setObjective(
        0.5 * admm.sum(admm.square(A @ x - b))
        + alpha * admm.norm(x, ord=1)
    )
    model.addConstr(x >= 0)  # Structural feasibility
    model.setOption(admm.Options.admm_max_iteration, 5000)  # Iteration budget
    model.optimize({"alpha": 0.1})  # Bind parameter data and solve

    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * model.ObjVal: ", round(model.ObjVal, 6))   # Expected: finite scalar objective
    print(" * x.X: ", np.round(np.asarray(x.X), 6))      # Expected: nearly nonnegative solution vector


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/modeling_workflow.py


The objective :math:`\tfrac{1}{2}\|Ax-b\|_2^2 + \alpha\|x\|_1` is written directly in
ADMM — no manual reformulation needed. The :math:`\ell_1` penalty promotes sparsity, while
the constraint :math:`x \ge 0` is enforced as a hard projection. Tiny negative entries in
the printed vector are solver-tolerance artifacts, not constraint violations.

The seven-step sequence (variable, parameter, objective, constraint, options, solve,
inspect) maps one-to-one from the math to the code and scales to vector, matrix, and
structured models.

Each step is expanded in :ref:`Variables <user-guide-variables>`,
:ref:`Parameters <user-guide-parameters>`, :ref:`Objective <user-guide-objective>`,
:ref:`Constraints <user-guide-constraints>`, :ref:`Solver Options <user-guide-solver-options>`,
and :ref:`Solve the Model <user-guide-solve-the-model>`.