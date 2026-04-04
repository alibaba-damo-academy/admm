.. include:: ../definition.hrst

.. _user-guide-solve-the-model:


Solve the Model
===============

Once the model is built, solve it with :py:meth:`Model.optimize`, either with or without
parameter binding. The solve call is only half of the workflow, though. After
``optimize()`` returns, read the results in a fixed order so that you do not over-interpret
an unsuccessful run.

Read Results in This Order
--------------------------

The following outputs usually matter first:

.. list-table:: Solver result interpretation
   :widths: 25 35 40
   :header-rows: 1
   :class: longtable

   * - Quantity
     - API
     - Interpretation
   * - termination status
     - :py:attr:`Model.StatusString`
     - normal termination or a limit
   * - objective value
     - :py:attr:`Model.ObjVal`
     - expected scale and sign
   * - primal solution
     - :py:attr:`Var.X`
     - plausibility and feasibility
   * - solve time
     - :py:attr:`Model.SolverTime`
     - backend runtime and rough scale of effort

This order is intentional:

1. check the status before trusting any numeric answer
2. read the objective only after the status looks usable
3. inspect variable values for plausibility and feasibility
4. look at solve time after you know whether the solve succeeded

The same post-solve quantities are available whether the model was solved directly or
with parameter binding.

A simple default post-solve routine looks like this:

.. code-block:: python

    model.optimize({"alpha": 0.1})

    print(" * model.StatusString: ", model.StatusString)  # Check this first
    print(" * model.ObjVal: ", model.ObjVal)              # Then read the scalar objective
    print(" * x.X: ", x.X)                                # Then inspect variable values
    print(" * model.SolverTime: ", model.SolverTime)      # Finally note runtime

If the status points to a failure or a hard limit, treat the remaining values as
diagnostic information rather than as a trustworthy solution.
Check these items before tuning low-level options or drawing conclusions from the solution.

.. list-table:: Solver status codes
   :widths: 10 25 65
   :header-rows: 1
   :class: longtable

   * - Code
     - Name
     - Action
   * - 0
     - ``SOLVE_UNKNOWN``
     - Termination reason was not resolved to a known state. Rerun with higher verbosity and inspect the model and data path.
   * - 1
     - ``SOLVE_OPT_SUCCESS``
     - Normal termination: optimal (convex) or stationary point (nonconvex).
   * - 2
     - ``SOLVE_INFEASIBLE``
     - Constraints may be contradictory. Check the formulation.
   * - 3
     - ``SOLVE_UNBOUNDED``
     - Objective appears unbounded. Add missing bounds, regularization, or sign checks.
   * - 4
     - ``SOLVE_OVER_MAX_ITER``
     - Increase ``admm_max_iteration`` or loosen termination tolerances.
   * - 5
     - ``SOLVE_OVER_MAX_TIME``
     - Increase ``solver_max_time_limit``.
   * - 6
     - ``SOLVE_NAN_FOUND``
     - Numerical issue. Try scaling data or adjusting ``initial_penalty_param_value``.
   * - 7
     - ``SOLVE_PRE_FAILURE``
     - Failure before the main iterations completed, often during setup or preprocessing. Simplify the formulation and rerun with more logs.
   * - 8
     - ``SOLVE_EXCEPT_ERROR``
     - Backend exception. Rerun with higher verbosity and reduce to a small reproducer if it persists.
   * - 9
     - ``SOLVE_GET_SOL_FAILURE``
     - Optimization finished but the solver could not retrieve the final solution. Rerun and inspect backend logs or state.
   * - 10
     - ``SOLVE_ERROR``
     - Generic solver failure. Treat this as an error path rather than a convergence result.

Error Handling and Troubleshooting
-----------------------------------

Always check the solver status and handle different outcomes appropriately. Here is a complete pattern:

.. code-block:: python

    import admm
    import numpy as np

    # Create a simple model
    model = admm.Model()
    x = admm.Var("x", 5)
    model.setObjective(admm.sum(admm.square(x - 1)))
    
    # Add potentially conflicting constraints for demonstration
    # model.addConstr(x >= 2)  # This would conflict with the objective
    # model.addConstr(x <= 0)  # Making the problem infeasible
    
    model.optimize()
    
    # Always check status first
    if model.StatusString == "SOLVE_OPT_SUCCESS":
        print("✓ Optimization succeeded!")
        print(f"  Solution: {x.X}")
        print(f"  Objective: {model.ObjVal}")
    elif model.StatusString == "SOLVE_INFEASIBLE":
        print("✗ Problem is infeasible - constraints contradict each other")
        print("  Debugging tips:")
        print("  1. Check for conflicting constraints (e.g., x >= 1 and x <= 0)")
        print("  2. Relax constraints temporarily to identify the issue")
        print("  3. Use smaller test cases to isolate the problem")
    elif model.StatusString == "SOLVE_UNBOUNDED":
        print("✗ Problem is unbounded - objective can improve indefinitely")
        print("  Debugging tips:")
        print("  1. Add missing bounds on variables")
        print("  2. Check for regularization terms in the objective")
        print("  3. Verify the objective sense (minimize vs maximize)")
    elif model.StatusString == "SOLVE_OVER_MAX_ITER":
        print("⚠ Solver reached maximum iterations")
        print("  Try:")
        print("  1. Increase max iterations: model.setOption('admm_max_iteration', 5000)")
        print("  2. Loosen tolerance: model.setOption('termination_absolute_error_threshold', 1e-4)")
        print("  3. Scale your data to reasonable ranges (0.01 to 100)")
    elif model.StatusString == "SOLVE_NAN_FOUND":
        print("✗ Numerical issue detected - NaN in computation")
        print("  Try:")
        print("  1. Scale data to avoid very large/small numbers")
        print("  2. Check for division by zero or log of negative numbers")
        print("  3. Adjust initial penalty parameter: model.setOption('initial_penalty_param_value', 1.0)")
    elif model.StatusString == "SOLVE_OVER_MAX_TIME":
        print("⚠ Solver reached time limit")
        print("  Try:")
        print("  1. Increase time limit: model.setOption('solver_max_time_limit', 60.0)")
        print("  2. Simplify the model or use warm starts")
        print("  3. Check if the problem size is too large")
    else:
        print(f"✗ Optimization failed with status: {model.StatusString}")
        print(f"  Primal gap: {model.PrimalGap}")
        print(f"  Dual gap: {model.DualGap}")
        print(f"  Solver time: {model.SolverTime}")

Common Failure Patterns
~~~~~~~~~~~~~~~~~~~~~~~

**1. Infeasible Constraints**

.. code-block:: python

    # BAD: Contradictory constraints
    model.addConstr(x >= 2)
    model.addConstr(x <= 0)  # Impossible!
    
    # GOOD: Check constraint logic
    model.addConstr(x >= 0)
    model.addConstr(x <= 10)  # Reasonable bounds

**2. Unbounded Objective**

.. code-block:: python

    # BAD: Missing regularization
    model.setObjective(-admm.sum(x))  # Can go to negative infinity
    
    # GOOD: Add regularization or bounds
    model.setObjective(-admm.sum(x) + 0.1 * admm.sum(admm.square(x)))
    model.addConstr(x >= 0)
    model.addConstr(x <= 100)

**3. Numerical Scaling Issues**

.. code-block:: python

    # BAD: Very large coefficients
    A = np.random.randn(100, 50) * 1e6  # Too large!
    
    # GOOD: Scale data to reasonable ranges
    A = np.random.randn(100, 50)  # Standard normal
    # Or normalize your data
    A = A / np.max(np.abs(A))  # Scale to [-1, 1]

**4. Warm Start for Faster Convergence**

If you have a good initial guess, use warm starts:

.. code-block:: python

    x = admm.Var("x", n)
    x.start = initial_guess  # Provide initial value
    model.optimize()

What to Do When the Solve Does Not Succeed
------------------------------------------

The status table above is the authoritative reference, but the usual interpretation is:

- ``SOLVE_INFEASIBLE`` or ``SOLVE_UNBOUNDED`` usually points back to the formulation itself, so revisit the constraints, bounds, and signs before changing solver options.
- ``SOLVE_OVER_MAX_ITER`` or ``SOLVE_OVER_MAX_TIME`` means the model may be reasonable but the current budget was too small, so check the formulation first and then adjust options.
- ``SOLVE_NAN_FOUND`` often indicates a numerical scaling problem, so simplify the model, inspect data magnitudes, and try more conservative settings.
- codes ``7`` through ``10`` indicate backend or internal failures rather than a modeling verdict. If they persist on a small reproducer, capture the log output before filing an issue.

For parameter-binding syntax, see :ref:`Parameters <user-guide-parameters>`.
For option tuning after a status check, see :ref:`Solver Options <user-guide-solver-options>`.