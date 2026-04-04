.. include:: ../definition.hrst

.. _user-guide-solver-options:

Solver Options
==============

|ADMM| exposes many solver options, but most first models should be solved with the defaults.
Treat tuning as a second pass:

1. solve once
2. read :py:attr:`Model.StatusString`
3. change the smallest option that matches the status or debugging need

In practice, the first knobs worth touching are the iteration budget, time limit, and verbosity.
Leave the more specialized settings alone until the default run tells you why you need them.

.. rubric:: Iteration and time budget

:py:attr:`admm_max_iteration <OptionConstClass.admm_max_iteration>` · default ``1000``
   Raise this first if the status is ``SOLVE_OVER_MAX_ITER``.
   A common next try is ``5000``–``10000``.

:py:attr:`solver_max_time_limit <OptionConstClass.solver_max_time_limit>` · default ``1000 s``
   Raise this if the model is otherwise reasonable but is stopping on wall-clock time.

.. rubric:: Output control

:py:attr:`solver_verbosity_level <OptionConstClass.solver_verbosity_level>` · default ``2`` (SUMMARY)
   Set to ``1`` (INFO) or ``0`` (DETAIL) when diagnosing a stubborn solve.
   Use ``3`` (SILENT) to suppress output entirely.

.. note::
   The numbering is: 0 = most verbose (DETAIL), 3 = silent.
   This is the opposite convention of some other solvers.

.. rubric:: Convergence tolerances

:py:attr:`termination_absolute_error_threshold <OptionConstClass.termination_absolute_error_threshold>` · default ``1e-6``
   Loosen to ``1e-4`` for faster approximate answers, or tighten only when you truly need more accuracy.

:py:attr:`termination_relative_error_threshold <OptionConstClass.termination_relative_error_threshold>` · default ``1e-6``
   Usually move this together with the absolute threshold rather than changing only one of them.

.. rubric:: Penalty parameter

:py:attr:`initial_penalty_param_value <OptionConstClass.initial_penalty_param_value>` · default ``1.0``
   Leave this alone at first; try ``0.1`` or ``10.0`` only when convergence is slow or numerically awkward.

:py:attr:`penalty_param_auto <OptionConstClass.penalty_param_auto>` · default ``1`` (enabled)
   Keep automatic penalty adjustment on for routine runs.
   Disable it only for deliberate manual rho tuning.

.. rubric:: Constraint handling

:py:attr:`strict_constraint_tol <OptionConstClass.strict_constraint_tol>` · default ``1e-8``
   Usually leave this at the default unless your formulation explicitly relies on strict constraints.

----

As a beginner rule, do not guess at options before you have a status to react to. If the solve succeeds,
you may not need any tuning at all. If it stops on an iteration or time limit, raise the corresponding
budget first. If the logs are not informative enough, increase verbosity. Only after that should you reach
for tighter tolerances or manual penalty tuning.

.. code-block:: python

    model.optimize()
    print("status:", model.StatusString)

    if model.StatusString == "SOLVE_OVER_MAX_ITER":
        model.setOption(admm.Options.admm_max_iteration, 5000)
        model.optimize()
    elif model.StatusString == "SOLVE_OVER_MAX_TIME":
        model.setOption(admm.Options.solver_max_time_limit, 5000)
        model.optimize()

    # If you need more logs on the next run:
    model.setOption(admm.Options.solver_verbosity_level, 1)

For the full option list, see the :doc:`../5_API_Document/index`.
For status interpretation after a run, see :ref:`Solve the Model <user-guide-solve-the-model>`.
