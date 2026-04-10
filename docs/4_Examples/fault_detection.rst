.. include:: ../definition.hrst

.. _example-fault-detection:

Fault Detection
===============

Fault detection starts with a modeling choice: instead of using a hard Boolean
variable for each possible fault, we use a *relaxed fault indicator*
:math:`x_i \in [0, 1]`. Values near 0 mean "probably healthy," values near 1
mean "strongly suspected fault," and intermediate values act like soft scores.

The sensing model says the measurement vector :math:`y` should be explained by
:math:`A x`, so the optimization problem balances measurement fit against a
preference for sparse fault patterns.

.. math::

   \begin{array}{ll}
   \min\limits_x & \|Ax - y\|_2^2 + \tau \mathbf{1}^\top x \\
   \text{s.t.} & 0 \le x \le 1.
   \end{array}

Each part of this model has a specific job:

- :math:`x` stores the relaxed fault indicators
- :math:`0 \le x \le 1` keeps those indicators inside the box
- :math:`\|A x - y\|_2^2` rewards explanations that match the measured data
- :math:`\tau \mathbf{1}^\top x` discourages using too many faults at once

This is a standard convex relaxation of a sparse combinatorial fault-selection
problem.

**Step 1: Generate a sparse synthetic fault scenario**

We create a random sensing matrix ``A``, a sparse ground-truth fault vector
``x_true``, and noisy measurements ``y``. The parameter ``tau`` is chosen from
the planted fault probability and noise level so the linear penalty has a
meaningful scale.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 200
    m = 40
    p_fault = 0.03
    snr = 5.0
    sigma = np.sqrt(p_fault * n / (snr ** 2))
    A = np.random.randn(m, n)
    x_true = (np.random.rand(n) <= p_fault).astype(float)
    y = A @ x_true + sigma * np.random.randn(m)
    tau = 2 * np.log(1 / p_fault - 1) * sigma ** 2

**Step 2: Create the model, increase the iteration cap, and define ``x``**

The decision variable ``x`` is the relaxed fault-indicator vector. We also
raise ``admm_max_iteration`` to ``5000`` to match the reference runnable
example for this noisy box-constrained problem.

.. code-block:: python

    model = admm.Model()
    model.setOption(admm.Options.admm_max_iteration, 5000)
    x = admm.Var("x", n)

**Step 3: Write the quadratic data-fit term and the linear sparsity term**

The residual vector is ``A @ x - y``. Squaring and summing that residual gives
the quadratic measurement-fit term, while ``tau * admm.sum(x)`` adds a linear
penalty on the total fault mass. Because ``x >= 0``, this linear term pushes
many entries of ``x`` toward zero.

.. code-block:: python

    residual = A @ x - y
    model.setObjective(admm.sum(admm.square(residual)) + tau * admm.sum(x))

**Step 4: Add the box constraints**

These two constraints are the relaxation itself. They keep every component of
``x`` between 0 and 1, so the solution behaves like a vector of soft fault
scores rather than an unconstrained regression coefficient.

.. code-block:: python

    model.addConstr(x >= 0)
    model.addConstr(x <= 1)

**Step 5: Solve and inspect the result**

Once optimized, ``model.ObjVal`` reports the best tradeoff between residual fit
and sparsity encouragement, and ``model.StatusString`` reports whether the
solver finished successfully.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 15.294052961638492
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 200
    m = 40
    p_fault = 0.03
    snr = 5.0
    sigma = np.sqrt(p_fault * n / (snr ** 2))
    A = np.random.randn(m, n)
    x_true = (np.random.rand(n) <= p_fault).astype(float)
    y = A @ x_true + sigma * np.random.randn(m)
    tau = 2 * np.log(1 / p_fault - 1) * sigma ** 2

    model = admm.Model()
    model.setOption(admm.Options.admm_max_iteration, 5000)
    x = admm.Var("x", n)
    model.setObjective(admm.sum(admm.square(A @ x - y)) + tau * admm.sum(x))
    model.addConstr(x >= 0)
    model.addConstr(x <= 1)
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 15.294052961638492
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/fault_detection.py


The solution :math:`x \in [0,1]^n` is a relaxed fault indicator:
:math:`\|Ax - y\|_2^2` fits the measurements while :math:`\mathbf{1}^\top x` penalizes
activating too many faults. Entries near 1 flag likely faults.