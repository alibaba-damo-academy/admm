.. include:: ../definition.hrst

.. _example-least-squares:

Least Squares
=============

Least squares is the basic model for fitting a linear predictor to observed data.
The key idea is the *residual*, meaning the gap between what the model predicts and
what we actually observed.

The standard optimization model is

.. math::

   \begin{array}{ll}
   \min\limits_x & \|A x - b\|_2^2.
   \end{array}

Here :math:`A \in \mathbb{R}^{m \times n}` is the design matrix,
:math:`b \in \mathbb{R}^m` is the observation vector, and
:math:`x \in \mathbb{R}^n` is the parameter vector we want to estimate.

If the model predicts :math:`A x`, then the residual is
:math:`r = A x - b`. Least squares makes all of those residual entries small in a
single quadratic objective.

**Step 1: Generate a synthetic fitting problem**

We create a random matrix :math:`A`, a hidden ground-truth parameter vector
:math:`x_{\text{true}}`, and then build observations
:math:`b = A x_{\text{true}} + \text{noise}`. That gives us data with a clear
linear signal plus a small amount of measurement noise.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 40
    n = 12
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true + 0.1 * np.random.randn(m)

**Step 2: Create the model and decision variable**

The unknown quantity is the coefficient vector :math:`x`. In the symbolic API,
``admm.Var("x", n)`` creates a length-:math:`n` vector variable.

.. code-block:: python

    model = admm.Model()
    x = admm.Var("x", n)

**Step 3: Write the residual expression and objective**

The prediction produced by the current candidate vector is ``A @ x``. Subtracting
the observed data ``b`` gives the residual vector.

.. code-block:: python

    residual = A @ x - b
    model.setObjective(admm.sum(admm.square(residual)))

The first line is the code version of :math:`r = A x - b`. The second line squares
those residual entries and sums them, which is exactly the symbolic form of
minimizing :math:`\|A x - b\|_2^2`.

**Step 4: Add constraints**

This least-squares example has no explicit constraints, so there are no
``model.addConstr(...)`` calls. Once the objective is set, the model is fully
specified.

**Step 5: Solve and inspect the result**

After optimizing, we print the standard solver outputs.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 0.2947794914868591
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    m = 40
    n = 12
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true + 0.1 * np.random.randn(m)

    model = admm.Model()
    x = admm.Var("x", n)
    model.setObjective(admm.sum(admm.square(A @ x - b)))
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 0.2947794914868591
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/least_squares.py


The solution :math:`x` minimizes :math:`\|Ax - b\|_2^2` — one line of ADMM code for
the objective, no manual normal-equation derivation needed.