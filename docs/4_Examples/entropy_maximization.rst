.. include:: ../definition.hrst

.. _example-entropy-maximization:

Entropy Maximization
====================

Entropy maximization starts by treating :math:`x` as a probability vector.
Each entry :math:`x_i` is the probability assigned to outcome :math:`i`, so
the conditions :math:`x \ge 0` and :math:`\mathbf{1}^\top x = 1` force
:math:`x` to live on the probability simplex.

Among all probability vectors that satisfy the side information, we want the
one with the largest Shannon entropy, meaning the least concentrated and most
diffuse distribution.

.. math::

   \begin{array}{ll}
   \max\limits_x & -\sum_{i=1}^{n} x_i \log x_i \\
   \text{s.t.} & A x = b, \\
               & F x \le g, \\
               & \mathbf{1}^\top x = 1, \\
               & x \ge 0.
   \end{array}

Here :math:`A x = b` and :math:`F x \le g` encode additional information we
want the distribution to respect.

The API used in these examples is a minimization API, so the code will not
write the objective with a leading minus sign. Instead, it minimizes
:math:`\sum_i x_i \log x_i`, which is exactly the negative of the Shannon
entropy. Minimizing negative entropy is the same as maximizing entropy.

**Step 1: Build a feasible probability model**

We first create a random probability vector ``x0`` and then use it to generate
the right-hand sides ``b`` and ``g``. That guarantees the optimization problem
is feasible, because ``x0`` itself already satisfies the constraints.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 12
    m = 3
    p = 2
    x0 = np.random.rand(n)
    x0 = x0 / np.sum(x0)
    A = np.random.randn(m, n)
    b = A @ x0
    F = np.random.randn(p, n)
    g = F @ x0 + np.random.rand(p)

**Step 2: Create the model and the probability variable**

The decision variable is another length-``n`` vector called ``x``. It will be
the probability distribution chosen by the solver.

.. code-block:: python

    model = admm.Model()
    x = admm.Var("x", n)

**Step 3: Write the entropy objective in minimization form**

The Shannon entropy is
:math:`H(x) = -\sum_i x_i \log x_i`. The symbolic function
``admm.entropy(x)`` gives the entrywise term :math:`x_i \log x_i`, so
``admm.sum(admm.entropy(x))`` is :math:`\sum_i x_i \log x_i = -H(x)`.
Minimizing that quantity is therefore the same as maximizing Shannon entropy.

.. code-block:: python

    model.setObjective(admm.sum(admm.entropy(x)))

**Step 4: Add the affine side information and simplex constraints**

The equality and inequality constraints encode the external information we want
the distribution to satisfy. The last two constraints,
``admm.sum(x) == 1`` and ``x >= 0``, are what make ``x`` a probability vector.

.. code-block:: python

    model.addConstr(A @ x == b)
    model.addConstr(F @ x <= g)
    model.addConstr(admm.sum(x) == 1)
    model.addConstr(x >= 0)

**Step 5: Solve and inspect the result**

After solving, ``model.ObjVal`` is the optimal value of the minimization form
:math:`\sum_i x_i \log x_i`, while ``model.StatusString`` tells us whether the
solver succeeded.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -2.4722786823012264
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    n = 12
    m = 3
    p = 2
    x0 = np.random.rand(n)
    x0 = x0 / np.sum(x0)
    A = np.random.randn(m, n)
    b = A @ x0
    F = np.random.randn(p, n)
    g = F @ x0 + np.random.rand(p)

    model = admm.Model()
    x = admm.Var("x", n)
    model.setObjective(admm.sum(admm.entropy(x)))  # minimize sum(x*log(x)) = maximize Shannon entropy
    model.addConstr(A @ x == b)
    model.addConstr(F @ x <= g)
    model.addConstr(admm.sum(x) == 1)
    model.addConstr(x >= 0)
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -2.4722786823012264
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/entropy_maximization.py


The solution is the maximum-entropy distribution on the simplex
:math:`\{x \ge 0,\; \mathbf{1}^\top x = 1\}` that satisfies the moment constraints
:math:`Ax = b` and :math:`Fx \le g`. ADMM handles :py:func:`admm.entropy` and the
simplex projection natively.