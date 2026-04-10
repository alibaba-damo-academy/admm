.. include:: ../definition.hrst

.. _example-water-filling:

Water Filling
=============

Water filling is a resource-allocation model. We have a fixed total budget
:math:`P` to distribute across channels, and each channel already starts with a
baseline level :math:`\alpha_i > 0`.

Giving more resource to a channel improves utility, but with diminishing
returns. That is why the objective uses a logarithm.

.. math::

   \begin{array}{ll}
   \max\limits_x & \sum_{i=1}^{n} \log(\alpha_i + x_i) \\
   \text{s.t.} & \sum_{i=1}^{n} x_i = P, \\
               & x_i \ge 0,\quad i=1,\ldots,n.
   \end{array}

Here :math:`x_i` is the extra resource assigned to channel :math:`i`,
:math:`\alpha_i` is its baseline level, and :math:`P` is the total amount we
are allowed to allocate.

Because the API in these examples is written as a minimization API, the code
uses the equivalent objective
:math:`\min_x \sum_i -\log(\alpha_i + x_i)`. Minimizing negative utility is the
same as maximizing utility.

**Step 1: Choose channel baselines and the total budget**

This example uses five channels with different baseline levels ``alpha`` and a
total available resource ``total_power``.

.. code-block:: python

    import numpy as np
    import admm

    alpha = np.array([0.5, 0.8, 1.0, 1.3, 1.6])
    total_power = 2.0
    n = len(alpha)

**Step 2: Create the model and the allocation variable**

The decision variable ``x`` is the vector of extra resource assignments. It has
one entry per channel.

.. code-block:: python

    model = admm.Model()
    x = admm.Var("x", n)

**Step 3: Write the objective in minimization form**

The natural application story is "maximize total log-utility." In code, we flip
the sign and minimize the negative of that same quantity:

.. code-block:: python

    model.setObjective(admm.sum(-admm.log(alpha + x)))

This line is the direct minimization-form equivalent of
:math:`\max_x \sum_i \log(\alpha_i + x_i)`.

**Step 4: Add the budget and nonnegativity constraints**

The equality ``admm.sum(x) == total_power`` says we must use exactly the full
resource budget. The inequality ``x >= 0`` says resource can be added to a
channel but not taken away.

.. code-block:: python

    model.addConstr(admm.sum(x) == total_power)
    model.addConstr(x >= 0)

**Step 5: Solve and inspect the result**

After optimization, ``model.ObjVal`` is the optimal value of the minimization
form, and ``model.StatusString`` reports solver success.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -1.8158925751409778
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    alpha = np.array([0.5, 0.8, 1.0, 1.3, 1.6])
    total_power = 2.0
    n = len(alpha)

    model = admm.Model()
    x = admm.Var("x", n)
    model.setObjective(admm.sum(-admm.log(alpha + x)))
    model.addConstr(admm.sum(x) == total_power)
    model.addConstr(x >= 0)
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -1.8158925751409778
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/water_filling.py


The concavity of :math:`\log(\alpha_i + x_i)` produces the classic water-filling
allocation: channels with better :math:`\alpha_i` receive more budget, but
diminishing returns prevent over-concentration.