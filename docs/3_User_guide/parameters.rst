.. include:: ../definition.hrst

.. _user-guide-parameters:

Parameters
==========

Parameters are named placeholders whose numeric values are supplied when solving the
model. Use them when the symbolic structure stays fixed but some data changes between
solves. That lets you build the model once, then reuse it with different numeric inputs.
If you want the smallest one-solve baseline before this reusable pattern, start with
:ref:`Minimal Model <user-guide-minimal-model>`.

.. list-table:: Parameter patterns
   :widths: 28 44 28
   :header-rows: 1
   :class: longtable

   * - Pattern
     - Example
     - Shape
   * - scalar parameter
     - ``admm.Param("alpha")``
     - ``()``
   * - vector parameter
     - ``admm.Param("b", n)``
     - ``(n,)``
   * - matrix parameter
     - ``admm.Param("A", m, n)``
     - ``(m, n)``


.. rst-class:: landing-page-section-heading

.. rubric:: Step 1: Declare parameters with the right shape

Declare each parameter with the same shape as the numeric data you plan to bind later.
Scalars use no shape arguments, vectors use one dimension, and matrices use two.

.. code-block:: python

    lam = admm.Param("lam")
    b = admm.Param("b", m)
    A = admm.Param("A", m, n)

The parameter name is important because solve-time binding uses that name as the key.

.. rst-class:: landing-page-section-heading

.. rubric:: Step 2: Build the symbolic model once

After declaring the parameters, use them directly inside expressions just as you would
use constants. The difference is that their values will not be fixed until solve time.

For example, in a Lasso-style model, ``lam`` can stay symbolic while the rest of the
problem structure is built only once.

.. code-block:: python

    model = admm.Model()
    x = admm.Var("x", n)
    lam = admm.Param("lam")

    model.setObjective(
        0.5 * admm.sum(admm.square(A @ x - b))
        + lam * admm.norm(x, ord=1)
    )

.. rst-class:: landing-page-section-heading

.. rubric:: Step 3: Bind values when you call optimize()

Solve a model with :py:meth:`Model.optimize`.
If the formulation contains parameters, pass a dictionary that maps parameter names to numeric values.
The dictionary keys must match the declared parameter names, and the supplied scalar, vector, or matrix data
must match the declared shape.

.. code-block:: python

    model.optimize()                    # Solve when no parameters are present

    model.optimize({"alpha": 0.1})      # Bind one scalar parameter

    model.optimize({
        "alpha": 0.1,                   # Scalar parameter
        "b": [1, 2, 3],                 # Vector parameter
        "A": [[1, 0], [0, 1], [1, 1]],  # Matrix parameter
    })


Parameter binding fixes the numeric data while keeping the symbolic model unchanged.
At solve time, the parameter dictionary is the extra input; after the call returns, the
usual post-solve quantities are available just as in any other model.

.. rst-class:: landing-page-section-heading

.. rubric:: Step 4: Reuse the same model structure

If only the parameter values change, call :py:meth:`Model.optimize` again with a new
dictionary. There is no need to rebuild the variables, objective, or constraints each time.
This is the main payoff of modeling with parameters.

.. list-table:: Parameter solve-time checklist
   :widths: 30 42 44
   :header-rows: 1
   :class: longtable

   * - Item
     - API
     - Interpretation
   * - parameter values
     - :py:meth:`Model.optimize`
     - pass a dictionary that assigns numeric data to :py:class:`Param` objects by name
   * - post-solve inspection
     - :py:attr:`Model.StatusString`, :py:attr:`Model.ObjVal`, :py:attr:`Model.SolverTime`, :py:attr:`Var.X`
     - inspect termination, objective scale, runtime, and variable values after the bound solve finishes


.. rst-class:: landing-page-section-heading

.. rubric:: Complete Example: Reusing a Lasso Model

The example below builds one Lasso-style model and solves it twice with two different
regularization weights. This is the simplest way to see why parameters matter: the
model structure stays fixed, but the parameter value changes the result.
Depending on the backend and verbosity settings, solver logs may appear before the
labeled prints below.

In mathematical form, both solves use the same symbolic problem family:

.. math::

   \min_x \quad \frac{1}{2}\|Ax - b\|_2^2 + \lambda \|x\|_1

The symbolic model stays fixed while :math:`\lambda` changes
between solves. In the code, the mathematical parameter :math:`\lambda` is represented
by ``lam``, which is bound to different numeric values when
:py:meth:`Model.optimize` is called.

.. code-block:: python

    import admm
    import numpy as np

    np.random.seed(1)                   # Reproducible data
    m = 30
    n = 10

    A = np.random.randn(m, n)           # Data matrix
    b = np.random.randn(m)              # Observations

    model = admm.Model()
    x = admm.Var("x", n)                # Regression coefficients
    lam = admm.Param("lam")             # Regularization weight as a parameter

    model.setObjective(
        0.5 * admm.sum(admm.square(A @ x - b))
        + lam * admm.norm(x, ord=1)
    )
    model.setOption(admm.Options.admm_max_iteration, 5000)

    model.optimize({"lam": 0.05})
    x_small_penalty = np.asarray(x.X).copy()

    print(" * first solve model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * first solve model.ObjVal: ", round(model.ObjVal, 6))    # Expected: finite scalar objective
    print(" * first solve x.X: ", np.round(x_small_penalty, 6))       # Expected: less regularized coefficients
    print(" * first solve ||x||_1: ", round(np.linalg.norm(x_small_penalty, 1), 6))

    model.optimize({"lam": 0.2})
    x_large_penalty = np.asarray(x.X).copy()

    print(" * second solve model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * second solve model.ObjVal: ", round(model.ObjVal, 6))    # Expected: finite scalar objective
    print(" * second solve x.X: ", np.round(x_large_penalty, 6))       # Expected: more heavily shrunk coefficients
    print(" * second solve ||x||_1: ", round(np.linalg.norm(x_large_penalty, 1), 6))


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/lasso_regression.py


The two solves share the same symbolic model; only the numeric binding of ``lam``
changes. Larger :math:`\lambda` penalizes :math:`\|x\|_1` more heavily, driving more
coefficients to zero. Compare the printed :math:`\|x\|_1` values across runs to see the
effect directly.

For the surrounding workflow, see :ref:`Modeling Workflow <user-guide-modeling-workflow>`.
For the full status-code table and post-solve interpretation, see :ref:`Solve the Model <user-guide-solve-the-model>`.