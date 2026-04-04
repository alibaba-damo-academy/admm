.. include:: ../definition.hrst

.. _user-guide-variables:

Variables
=========

After you have checked :ref:`Supported Problem Structure <user-guide-supported-problem-structure>`, the
next modeling decision is the variable shape. For most first models, the useful order is:

1. choose scalar, vector, or matrix form
2. decide whether any sign or cone structure belongs to the variable itself
3. learn warm starts and reshaping only after the declaration pattern feels natural

Variable dimensions are limited to at most 2; higher-dimensional array variables are not supported.

Choose the Shape First
----------------------

Start with the object you are trying to solve for:

- use a scalar when the unknown is one number
- use a vector when the unknown is a list of parallel coefficients
- use a matrix when the unknown is naturally two-dimensional

.. list-table:: Variable creation patterns
   :widths: 15 31 15 39
   :header-rows: 1
   :class: longtable

   * - Pattern
     - Example
     - Shape
     - Typical use
   * - scalar
     - ``admm.Var("offset")``
     - ``()``
     - single coefficient or intercept
   * - vector
     - ``admm.Var("x", n)``
     - ``(n,)``
     - regression coefficients or allocations
   * - matrix
     - ``admm.Var("X", m, n)``
     - ``(m, n)``
     - linear maps, images, covariance objects

.. code-block:: python

    x0 = admm.Var("offset")
    x1 = admm.Var("direction", 2)
    X = admm.Var("matrix", 3, 4)

Variable names are optional but strongly recommended for readability and model persistence.

Attributes vs. Explicit Constraints
-----------------------------------

Once the shape is right, decide whether a property is intrinsic to one variable or is just one model
condition among many.

- use a variable attribute when every feasible value of that variable must have the property by design
- use :py:meth:`Model.addConstr` when the rule relates expressions, is easier to read as a separate line,
  or is only part of the final model logic

The library supports several structural attribute keywords directly in :py:meth:`Var.__init__`.
The active attributes can be inspected through :py:attr:`Var.attr`.
Mathematically, these attributes restrict the decision variable to a set such as
:math:`\mathbb{R}_+^n`, :math:`\mathbb{S}^n`, or :math:`\mathbb{S}_+^n` before any additional constraints are
added.

.. list-table:: Supported variable attributes
   :widths: 21 32 47
   :header-rows: 1
   :class: longtable

   * - Keyword
     - Meaning
     - Example
   * - ``nonneg=True``
     - Elementwise nonnegative variable
     - ``admm.Var("x", 6, nonneg=True)``
   * - ``nonpos=True``
     - Elementwise nonpositive variable
     - ``admm.Var("x", 6, nonpos=True)``
   * - ``symmetric=True``
     - Symmetric matrix
     - ``admm.Var("S", 3, 3, symmetric=True)``
   * - ``diag=True``
     - Diagonal matrix structure
     - ``admm.Var("D", 3, 3, diag=True)``
   * - ``PSD=True``
     - Symmetric positive semidefinite matrix structure
     - ``admm.Var("X", 5, 5, PSD=True)``
   * - ``NSD=True``
     - Symmetric negative semidefinite matrix structure
     - ``admm.Var("X", 5, 5, NSD=True)``
   * - ``pos=True``
     - Elementwise positive variable
     - ``admm.Var("x", 6, pos=True)``
   * - ``neg=True``
     - Elementwise negative variable
     - ``admm.Var("x", 6, neg=True)``

For example, the statements

.. math::

   x \ge 0, \qquad X \succeq 0

can often be modeled more directly by declaring ``x`` as nonnegative or ``X`` as PSD at creation time.

For PSD structure, two common forms are:

.. code-block:: python

    # Method 1: declare the variable itself as PSD
    X = admm.Var("X", n, n, PSD=True)

    # Method 2: create the matrix variable first, then add an explicit PSD constraint
    X = admm.Var("X", n, n)  # Unstructured matrix variable
    model.addConstr(X >> 0)  # PSD cone constraint

Use the first form when PSD is intrinsic to the variable. Use the second when you want PSD written as an
explicit constraint or you are teaching the model through explicit feasibility conditions. For more on the
constraint side of that choice, see :ref:`Constraints <user-guide-constraints>`.

Accessing Solution Values
-------------------------

After calling :py:meth:`Model.optimize()`, you can retrieve the optimal solution from a variable using the
:py:attr:`Var.X` attribute. This returns a NumPy array containing the numerical values.

.. code-block:: python

    import admm
    import numpy as np

    # Create and solve a simple model
    model = admm.Model()
    x = admm.Var("x", 5)
    model.setObjective(admm.sum(admm.square(x - 1)))
    model.optimize()

    # Access the solution
    solution = x.X
    print(f"Optimal solution: {solution}")
    print(f"Type: {type(solution)}")  # <class 'numpy.ndarray'>
    print(f"Shape: {solution.shape}")  # (5,)

For matrix variables, :py:attr:`Var.X` returns a 2D NumPy array:

.. code-block:: python

    X = admm.Var("X", 3, 3, PSD=True)
    model.optimize()
    
    # Access the matrix solution
    X_optimal = X.X
    print(f"Matrix shape: {X_optimal.shape}")  # (3, 3)

.. note::

    The :py:attr:`Var.X` attribute is only populated after a successful optimization call.
    Before calling :py:meth:`Model.optimize()`, accessing :py:attr:`Var.X` will return ``None``.

Always check the solver status before using the solution:

.. code-block:: python

    model.optimize()

    if model.StatusString == "SOLVE_OPT_SUCCESS":
        print(f"Solution: {x.X}")
    else:
        print(f"Optimization failed: {model.StatusString}")

Warm Start
----------

Once variable creation feels routine, the next convenience to learn is warm start. A variable can be given a
starting point via :py:attr:`Var.start`, which is useful when you already have a good approximate solution
from a previous run or an external heuristic.

.. code-block:: python

    x = admm.Var("x", n)
    x.start = initial_guess
    model.optimize()

Basic Operators
---------------

After declaration, variables behave like symbolic scalar, vector, and matrix objects. The operators below
cover most day-to-day modeling syntax.

.. list-table:: Supported variable operators
   :widths: 16 37 22 25
   :header-rows: 1
   :class: longtable

   * - Operator
     - Meaning
     - Example
     - NumPy analogue
   * - ``+``
     - Elementwise addition
     - ``a + b``
     - ``numpy.add``
   * - ``-``
     - Elementwise subtraction
     - ``a - b``
     - ``numpy.subtract``
   * - ``*``
     - Elementwise multiplication
     - ``a * b``
     - ``numpy.multiply``
   * - ``/``
     - Elementwise division
     - ``a / b``
     - ``numpy.divide``
   * - ``@``
     - Matrix multiplication
     - ``A @ x``
     - ``numpy.matmul``
   * - ``**``
     - Elementwise power
     - ``x ** 2``
     - ``numpy.power``

Broadcasting follows NumPy rules when shapes are compatible.

Variable Shapes and Transformations
-----------------------------------

When you start writing block constraints or matrix reformulations, these shape tools are the next ones to
reach for.

.. list-table:: Supported variable shape tools
   :widths: 30 32 38
   :header-rows: 1
   :class: longtable

   * - Tool
     - Example
     - Output effect
   * - :py:attr:`TensorLike.shape`
     - ``X.shape``
     - returns dimensions
   * - :py:attr:`TensorLike.ndim`
     - ``X.ndim``
     - returns rank
   * - :py:attr:`TensorLike.T`
     - ``X.T``
     - transpose
   * - :py:meth:`TensorLike.reshape`
     - ``x.reshape(m, n)``
     - reindex without changing element count
   * - slicing
     - ``X[:, 0]``, ``X[1:3, :]``
     - subobject selection

.. code-block:: python

    X = admm.Var("X", 2, 3)

    XT = X.T
    left_block = X[:, 0:2]
    first_column = X[:, 0]
    reshaped = X.reshape(3, 2)

These transformations follow NumPy-style conventions and are often the cleanest way to express block
constraints or matrix-to-vector reformulations.