.. include:: ../definition.hrst

.. _user-guide-udf:

User-Defined Proximal Extensions
================================

Read this page when the built-in modeling language gets you most of the way to the model you want, but one
proximal term is missing. The goal is not to replace the whole solver interface. The goal is to teach |ADMM|
one additional building block while keeping the rest of the model in the usual symbolic form.

A good rule of thumb is:

- stay with the built-in atoms whenever the term you need already appears in
  :ref:`Supported Building Blocks <user-guide-objective-building-blocks>`
- stay with the built-in atoms when a clean equivalent form is already recognized through
  :ref:`Symbolic Canonicalization <user-guide-symbolic-canonicalization>`
- move to a UDF when the overall modeling pattern is still a strong fit for |ADMM|, but one key proximal
  term is missing
- avoid UDFs when the difficulty is really unsupported global nonlinear structure rather than one extendable building block

A quick comparison helps:

- if convex sparsity encouragement is enough, use the built-in form ``lam * admm.norm(x, ord=1)``
- if you specifically need the nonconvex count of nonzeros :math:`\|x\|_0`, a UDF is the right extension point
- if the issue is an arbitrary coupled nonlinear program outside the documented support boundary, a UDF is
  usually not the right fix

This extension point is also one of the main places where |ADMM| differs from CVXPY.
Many of the linked examples are intentionally nonconvex — they fall outside the disciplined convex
programming (DCP) rules and cannot be passed to CVXPY. With |ADMM| and
a suitable UDF, they can still be modeled directly. In the nonconvex setting, the solver acts as a
practical local method and should be expected to converge to a locally optimal solution or stationary
point, not a globally optimal one.

What a UDF Provides
-------------------

A UDF is a Python class derived from :py:class:`UDFBase`. It does not replace variables, constraints, or
the rest of the objective. Instead, it gives the solver the function value and the proximal step for one
custom term.

|ADMM| supports two UDF interfaces. You implement **either** ``argmin`` or ``grad`` — not both:

.. list-table:: UDF interface: two paths
   :widths: 20 40 40
   :header-rows: 1
   :class: longtable

   * - Method
     - ``eval`` + ``argmin`` path
     - ``eval`` + ``grad`` path
   * - :py:meth:`UDFBase.arguments`
     - associated variables or expressions
     - associated variables or expressions
   * - :py:meth:`UDFBase.eval`
     - scalar function value
     - scalar function value
   * - :py:meth:`UDFBase.argmin`
     - list of proximal minimizers, or ``None``
     - *(not needed)*
   * - :py:meth:`UDFBase.grad`
     - *(not needed)*
     - list of gradient arrays

The ``argmin`` path is the original interface: you supply the closed-form proximal operator.
The ``grad`` path is the alternative: you supply the gradient and the C++ backend solves the
proximal subproblem automatically via gradient descent with backtracking line search.

.. list-table:: Choosing between ``argmin`` and ``grad``
   :widths: 50 50
   :header-rows: 1
   :class: longtable

   * - Use ``argmin`` when
     - Use ``grad`` when
   * - the proximal operator has a known closed form
     - the function is smooth but has no simple proximal formula
   * - you want maximum per-iteration efficiency
     - you want to prototype quickly without deriving the proximal step
   * - the function is nonsmooth (indicators, L0, rank)
     - the function is differentiable (log-cosh, Cauchy loss, custom loss)

The examples below show the kinds of terms that are natural UDF candidates: exact sparsity, rank,
manifold indicators, and other building blocks that are easy to describe through a proximal operator.
The ``grad`` path further opens UDFs to smooth functions from statistics, machine learning, and
signal processing where closed-form proximal operators are unavailable.

The UDF examples linked below are ordered from sparsity penalties to rank models to manifold or other
structured indicators. The last column highlights whether each model is within the usual CVXPY DCP rules, so it is easy to see where |ADMM| plus UDFs extend the modeling range.

.. list-table:: UDF examples
   :widths: 30 45 25
   :header-rows: 1
   :class: longtable

   * - Example
     - Common role
     - Follows DCP rules?
   * - :ref:`L0 Norm <udf-example-l0>`
     - promote sparsity
     - no
   * - :ref:`L0 Ball Indicator <udf-example-l0-ball>`
     - enforce a sparsity budget
     - no
   * - :ref:`L1/2 Quasi-Norm <udf-example-lhalf>`
     - classical nonconvex sparsity promotion
     - no
   * - :ref:`Group Sparsity <udf-example-group-sparsity>`
     - promote exact group sparsity
     - no
   * - :ref:`Matrix Rank Function <udf-example-rank>`
     - promote low-rank structure
     - no
   * - :ref:`Rank-r Indicator <udf-example-rank-r>`
     - enforce a rank cap
     - no
   * - :ref:`The Unit-Sphere Indicator <udf-example-unit-sphere>`
     - enforce unit-norm solutions
     - no
   * - :ref:`The Stiefel-Manifold Indicator <udf-example-stiefel>`
     - enforce orthonormal columns
     - no
   * - :ref:`The Simplex Indicator <udf-example-simplex>`
     - enforce probability or mixture weights
     - yes
   * - :ref:`The Binary Indicator <udf-example-binary>`
     - model binary decisions
     - no
   * - :ref:`Log-Cosh Loss <udf-grad-example-log-cosh>`
     - smooth robust regression (L1-like)
     - yes
   * - :ref:`Cauchy Loss <udf-grad-example-cauchy>`
     - heavy-tailed robust regression
     - no
   * - :ref:`Smooth Quantile Loss <udf-grad-example-smooth-quantile>`
     - quantile regression with smooth loss
     - yes
   * - :ref:`Wing Loss <udf-grad-example-wing-loss>`
     - facial landmark localization
     - no
   * - :ref:`Smooth Total Variation <udf-grad-example-smooth-tv>`
     - differentiable TV regularization
     - yes
   * - :ref:`Gamma Regression <udf-grad-example-gamma-regression>`
     - GLM deviance for positive responses
     - yes

Walkthrough: The ``L0`` Norm as a UDF
-------------------------------------

The tutorial below shows the full pattern on the ``L0`` norm. It is a good first UDF example because the
data-fit term is ordinary built-in modeling, while the custom sparsity penalty is the only missing piece.
The full optimization problem is

.. math::

   \min_x \; \frac{1}{2}\|x - y\|_2^2 + \lambda \|x\|_0,

where :math:`y` is the observed vector and :math:`\lambda > 0` controls how strongly sparsity is encouraged.
The role of the UDF is to supply the custom term :math:`\|x\|_0` in the form that |ADMM| needs.

The first custom method is :py:meth:`UDFBase.eval`. It returns the scalar value of the function at a concrete
numeric point. For the ``L0`` norm, the mathematical definition is

.. math::

   f(x) = \|x\|_0 = \#\{i : x_i \neq 0\},

so ``eval`` just counts how many entries are numerically nonzero:

.. code-block:: python

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.count_nonzero(np.abs(x) > 1e-12))

The second custom method is :py:meth:`UDFBase.argmin`. It returns the proximal minimizer that |ADMM| will call
repeatedly inside the algorithm. If the incoming point is :math:`v`, then this method must solve

.. math::

   \operatorname{prox}_{\lambda \|\cdot\|_0}(v)
   = \operatorname*{argmin}_x \; \lambda \|x\|_0 + \frac{1}{2}\|x - v\|_2^2,
   \qquad
   x_i^\star =
   \begin{cases}
      0, & |v_i| \le \sqrt{2\lambda}, \\
      v_i, & |v_i| > \sqrt{2\lambda}.
   \end{cases}

This is the classical hard-thresholding proximal operator, so the implementation is short:

.. code-block:: python

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        threshold = np.sqrt(2.0 * lamb)
        prox = np.where(np.abs(v) <= threshold, 0.0, v)
        return [prox.tolist()]

The remaining method, :py:meth:`UDFBase.arguments`, is the glue between the symbolic model and the numeric
``arglist`` received by ``eval`` and ``argmin``. In this example the custom function depends on exactly one
symbolic expression, namely :math:`x`, so ``arguments`` returns a one-element list. You can read this method as
"this UDF is a function of ``x``":

.. code-block:: python

    def arguments(self):
        return [self.arg]

Because ``arguments()`` returns ``[self.arg]``, the first numeric argument passed later into ``eval(arglist)``
and ``argmin(lamb, arglist)`` is the current value of that same symbolic quantity, accessed as ``arglist[0]``.

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    class L0Norm(admm.UDFBase):
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            return float(np.count_nonzero(np.abs(x) > 1e-12))

        def argmin(self, lamb, arglist):
            v = np.asarray(arglist[0], dtype=float)
            threshold = np.sqrt(2.0 * lamb)
            prox = np.where(np.abs(v) <= threshold, 0.0, v)
            return [prox.tolist()]

    y = np.array([0.2, 2.0, 0.6, 2.2])
    lam = 1.0

    model = admm.Model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + lam * L0Norm(x))
    model.optimize()

    print(" * x: ", np.asarray(x.X))  # Expected: ≈ [0, 2, 0, 2.2]
    print(" * model.ObjVal: ", model.ObjVal)  # Expected: ≈ 2.2
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_intro_l0.py


In this concrete example, :math:`\lambda = 1`, so the hard-threshold is

.. math::

   \sqrt{2\lambda} = \sqrt{2} \approx 1.414.

Therefore the coordinates behave as follows:

.. math::

   0.2 \mapsto 0,
   \qquad
   2.0 \mapsto 2.0,
   \qquad
   0.6 \mapsto 0,
   \qquad
   2.2 \mapsto 2.2.

So the hard-thresholding proximal step keeps the large entries and removes the small ones, giving

.. math::

   x^\star \approx [0,\; 2,\; 0,\; 2.2].

Small entries are removed; large entries survive unchanged.

Walkthrough: Log-Cosh Loss via ``grad``
---------------------------------------

Not every custom function has a convenient closed-form proximal operator. For smooth functions, you can
supply the **gradient** instead and let |ADMM| solve the proximal subproblem automatically. This is the
``grad`` path.

The tutorial below shows the full pattern on the log-cosh loss, a smooth robust alternative to least squares.
The full optimization problem is

.. math::

   \min_x \; \frac{1}{2}\|x - y\|_2^2 + \lambda \sum_i \log\!\bigl(\cosh(x_i)\bigr),

where :math:`y` is the observed vector, :math:`\lambda > 0` controls the strength of the robust penalty,
and :math:`\log(\cosh(\cdot))` behaves like :math:`\frac{1}{2}x^2` for small :math:`|x|` and like
:math:`|x|` for large :math:`|x|` — similar to the Huber loss but everywhere differentiable. Deriving a
closed-form proximal operator for this function is nontrivial, but the gradient is immediate. The role
of the UDF is to supply :math:`f(x) = \sum_i \log(\cosh(x_i))` and its gradient.

The first custom method is :py:meth:`UDFBase.eval`. It returns the scalar value of the function at a
concrete numeric point. The mathematical definition is

.. math::

   f(x) = \sum_i \log\!\bigl(\cosh(x_i)\bigr).

So ``eval`` sums the elementwise log-cosh values:

.. code-block:: python

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.sum(np.log(np.cosh(x))))

The second custom method is :py:meth:`UDFBase.grad`. Instead of the proximal minimizer (``argmin``),
it returns the **gradient** of the function. The gradient of log-cosh is:

.. math::

   \nabla f(x)_i = \tanh(x_i),

which is bounded in :math:`[-1, 1]`. This is the key to robustness: large values of :math:`x_i`
contribute a gradient of at most :math:`\pm 1`, limiting outlier influence. The implementation is
a single line:

.. code-block:: python

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return [np.tanh(x)]

Note that ``grad`` returns a **list of arrays** (one per argument), just as ``argmin`` does. Each array
must have the same shape as the corresponding input.

The remaining method, :py:meth:`UDFBase.arguments`, works exactly the same as in the ``argmin`` path.
It tells |ADMM| which symbolic expression this UDF depends on:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Because ``arguments()`` returns ``[self.arg]``, the first numeric argument passed into ``eval(arglist)``
and ``grad(arglist)`` is the current value of that symbolic quantity, accessed as ``arglist[0]``.

Behind the scenes, at each ADMM iteration the C++ backend uses the user-supplied ``eval`` and ``grad``
to solve the proximal subproblem

.. math::

   \operatorname*{argmin}_x \; \lambda \sum_i \log(\cosh(x_i)) + \frac{1}{2}\|x - v\|_2^2

via gradient descent with Armijo backtracking line search. The user **never** needs to derive the proximal
formula — just supply :math:`f(x)` and :math:`\nabla f(x)`.

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    class LogCoshLoss(admm.UDFBase):
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            return float(np.sum(np.log(np.cosh(x))))

        def grad(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            return [np.tanh(x)]

    y = np.array([0.2, 5.0, -3.0, 0.1])
    lam = 0.5

    model = admm.Model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + lam * LogCoshLoss(x))
    model.optimize()

    print(" * x: ", np.asarray(x.X))
    print(" * model.ObjVal: ", model.ObjVal)
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Usage is identical to the ``argmin`` path — the solver detects which method is present and chooses the
solve strategy accordingly.

In this concrete example, :math:`\lambda = 0.5` and :math:`y = [0.2, 5.0, -3.0, 0.1]`. The optimal
:math:`x` balances the data-fit term :math:`\frac{1}{2}\|x - y\|_2^2` (pulling :math:`x` toward
:math:`y`) against the log-cosh penalty (pulling :math:`x` toward zero). Because log-cosh grows like
:math:`|x|` for large values, the penalty is less harsh on large entries than an L2 penalty would be,
but stronger than no penalty at all. The solver recovers a regularized estimate where large entries
(:math:`y_2 = 5`, :math:`y_3 = -3`) are shrunk modestly and small entries (:math:`y_1 = 0.2`,
:math:`y_4 = 0.1`) are shrunk more strongly.

Multi-argument ``grad`` UDFs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-argument UDFs work the same way with the ``grad`` path. For a function of two variables,
``arguments`` returns two symbolic quantities, and ``grad`` returns a list of two gradient arrays —
one per variable:

.. code-block:: python

    class CoupledQuad(admm.UDFBase):
        """f(x, y) = sum((x - y)^2)"""
        def __init__(self, x_arg, y_arg):
            self.x_arg = x_arg
            self.y_arg = y_arg

        def arguments(self):
            return [self.x_arg, self.y_arg]

        def eval(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            y = np.asarray(arglist[1], dtype=float)
            return float(np.sum((x - y) ** 2))

        def grad(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            y = np.asarray(arglist[1], dtype=float)
            return [2 * (x - y), -2 * (x - y)]

Here ``grad`` returns ``[df/dx, df/dy]`` — each array has the same shape as the corresponding input
variable.

The ``grad`` path makes |ADMM| a natural fit for custom smooth losses from statistics and machine learning
— quantile regression, Cauchy loss, wing loss, and other functions where the gradient is easy to write but the
proximal operator is not.

More UDF examples are collected in
:ref:`Examples with User-Defined Proximal Functions <examples-udf>` in the :ref:`Examples chapter <doc-examples>`.
For the scope boundary that usually makes UDFs worthwhile, see
:ref:`Support Boundary <user-guide-support-boundary>`.