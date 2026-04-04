.. include:: ../definition.hrst

.. _user-guide-objective-building-blocks:

Supported Building Blocks
=========================

.. note::
   Read :ref:`Objective <user-guide-objective>` and :ref:`Constraints <user-guide-constraints>` first.
   Then use this page as a compact catalog when you know what role a term should play in the model and
   need the right built-in atom or declaration.

This page is easiest to use after you can already separate a model into three questions:

1. what measures mismatch to the data?
2. what kind of solution do you prefer among many plausible ones?
3. what must hold exactly, with no trade-off allowed?

Once you know which question you are answering, jump to the matching section below and then use the
compact tables near the end as a lookup reference.

.. list-table:: How this reference is organized
   :widths: 28 32 40
   :header-rows: 1
   :class: longtable

   * - Modeling role
     - Typical object family
     - Where it appears
   * - fit or likelihood
     - elementwise losses
     - scalar objective after aggregation
   * - regularization or structure
     - vector and matrix penalties
     - scalar objective
   * - hard feasibility
     - domain constraints and structural declarations
     - explicit constraints or variable attributes


Three Modeling Roles
--------------------

Most models mix all three roles, but it helps to keep them conceptually separate:

- **fit** measures how well the current variables explain the observations. It is usually built from
  elementwise losses on residuals, margins, or probabilities, then aggregated into one scalar.
- **regularization or soft structure** expresses a preference rather than a rule. It can encourage
  sparsity, small norm, smoothness, or low rank, but the solver may accept some of the unwanted
  behavior if that improves the overall trade-off.
- **hard feasibility** defines the admissible set. Bounds, cone membership, symmetry, nonnegativity,
  and exact conservation laws belong here. Violating one of these conditions makes the point infeasible.


Worked Example: One Model, Three Roles
--------------------------------------

Suppose you want a nonnegative sparse weight vector that fits data robustly and stays within a hard
budget:

.. code-block:: python

    model = admm.Model()
    x = admm.Var("x", n, nonneg=True)
    lam = 0.1
    tau = 3.0

    fit = admm.sum(admm.huber(A @ x - b, 1.0))
    regularization = lam * admm.norm(x, ord=1)

    model.setObjective(fit + regularization)
    model.addConstr(admm.sum(x) <= tau)

Read this model by role:

- ``admm.sum(admm.huber(A @ x - b, 1.0))`` is the fit term: it matches data while softening the effect
  of outliers.
- ``lam * admm.norm(x, ord=1)`` is a soft structural preference: it encourages sparsity, but does not
  require exact zeros.
- ``nonneg=True`` and ``admm.sum(x) <= tau`` are hard feasibility: every accepted solution must stay
  nonnegative and respect the budget.

A useful modeling habit is to ask whether a structure should be soft or hard. Sparsity often starts as an
L1 penalty, while sign, interval, and cone restrictions are usually cleaner as feasibility conditions.


Elementwise Atoms
-----------------

Elementwise atoms are the usual starting point for fit terms. They act entrywise on residuals, scores, or
probabilities and are then aggregated into a scalar objective.
The table examples below show the expression of the atoms.

.. list-table:: Elementwise atoms
   :widths: 56 44
   :header-rows: 1
   :class: longtable

   * - Example
     - Application
   * - :py:func:`admm.abs(r) <abs>`
     - robust fitting
   * - :py:func:`admm.square(r) <square>`
     - least squares regression
   * - :py:func:`admm.huber(r, 1.0) <huber>`
     - outlier-robust regression
   * - :py:func:`admm.logistic(z, 1) <logistic>`
     - sparse logistic regression
   * - :py:func:`admm.squared_hinge(t) <squared_hinge>`
     - large-margin learning
   * - :py:func:`admm.entropy(p) <entropy>`
     - probabilistic modeling
   * - :py:func:`admm.kl_div(p, q) <kl_div>`
     - distribution matching

Wrap with :py:func:`sum` when you need a scalar objective.

.. code-block:: python

    admm.sum(admm.square(A @ x - b))
    admm.sum(admm.huber(A @ x - b, 1.0))
    admm.sum(admm.logistic(-y * (X @ w + v), 1))  # logistic(z, 1) = log(exp(z) + 1)
    admm.sum(admm.kl_div(p, q))


Vector and Matrix Atoms
-----------------------

These atoms act on whole vectors or matrices and usually define the regularization or soft-structure part
of the objective.

Typical examples include :math:`\|x\|_1`, :math:`\|x\|_2^2`,
:math:`\|X\|_*`, and :math:`-\log\det(X)`.

.. list-table:: Vector and matrix atoms
   :widths: 56 44
   :header-rows: 1
   :class: longtable

   * - Example
     - Application
   * - ``lam * admm.norm(x, ord=1)``
     - l1 regularization
   * - ``lam * admm.sum(admm.square(x))``
     - ridge regularization
   * - ``admm.norm(A @ X - B, ord="fro")``
     - matrix fitting
   * - ``admm.norm(L, ord="nuc")``
     - low-rank matrix completion
   * - ``admm.trace(S @ X)``
     - covariance matrix estimation
   * - ``-admm.log_det(X) + admm.trace(S @ X)``
     - precision-matrix estimation

Representative examples:

.. code-block:: python

    admm.sum(admm.square(A @ x - b)) + lam * admm.norm(x, ord=1)    # Lasso
    admm.norm(L, ord="nuc") + mu * admm.sum(admm.abs(S))            # Robust PCA
    -admm.log_det(X) + admm.trace(S @ X)                            # Sparse covariance


Structural Variable Attributes and Domain Constraints
-----------------------------------------------------

Use this section when the property must hold exactly rather than be traded off in the objective. Some
structure is best written directly as a feasible set rather than as an atom.
Typical examples are sign restrictions, symmetry, diagonal structure, and PSD or NSD cones, e.g.
:math:`x \in \mathbb{R}_+^n` or :math:`X \in \mathbb{S}_+^p`.
These are hard constraints.
For example, instead of penalizing violations of :math:`1 \le x + 1 \le 3`, model the interval directly:

.. code-block:: python

    x = admm.Var("x")
    model.addConstr(admm.inrange(x + 1, 1, 3))  # enforce 1 <= x + 1 <= 3

This is zero on the admissible interval and infeasible outside it, so it is a hard constraint.

.. list-table:: Structural variable attributes and domain constraints
   :widths: 24 76
   :header-rows: 1
   :class: longtable

   * - Structure
     - Syntax
   * - nonnegativity
     - ``x = admm.Var("x", n, nonneg=True)`` or ``model.addConstr(x >= 0)``
   * - nonpositivity
     - ``x = admm.Var("x", n, nonpos=True)`` or ``model.addConstr(x <= 0)``
   * - symmetry
     - ``S = admm.Var("S", n, n, symmetric=True)``
   * - diagonal structure
     - ``D = admm.Var("D", n, n, diag=True)``
   * - PSD structure
     - ``X = admm.Var("X", n, n, PSD=True)`` or ``model.addConstr(X >> 0)``
   * - NSD structure
     - ``X = admm.Var("X", n, n, NSD=True)`` or ``model.addConstr(X << 0)``
   * - interval restriction
     - ``admm.inrange(x, lb, ub)``

Representative example:

.. code-block:: python

    model = admm.Model()
    x = admm.Var("x", n, nonneg=True)   # Nonnegative vector
    X = admm.Var("X", n, n, PSD=True)   # PSD matrix variable

    model.addConstr(admm.sum(x) == 1)   # Probability-simplex constraint
    model.addConstr(admm.trace(X) == 1) # Normalize matrix scale


Compact Atom Reference
----------------------

The tables below provide a compact lookup view when you want to check the output type,
the typical usage syntax, and the corresponding mathematical expression of the atoms.

.. rubric:: Elementwise Atoms

.. list-table:: Elementwise atoms
   :widths: 40 18 42
   :header-rows: 1
   :class: longtable

   * - Example
     - Output
     - Meaning
   * - :py:func:`admm.abs(x) <abs>`
     - same shape
     - :math:`f(x)=\lvert x\rvert`
   * - :py:func:`admm.exp(x) <exp>`
     - same shape
     - :math:`f(x)=e^x`
   * - :py:func:`admm.power(x, p) <power>`
     - same shape
     - :math:`f(x)=x^p`
   * - :py:func:`admm.square(x) <square>`
     - same shape
     - :math:`f(x)=x^2`
   * - :py:func:`admm.sqrt(x) <sqrt>`
     - same shape
     - :math:`f(x)=\sqrt{x}`
   * - :py:func:`admm.log(x) <log>`
     - same shape
     - :math:`f(x)=\log(x)`
   * - :py:func:`admm.logistic(x, b) <logistic>`
     - same shape
     - :math:`f(x,b)=\log(e^x+b)`
   * - :py:func:`admm.huber(x, delta) <huber>`
     - same shape
     - :math:`f(x,d)=\begin{cases}\tfrac12 x^2,&\lvert x\rvert\le d\\d\lvert x\rvert-\tfrac12 d^2,&\text{otherwise}\end{cases}`
   * - :py:func:`admm.squared_hinge(x) <squared_hinge>`
     - same shape
     - :math:`f(x)=\max(1-x,0)^2`
   * - :py:func:`admm.entropy(x) <entropy>`
     - same shape
     - :math:`f(x)=x\log(x)` (Shannon entropy is ``-admm.sum(admm.entropy(x))``)
   * - :py:func:`admm.kl_div(x, y) <kl_div>`
     - same shape
     - :math:`f(x,y)=x\log(x/y)`
   * - :py:func:`admm.maximum(x, y) <maximum>`
     - same shape
     - :math:`f_{\max}(x,y)=\max(x,y)`
   * - :py:func:`admm.minimum(x, y) <minimum>`
     - same shape
     - :math:`f_{\min}(x,y)=\min(x,y)`
   * - :py:func:`admm.inrange(x, lb, ub) <inrange>`
     - same shape
     - :math:`f(x)=\begin{cases}0,&x\in[\mathrm{lb},\mathrm{ub}]\\+\infty,&\text{otherwise}\end{cases}`
   * - :py:func:`admm.bathtub(x, d) <bathtub>`
     - same shape
     - :math:`f(x,d)=\max(\lvert x\rvert-d,0)`
   * - :py:func:`admm.squared_bathtub(x, d) <squared_bathtub>`
     - same shape
     - :math:`f(x,d)=\tfrac12\max(\lvert x\rvert-d,0)^2`
   * - :py:func:`admm.scalene(x, a, b) <scalene>`
     - same shape
     - :math:`f(x,a,b)=a\min(x,0)+b\max(x,0)`

.. rubric:: Aggregation and Norm Atoms

.. list-table:: Aggregation and norm atoms
   :widths: 40 18 42
   :header-rows: 1
   :class: longtable

   * - Example
     - Output
     - Meaning
   * - :py:func:`admm.sum(x) <sum>`
     - scalar
     - :math:`f(x)=\sum_i x_i`
   * - :py:func:`admm.max(x) <max>`
     - scalar
     - :math:`f(x)=\max_i x_i`
   * - :py:func:`admm.min(x) <min>`
     - scalar
     - :math:`f(x)=\min_i x_i`
   * - :py:func:`admm.norm(x, ord=1) <norm>`
     - scalar
     - :math:`f(x)=\lVert x\rVert_1`
   * - :py:func:`admm.norm(X, ord=1) <norm>`
     - scalar
     - :math:`f(X)=\max_j \sum_i \lvert X_{ij}\rvert`
   * - :py:func:`admm.norm(x, ord=2) <norm>`
     - scalar
     - :math:`f(x)=\lVert x\rVert_2`
   * - :py:func:`admm.norm(X, ord=2) <norm>`
     - scalar
     - :math:`f(X)=\lVert X\rVert_2`
   * - :py:func:`admm.norm(x, ord=inf) <norm>`
     - scalar
     - :math:`f(x)=\lVert x\rVert_\infty`
   * - :py:func:`admm.norm(X, ord=inf) <norm>`
     - scalar
     - :math:`f(X)=\max_i \sum_j \lvert X_{ij}\rvert`
   * - :py:func:`admm.norm(X, ord="fro") <norm>`
     - scalar
     - :math:`f(X)=\lVert X\rVert_F`
   * - :py:func:`admm.norm(X, ord="nuc") <norm>`
     - scalar
     - :math:`f(X)=\lVert X\rVert_*`
   * - ``x @ Q @ x``
     - scalar
     - :math:`f(x)=x^\top Qx`
   * - :py:func:`admm.vapnik(x, eps) <vapnik>`
     - scalar
     - :math:`f(x)=\max(\lVert x\rVert_2-\epsilon,0)`

.. note:: When ``ord`` is omitted, ``admm.norm`` returns the L2 norm for vectors and the Frobenius norm
   for matrices.

.. rubric:: Matrix, Cone, and Structure Atoms

.. list-table:: Matrix, cone, and structure atoms
   :widths: 50 18 32
   :header-rows: 1
   :class: longtable

   * - Example
     - Output
     - Meaning
   * - :py:func:`admm.trace(X) <trace>`
     - scalar
     - :math:`f(X)=\operatorname{tr}(X)`
   * - :py:func:`admm.log_det(X) <log_det>`
     - scalar
     - :math:`f(X)=\log(\det(X)),\ X \succ 0`
   * - :py:func:`admm.diag(x) <diag>`
     - vector or matrix
     - :math:`\operatorname{diag}(\cdot)\ \text{extraction/construction}`
   * - :py:class:`admm.Var("X", n, n, PSD=True) <Var>`
     - variable
     - create a PSD matrix variable :math:`X`
   * - ``model.addConstr(X >> 0)``
     - constraint
     - add a PSD cone constraint :math:`X \succeq 0`
   * - ``model.addConstr(admm.norm(x) <= r)``
     - constraint
     - add a L2-ball constraint :math:`\lVert x\rVert_2 \le r`
   * - ``model.addConstr(admm.norm(x) <= s)``
     - constraint
     - add a second-order cone constraint :math:`(x,s)\in\{(x,s): \lVert x\rVert_2 \le s\}`

.. rubric:: TV and Convolution Atoms

.. list-table:: TV and Convolution atoms
   :widths: 38 12 50
   :header-rows: 1
   :class: longtable

   * - Example
     - Output
     - Meaning
   * - :py:func:`admm.tv1d(x) <tv1d>`
     - scalar
     - :math:`f(x)=\sum_i \lvert x_{i+1}-x_i\rvert`
   * - :py:func:`admm.tv2d(X) <tv2d>`
     - scalar
     - :math:`f(X)=\sum\lvert \partial_i X\rvert+\sum\lvert \partial_j X\rvert` for the default case ``p=1``
   * - :py:func:`admm.conv2d(x, k, mode) <conv2d>`
     - array
     - :math:`\text{2D convolution}`
   * - :py:func:`admm.corr2d(x, k, mode) <corr2d>`
     - array
     - :math:`\text{2D cross-correlation}`

For advanced rewrite behavior over these atoms, see
:ref:`Symbolic Canonicalization <user-guide-symbolic-canonicalization>`.
For extensions beyond the built-in catalog, see
:ref:`User-Defined Proximal Extensions <user-guide-udf>`.