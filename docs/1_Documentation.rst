.. include:: definition.hrst

.. _doc-overview:

Overview
========

|ADMM| is a Python library for building and solving optimization problems.
Users write their models — objectives and constraints — in a natural mathematical style, and |ADMM|
automatically translates the formulation into an efficient numerical solve.

In this documentation, |ADMM| refers to ``ADMM (Automatic Decomposition Method by MindOpt)``, while ``admm`` is the
Python package name. The product name echoes the well-known ADMM algorithm
(Alternating Direction Method of Multipliers), which is also the core algorithm used internally.

The Alternating Direction Method of Multipliers is a numerical method
that breaks a large optimization problem into smaller, easier subproblems and solves them iteratively.
Users do not need to perform this decomposition by hand.
|ADMM| handles the entire pipeline: it takes the user's model, rewrites it into a solver-friendly form
(a step called *canonicalization*), splits it into subproblems, and solves them.

The central idea is: **model the problem, not the solver internals.**
Users should not need to introduce extra variables or write low-level update rules.
|ADMM| preserves the user-facing structure and solves the resulting formulation directly.


.. rst-class:: landing-page-section-heading

.. rubric:: When To Use |ADMM|

|ADMM| is designed for optimization problems that combine several of the following ingredients:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Ingredient
     - Examples
   * - linear or quadratic objectives
     - :math:`c^\top x`, :math:`x^\top Q x`
   * - smooth data-fitting terms
     - least squares, logistic regression, Huber loss
   * - nonsmooth regularization
     - L1 norm (sparsity), nuclear norm (low rank)
   * - structural variable attributes
     - nonnegativity (:math:`x \ge 0`), symmetry, PSD/NSD matrix structure
   * - affine equality and inequality constraints
     - :math:`A x = b`, :math:`G x \le h`, :math:`\mathbf{1}^\top x = 1`, scaled affine forms
   * - norm-ball and cone constraints
     - :math:`\|x\|_2 \le r`, :math:`\|A x - b\|_2 \le s`, :math:`\|X\|_F \le \rho`, :math:`X \succeq 0`
   * - matrix-valued objectives
     - trace, log-determinant, Frobenius norm
   * - custom user-defined terms
     - L0 norm, rank constraint, manifold projection

Typical application areas include:

- **Linear and quadratic programming** — resource allocation, portfolio optimization, planning
- **Sparse and regularized learning** — Lasso, sparse logistic regression, support vector machines
- **Matrix and covariance estimation** — robust PCA, sparse inverse covariance, semidefinite programs
- **Signal and image processing** — deblurring, total-variation denoising, compressed sensing
- **Structured nonconvex models** — exact sparsity (L0), low-rank constraints, manifold feasibility

|ADMM| also supports *nonconvex* models through user-defined proximal extensions (see :py:class:`UDFBase`).
In the nonconvex setting, the solver acts as a practical local method and may converge to a local optimum rather than a global one.


.. rst-class:: landing-page-section-heading

.. rubric:: Key Terminology

The following terms appear throughout this documentation:

.. list-table::
   :widths: 31 69
   :header-rows: 1

   * - Term
     - Meaning
   * - **convex / nonconvex**
     - A *convex* problem has a single global minimum; any local minimum is also global.
       A *nonconvex* problem may have multiple local minima, and solvers typically find one of them.
   * - **positive semidefinite (PSD)**
     - A symmetric matrix :math:`X` is PSD (written :math:`X \succeq 0`) if all its eigenvalues are
       nonnegative. PSD constraints appear in covariance estimation, semidefinite programming, and
       many matrix optimization models.
   * - **proximal operator**
     - A building block for solving optimization subproblems. Given a function :math:`f` and a
       point :math:`v`, the proximal operator finds the point that balances minimizing :math:`f`
       with staying close to :math:`v`. Users do not need to compute proximal operators for built-in
       functions; they only matter when writing custom extensions via :py:class:`UDFBase`.
   * - **canonicalization**
     - The automatic step where |ADMM| rewrites the user's formulation into a simpler, equivalent
       form that the solver backend can handle efficiently.
   * - **decomposition**
     - The strategy of splitting one large problem into smaller subproblems that can be solved
       independently or in sequence. This is how ADMM scales to large formulations.
   * - **atom**
     - A recognized function such as :py:func:`admm.norm(x) <norm>`,
       :py:func:`admm.square(x) <square>`, :py:func:`admm.log_det(X) <log_det>`, or
       :py:func:`admm.huber(x, delta) <huber>`. The library knows how to handle each
       atom efficiently.
   * - **UDF (User-Defined Function)**
     - A custom proximal operator provided by the user via :py:class:`UDFBase`, used when a needed
       function is not available as a built-in atom.


.. rst-class:: landing-page-section-heading

.. rubric:: Supported Problem Structure

|ADMM| is designed for problems that fit the following general pattern:

.. math::

   \begin{array}{ll}
   \min\limits_x & F_0(x) + \sum_{i=1}^m F_i(A_i x + b_i) \\
   \text{s.t.} & Cx = d, \\
               & x \in \mathcal{K},
   \end{array}

where:

- :math:`x` is the decision variable (a scalar, vector, or matrix)
- :math:`F_0, F_1, \ldots, F_m` are objective terms — each is a supported loss function,
  indicator, or user-defined proximal term
- :math:`A_i x + b_i` are linear transformations of the variable
- :math:`Cx = d` represents affine equality constraints
- :math:`\mathcal{K}` is a structural set such as sign restrictions, PSD/NSD cones, or other supported indicator sets

In practical terms, |ADMM| is designed for formulations built from combinations of:

- linear operators and affine constraints
- aggregate constraints such as :math:`\sum_i x_i = 1` or scaled affine relations
- norm-ball and cone constraints such as :math:`\|x\|_2 \le r`, :math:`\|A x - b\|_2 \le s`, and :math:`X \succeq 0`
- quadratic terms
- losses and regularizers with known proximal operators
- structural variable attributes such as nonnegativity, symmetry, and PSD/NSD constraints
- user-defined proximal operators for nonconvex penalties such as L0 norm, rank, and manifold constraints


.. rst-class:: landing-page-section-heading

.. rubric:: What |ADMM| Does Not Cover

The boundary of what |ADMM| can handle is structural: the formulation must be reducible to supported
smooth terms, proximal terms, affine constraints, and structural indicators.

Formulations that fall outside this scope include:

- arbitrary nonlinear expressions that do not decompose into supported atoms
- variable types beyond scalars, vectors, and matrices (higher-dimensional arrays are not supported)
- integer or combinatorial constraints (beyond what can be modeled through UDF projections)

For convex models, |ADMM| targets the global optimum.
For nonconvex UDF-based models, the result should be interpreted as a locally optimal solution or
stationary point.

The library provides two main extension mechanisms:

- **Symbolic canonicalization** rewrites equivalent algebra into cleaner solver forms, recognizes
  supported atoms, and inserts implied domain information automatically.
- **User-defined proximal extensions** (:py:class:`UDFBase`) let users add custom proximal operators
  when a needed penalty or constraint is not available as a built-in atom. This is one of the main
  ways |ADMM| extends beyond the disciplined convex programming scope of tools such as CVXPY.

For details, see :ref:`Support Boundary <user-guide-support-boundary>` in the User Guide.


.. rst-class:: landing-page-section-heading

.. rubric:: Illustrative Example

The following example shows a mean-variance portfolio optimization problem.
An investor wants to allocate a budget across :math:`n` assets to maximize expected return while
controlling risk (variance of returns).

The mathematical formulation is:

.. math::

   \begin{array}{ll}
   \min\limits_w & -\mu^\top w + \gamma \, w^\top \Sigma w \\[4pt]
   \text{s.t.} & \sum_{i=1}^n w_i = 1, \\
               & w_i \ge 0, \quad i = 1, \ldots, n,
   \end{array}

where:

- :math:`w \in \mathbb{R}^n` is the vector of portfolio weights (how much to invest in each asset)
- :math:`\mu \in \mathbb{R}^n` is the expected return of each asset
- :math:`\Sigma \in \mathbb{R}^{n \times n}` is the covariance matrix of asset returns (measures risk)
- :math:`\gamma > 0` is a risk-aversion parameter: larger values penalize risk more heavily
- the constraint :math:`\sum_i w_i = 1` ensures the weights sum to 1 (fully invested)
- the constraint :math:`w_i \ge 0` prevents short-selling (long-only portfolio)

The corresponding |ADMM| code:

.. code-block:: python

    import admm
    import numpy as np

    np.random.seed(1)
    
    n = 20
    mu = np.abs(np.random.randn(n))                   # Expected returns of assets
    Sigma = np.random.randn(n + 3, n)                 # Random factor matrix
    Sigma = Sigma.T @ Sigma + 0.1 * np.eye(n)         # Covariance matrix of asset returns
    gamma = 0.5                                       # Risk-aversion parameter

    model = admm.Model()                              # Create model
    w = admm.Var("w", n)                              # Create variable: Portfolio weights
    model.setObjective(-mu.T @ w + gamma * (w.T @ Sigma @ w)) # Set objective
    model.addConstr(admm.sum(w) == 1)                 # Add constraint: Fully invested
    model.addConstr(w >= 0)                           # Add constraint: Long-only
    model.optimize()                                  # Solve
    print("model.ObjVal: ", model.ObjVal)             # Expected: ≈ -1.08
    print("model.StatusString: ", model.StatusString) # Expected: SOLVE_OPT_SUCCESS



This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/overview_portfolio.py


.. rst-class:: landing-page-section-heading

.. rubric:: About the MindOpt Team

.. raw:: latex

   \enlargethispage{1.5\baselineskip}

|ADMM| is developed by the MindOpt Team at the Decision Intelligence Lab, DAMO Academy, Alibaba Group.
The team focuses on mathematical optimization solvers and tools for large-scale structured problems.

Contributors:

- Liyun Dai — C++ core
- Qiuwei Li — symbolics, examples
- Xingjian Song — proximal operators
- Yuhua Song — packaging
- Kaizhao Sun — Julia pilot
- Feng Wang — interfaces
- Wotao Yin — methodology, team lead

We also gratefully acknowledge Prof. Kun Yuan and his group for their contributions:

- Peijin Li — acceleration
- Mingyu Mo — pathology detection
- Yilong Song — acceleration
- Hao Yuan — examples

.. raw:: latex

   \enlargethispage{6\baselineskip}

.. rst-class:: landing-page-section-heading

.. rubric:: Next Steps

- :ref:`Installation <doc-install>` — confirm platform support and install the ``admm`` package
- :ref:`User Guide <doc-user-guide>` — learn the full modeling workflow
- :ref:`Examples <doc-examples>` — see representative end-to-end formulations
