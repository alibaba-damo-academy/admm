.. include:: ../definition.hrst

.. _user-guide-supported-problem-structure:

Supported Problem Structure
===========================

Before writing code, it helps to answer one question: does your mathematical model
fit the structure that the |ADMM| modeling layer expects?
This page shows how to make that decision from the abstract problem form before you
start translating expressions into Python.

The modeling layer accepts formulations of the form

.. math::

   \begin{array}{ll}
   \min\limits_x & F_0(x) + \sum_{i=1}^m F_i(A_i x + b_i) \\
   \text{s.t.} & Cx = d, \\
               & x \in \mathcal{K},
   \end{array}

Read the template from left to right:

- :math:`x` collects the decision variables the solver is allowed to change.
- :math:`F_0(x)` is the base objective term written directly in those variables.
- Each :math:`F_i(A_i x + b_i)` is a supported loss, penalty, indicator, or user-defined
  proximal term applied after an affine map.
- :math:`Cx = d` captures explicit affine coupling constraints.
- :math:`x \in \mathcal{K}` captures structural or conic sets such as nonnegativity,
  symmetry, or PSD cones.

If you can identify your formulation in these pieces, your model is usually a natural
fit for |ADMM|. The next step is to sort each part of the formulation into the same
groups that appear in the API.

.. list-table:: Structural view of the modeling interface
   :widths: 28 72
   :header-rows: 1
   :class: longtable

   * - Modeling layer
     - Typical objects
   * - objective terms
     - affine expressions, quadratic forms, proximable losses, regularizers
   * - linear maps
     - matrix-vector products, reshaping, convolution-like operators
   * - hard constraints
     - equalities, inequalities, norm constraints, PSD constraints
   * - variable structure
     - nonnegative, symmetric, diagonal, PSD, NSD

Map This Abstract Form to Code
------------------------------

Once you recognize the abstract pieces, you can usually map them into code with the
same left-to-right logic:

.. code-block:: python

    x = admm.Var("x", 20)              # Vector decision variable
    X = admm.Var("X", 5, 5)            # Matrix decision variable
    lam = admm.Param("lam")            # Solve-time regularization weight

    model = admm.Model()               # Create an empty optimization model
    model.setObjective(admm.sum(admm.square(A @ x - b)) + lam * admm.norm(x, ord=1))  # Fit + sparsity
    model.addConstr(C @ x == d)        # Affine coupling constraint
    model.addConstr(X >> 0)            # PSD cone constraint

In this sketch:

- ``x`` and ``X`` are pieces of the abstract variable :math:`x`
- the objective collects supported terms such as least squares and L1 regularization
- ``C @ x == d`` plays the role of :math:`Cx = d`
- ``X >> 0`` enforces membership in a structured set :math:`\mathcal{K}`

If a property is intrinsic to one variable, it is often cleaner to declare it on the
variable itself. If it is a relationship between expressions, write it as an explicit
constraint.


.. _user-guide-support-boundary:

Support Boundary
----------------

A formulation is usually within scope if it can be reduced to supported smooth terms,
proximal terms, affine couplings, and structured indicators, whether convex or not.
When you are unsure, walk through the checklist below in order. If most rows are an
easy "yes", the model is likely inside the support boundary. If several rows are unclear,
reformulate the math before worrying about solver options.

.. list-table:: Support-boundary checklist
   :widths: 34 66
   :header-rows: 1
   :class: longtable

   * - Checklist item
     - Recommended structure
   * - Is the objective scalar?
     - yes, built from one final scalar expression
   * - Are the terms supported?
     - quadratic, proximable, affine-composed, or structurally recognized
   * - Are the constraints reducible?
     - affine, conic, or structural in a form the library can lower
   * - Is the decomposition visible?
     - separable terms and coupling operators are naturally exposed

For nonconvex UDF-based models, the solver acts as a practical local method: the goal
is a locally optimal solution or stationary point, not a certificate of global optimality.


When |ADMM| Is a Strong Fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table below summarizes the patterns where the library is typically most effective.
These are the cases where the decomposition is already doing most of the hard work for you.

.. list-table:: When |ADMM| is a strong fit
   :widths: 38 62
   :header-rows: 1
   :class: longtable

   * - Formulation feature
     - Why |ADMM| handles it well
   * - smooth loss plus nonsmooth regularization
     - naturally separates structure terms into solver-friendly subproblems
   * - sparsity, rank, or manifold constraints
     - enforceable through custom proximal steps even outside DCP rules
   * - matrix structure or cone geometry
     - maps cleanly to PSD, trace, log-determinant, or norm-based operators
   * - large vectorized linear maps
     - preserves operator structure without manual reformulation

If the formulation does not fit cleanly, the usual next step is to expose the missing
affine decomposition, introduce auxiliary variables, or replace an unsupported term with
a supported proximal representation before expecting reliable results.
For the concrete atom families behind these patterns, see
:ref:`Supported Building Blocks <user-guide-objective-building-blocks>`.
If a needed term is not built in but still has a useful proximal step, see
:ref:`User-Defined Proximal Extensions <user-guide-udf>`.