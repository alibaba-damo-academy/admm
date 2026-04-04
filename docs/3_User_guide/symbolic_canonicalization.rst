.. include:: ../definition.hrst

.. _user-guide-symbolic-canonicalization:

Symbolic Canonicalization
=========================

.. note::
   This page is advanced and optional. Read it if you want to understand what the solver rewrites for you,
   why several equivalent expressions can map to the same supported atom, or when a manual reformulation
   might help. You do not need to understand canonicalization to use |ADMM| day to day.

Symbolic canonicalization is the preprocessing stage that runs before optimization. You write readable,
math-like Python expressions; |ADMM| then tries to recognize supported patterns and lower them into the
internal form it actually solves.

For most users, the practical lesson is simple:

1. write the clean mathematical expression first
2. let the solver recognize equivalent built-in structure when it can
3. reformulate manually only when the model falls outside the documented support boundary

This mainly helps in four ways:

.. list-table:: Canonicalization behaviors
   :widths: 49 51
   :header-rows: 1
   :class: longtable

   * - Formulation pattern
     - Recognized structure
   * - different equivalent expressions of an atom
     - a norm, loss, divergence, or conic constraint
   * - vectorized residual expressions
     - one fit term instead of many scalar terms
   * - functions with implicit domains like :py:func:`log`, :py:func:`sqrt`
     - the corresponding positivity or nonnegativity restriction
   * - composite expressions
     - an equivalent form with auxiliary variables or indicators

The main user-visible idea is that natural mathematical code is often enough, provided the formulation still
lies inside the documented supported modeling set.

Write Clean Math First
----------------------

When you derive a model on paper, keep that readable form in your code unless you have a concrete reason to
change it. If the natural expression is ``admm.log(1 + admm.exp(z))`` or ``s - admm.norm(u, ord=2) >= 0``,
start there. The solver can often recognize the underlying loss or cone constraint for you.

A good escalation path is:

1. write the direct expression
2. check it against :ref:`Supported Problem Structure <user-guide-supported-problem-structure>` if you are unsure
3. only introduce a manual reformulation, auxiliary variable, or
   :ref:`User-Defined Proximal Extensions <user-guide-udf>` when the direct form is outside that supported set


Walkthrough: Readable Math, Recognized Structure
------------------------------------------------

Suppose your notebook derivation leads you to write:

.. code-block:: python

    loss = admm.log(1 + admm.exp(z))
    residual_size = admm.sqrt(admm.sum(admm.square(r)))
    model.addConstr(s - admm.norm(u, ord=2) >= 0)

These lines read naturally:

- the first says "apply logistic loss to ``z``"
- the second says "take the Euclidean size of the residual ``r``"
- the third says "the slack ``s`` must dominate the L2 norm of ``u``"

Canonicalization tries to recognize those patterns as a logistic loss, an L2 norm, and a cone or ball
constraint, so you do not have to memorize one special syntax for each supported object.

Mathematical Rewrite Examples
-----------------------------

.. list-table:: Representative mathematical rewrites
   :widths: 22 49 29
   :header-rows: 1
   :class: longtable

   * - Math form
     - Python form
     - Recognized structure
   * - :math:`\sum_i |r_i|`
     - ``admm.sum(admm.abs(r))``
     - L1 norm
   * - :math:`|r_1| + \cdots + |r_k|`
     - ``admm.abs(r1) + admm.abs(r2)``
     - concatenated L1 norm
   * - :math:`\sum_i c_i |x_i|,\; c_i \ge 0`
     - ``c.T @ admm.abs(x)``
     - weighted L1 norm
   * - :math:`\sqrt{\sum_i r_i^2}`
     - ``admm.sqrt(admm.sum(admm.square(r)))``
     - L2 norm
   * - :math:`r^\top r`
     - ``r.T @ r``
     - quadratic function
   * - :math:`(ar)^\top (br),\; a,b \ge 0`
     - ``(a * r).T @ (b * r)``
     - scaled quadratic function
   * - :math:`\sum_j \|r_j\|_2^2`
     - ``r1.T @ r1 + r2.T @ r2``
     - concatenated quadratic function
   * - :math:`\sum_i \log(1 + e^{z_i})`
     - ``admm.log(1 + admm.exp(z))``
     - logistic loss function
   * - :math:`\sum_i x_i \log x_i`
     - ``x * admm.log(x)``
     - entropy function
   * - :math:`\sum_i x_i \log(x_i/y_i)`
     - ``x * admm.log(x / y)``
     - KL divergence function
   * - :math:`\sum_i \max(1-z_i,0)^2`
     - ``admm.maximum(1 - z, 0) ** 2``
     - squared hinge loss function
   * - :math:`s - \|u\|_2 \ge 0`
     - ``s - admm.norm(u, ord=2) >= 0``
     - second-order cone constraint
   * - :math:`c - \|u\|_2 \ge 0`
     - ``c - admm.norm(u, ord=2) >= 0``
     - L2-ball constraint

.. note:: In the last two rows, ``c`` is a constant and ``s`` is a variable.

Transform Examples With Auxiliary Variables
-------------------------------------------

Some supported compositions are easier to solve after rewriting them with an auxiliary variable. The key
point for a Python user is that you do not always need to introduce that variable manually. You can think of
the solver as naming an intermediate quantity and then splitting one composite expression into simpler pieces.

Consider the situation where a scalar loss is applied to a norm, i.e. :math:`\phi(\|r(x)\|_2)`.
This can be rewritten as:

.. math::

   \min_{x,\,z}\; \phi(z) \quad \text{s.t.} \quad z \ge \|r(x)\|_2.

In plain language, the solver can introduce a scalar ``z`` that stands for the size of the residual. It then
handles :math:`\phi(z)` and the norm constraint separately, turning the composition into solver-friendly
pieces. Note that :math:`\phi` can be any monotone scalar loss, e.g. :math:`\mathrm{Huber}`.
If a desired construction is still outside the recognized built-in forms, the next step is usually
:ref:`User-Defined Proximal Extensions <user-guide-udf>`.