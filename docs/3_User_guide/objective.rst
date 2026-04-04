.. include:: ../definition.hrst

.. _user-guide-objective:

Objective
=========

Every model has at most one objective, and that objective must evaluate to a scalar.
The model sense is minimization by default.

For a first model, it helps to think of the objective as one final scalar built from three layers:

1. a fit term that measures how well the model matches the data
2. a regularization term that expresses what kind of solution you prefer
3. an optional structural term for extra smoothness, barriers, or domain-specific shaping

.. list-table:: Objective rules at a glance
   :widths: 34 66
   :header-rows: 1
   :class: longtable

   * - Rule
     - What it means in practice
   * - one objective per model
     - combine fit, regularization, and other terms into one final scalar expression
   * - scalar-valued expression
     - aggregate vectorized losses with functions such as :py:func:`sum`
   * - minimization by default
     - write the quantity you want to minimize directly

Build the Objective in Layers
-----------------------------

Start with the simplest fit term that captures the task. Then add regularization only if it helps resolve
ambiguity or improves the kind of solution you want. If the model still needs extra structure, add a third
term and sum everything into one scalar before calling :py:meth:`Model.setObjective`.

.. math::

   \text{data fit} + \text{regularization} + \text{optional structural term}

For example, a common beginner pattern is a least-squares fit with L1 regularization and an optional
smoothness term:

.. code-block:: python

    fit = 0.5 * admm.sum(admm.square(A @ x - b))
    regularization = lam * admm.norm(x, ord=1)
    structure = mu * admm.sum(admm.square(D @ x))

    model.setObjective(fit + regularization + structure)

If you do not need the third term, leave it out. If your expression is already scalar, such as
``admm.norm(A @ x - b, ord=1)``, you can pass it directly to :py:meth:`Model.setObjective`.
For matrix models, the optional structural term might instead be a barrier or spectral expression such as
``-admm.log_det(X) + admm.trace(S @ X)``.

Common Objective Ingredients
----------------------------

Common objective functions are summarized below. In practice, most models choose one fit term from this
table and then add one regularizer before exploring more specialized structure.

.. list-table:: Typical objective functions
   :widths: 40 60
   :header-rows: 1
   :class: longtable

   * - Ingredient family
     - Example
   * - linear objective
     - ``c.T @ x``
   * - least-squares data fit
     - ``0.5 * admm.sum(admm.square(A @ x - b))``
   * - quadratic regularization
     - ``0.5 * admm.sum(admm.square(x))``
   * - sparsity regularization
     - ``lam * admm.norm(x, ord=1)``
   * - robust loss
     - ``admm.sum(admm.huber(A @ x - b))``
   * - classification loss
     - ``admm.sum(admm.logistic(z))``
   * - matrix barrier or spectral term
     - ``-admm.log_det(X) + admm.trace(S @ X)``
   * - user-defined proximal term
     - ``admm.sum(admm.square(x - y)) + lam * L0Norm(x)``

For the full atom catalog, see :ref:`Supported Building Blocks <user-guide-objective-building-blocks>`.
For feasibility conditions that accompany the objective, see :ref:`Constraints <user-guide-constraints>`.