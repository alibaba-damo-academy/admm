.. include:: ../definition.hrst

.. _udf-example-l0:

L0 Norm
=======

This example shows how to package the nonconvex sparsity penalty :math:`\|x\|_0` as a UDF. The full model is

.. math::

   \begin{array}{ll}
   \min\limits_x & \tfrac{1}{2}\|x-y\|_2^2 + \lambda \|x\|_0 \\
   \text{s.t.} & 0 \le x \le 1.
   \end{array}

Here :math:`\|x\|_0` counts the number of nonzero entries of :math:`x`, so the objective trades off fidelity to
the observed vector :math:`y` against sparsity. The :math:`L_0` term is nonconvex, so the solver acts as a
practical local method and the returned point should be interpreted as a locally optimal solution or stationary
point.

The function value returned by :py:meth:`UDFBase.eval` is exactly the :math:`L_0` count:

.. math::

   f(x) = \|x\|_0 = \#\{i : x_i \neq 0\}.

So ``eval`` just counts how many entries are numerically nonzero:

.. code-block:: python

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.count_nonzero(np.abs(x) > 1e-12))

The proximal step returned by :py:meth:`UDFBase.argmin` solves

.. math::

   \operatorname{prox}_{\lambda \|\cdot\|_0}(v)
   = \operatorname*{argmin}_x \; \lambda \|x\|_0 + \tfrac{1}{2}\|x-v\|_2^2,
   \qquad
   x_i^\star =
   \begin{cases}
      0, & |v_i| \le \sqrt{2\lambda}, \\
      v_i, & |v_i| > \sqrt{2\lambda}.
   \end{cases}

This is the classical hard-thresholding rule, so the implementation is short:

.. code-block:: python

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        threshold = np.sqrt(2.0 * lamb)
        prox = np.where(np.abs(v) <= threshold, 0.0, v)
        return [prox.tolist()]

The :py:meth:`UDFBase.arguments` method tells |ADMM| which symbolic object this UDF depends on. In this case
the custom term is a function of one vector variable :math:`x`, so ``arguments`` returns a one-element list:

.. code-block:: python

    def arguments(self):
        return [self.arg]

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

    y = np.array([0.2, 1.7, 0.6, 1.9])
    lam = 1.0

    model = admm.Model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + lam * L0Norm(x))
    model.addConstr(x >= 0)
    model.addConstr(x <= 1)
    model.setOption(admm.Options.admm_max_iteration, 10000)  # Give the constrained nonconvex solve enough iterations
    model.optimize()

    print(" * x: ", np.asarray(x.X))  # Expected: ≈ [0, 1, 0, 1]
    print(" * model.ObjVal: ", model.ObjVal)  # Expected: ≈ 2.85


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_l0_norm.py


In this concrete example, :math:`y = [0.2, 1.7, 0.6, 1.9]` and :math:`\lambda = 1`, so the threshold is

.. math::

   \sqrt{2\lambda} = \sqrt{2} \approx 1.414.

The entries :math:`0.2` and :math:`0.6` fall below that threshold and are driven to zero. The entries
:math:`1.7` and :math:`1.9` survive the hard-thresholding step, but they do not yet satisfy the box
constraint. The active upper bound :math:`x \le 1` clips them, so the final feasible point is

.. math::

   [0,\; 1.7,\; 0,\; 1.9]
   \xrightarrow{\;x \le 1\;}
   x^\star \approx [0,\; 1,\; 0,\; 1].

Geometrically: small entries are hard-thresholded to zero, surviving large entries are
clipped by the active box constraint :math:`x \le 1`.