.. include:: ../definition.hrst

.. _udf-example-binary:

The Binary Indicator
====================

This example shows how to encode binary-valued decisions with an indicator UDF. The model is

.. math::

   \min\limits_x \; \tfrac{1}{2}\|x-y\|_2^2 + \delta_{\{0,1\}^n}(x),

where :math:`\delta_{\{0,1\}^n}` is the indicator of the binary cube.

This is a discrete nonconvex constraint. The solver acts as a practical local method and does not
guarantee that the output is a globally optimal binary solution.

The value returned by :py:meth:`UDFBase.eval` is the indicator of the binary cube:

.. math::

   f(x) = \delta_{\{0,1\}^n}(x)
   =
   \begin{cases}
      0, & x_i \in \{0,1\} \text{ for all } i, \\
      +\infty, & \text{otherwise}.
   \end{cases}

So ``eval`` only checks whether each coordinate is numerically close to :math:`0` or :math:`1`:

.. code-block:: python

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        is_binary = np.logical_or(np.abs(x) <= 1e-9, np.abs(x - 1.0) <= 1e-9)
        return 0.0 if np.all(is_binary) else float("inf")

The proximal operator returned by :py:meth:`UDFBase.argmin` acts coordinatewise by projection onto
:math:`\{0,1\}`:

.. math::

   \bigl(\operatorname{prox}_{\delta_{\{0,1\}^n}}(v)\bigr)_i
   =
   \begin{cases}
   0, & v_i < \tfrac{1}{2}, \\
   1, & v_i \ge \tfrac{1}{2}.
   \end{cases}

So the code is just a threshold at :math:`0.5`:

.. code-block:: python

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        prox = np.where(v >= 0.5, 1.0, 0.0)
        return [prox.tolist()]

The :py:meth:`UDFBase.arguments` method tells |ADMM| that this indicator depends on one symbolic vector:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    class BinaryIndicator(admm.UDFBase):
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            is_binary = np.logical_or(np.abs(x) <= 1e-9, np.abs(x - 1.0) <= 1e-9)
            return 0.0 if np.all(is_binary) else float("inf")

        def argmin(self, lamb, arglist):
            v = np.asarray(arglist[0], dtype=float)
            prox = np.where(v >= 0.5, 1.0, 0.0)
            return [prox.tolist()]

    y = np.array([0.2, 0.8, 1.4, -0.3])

    model = admm.Model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + BinaryIndicator(x))
    model.optimize()

    print(" * x: ", np.asarray(x.X))  # Expected: ≈ [0, 1, 1, 0]
    print(" * model.ObjVal: ", model.ObjVal)  # Expected: ≈ 0.165


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_binary.py


In this concrete example, the threshold rule applies coordinatewise:

.. math::

   0.2 \mapsto 0,
   \qquad
   0.8 \mapsto 1,
   \qquad
   1.4 \mapsto 1,
   \qquad
   -0.3 \mapsto 0.

Hence the projection onto the binary cube is

.. math::

   x^\star = [0,\; 1,\; 1,\; 0].

Because :math:`x^\star \in \{0,1\}^4`, the indicator term vanishes:

.. math::

   \delta_{\{0,1\}^4}(x^\star) = 0.

Each coordinate is independently rounded onto :math:`\{0,1\}`.