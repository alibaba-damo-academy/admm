.. include:: ../definition.hrst

.. _udf-example-unit-sphere:

The Unit-Sphere Indicator
=========================

This example shows how to encode a fixed Euclidean norm with an indicator UDF. The model is

.. math::

   \min\limits_x \; \tfrac{1}{2}\|x-y\|_2^2 + \delta_{\{ \|x\|_2 = 1 \}}(x),

where :math:`\delta_{\{\|x\|_2 = 1\}}` is the indicator function of the unit sphere:

.. math::

   \delta_{\{ \|x\|_2 = 1 \}}(x)
   =
   \begin{cases}
   0, & \|x\|_2 = 1, \\
   +\infty, & \text{otherwise}.
   \end{cases}

Minimizing this model projects :math:`y` onto the unit sphere, so it can be used to obtain a unit-norm solution.
Because the sphere is nonconvex, the solver acts as a practical local method and the result should be
interpreted as a locally optimal solution or stationary point.

The value returned by :py:meth:`UDFBase.eval` is the indicator itself:

.. math::

   f(x) = \delta_{\{\|x\|_2 = 1\}}(x).

So ``eval`` only checks whether the current vector has unit norm:

.. code-block:: python

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        norm = np.linalg.norm(x)
        return 0.0 if abs(norm - 1.0) <= 1e-9 else float("inf")

For :math:`v \neq 0`, the proximal step returned by :py:meth:`UDFBase.argmin` is projection onto the sphere:

.. math::

   \operatorname{prox}_{\delta_{\{ \|x\|_2 = 1 \}}}(v)
   =
   \frac{v}{\|v\|_2}.

The implementation also handles the degenerate case :math:`v = 0` by returning a fixed unit vector:

.. code-block:: python

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        norm = np.linalg.norm(v)
        if norm <= 1e-12:
            prox = np.zeros_like(v)
            prox[0] = 1.0
            return [prox.tolist()]
        return [(v / norm).tolist()]

The :py:meth:`UDFBase.arguments` method says this indicator depends on one symbolic vector:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    class UnitSphereIndicator(admm.UDFBase):
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            norm = np.linalg.norm(x)
            return 0.0 if abs(norm - 1.0) <= 1e-9 else float("inf")

        def argmin(self, lamb, arglist):
            v = np.asarray(arglist[0], dtype=float)
            norm = np.linalg.norm(v)
            if norm <= 1e-12:
                prox = np.zeros_like(v)
                prox[0] = 1.0
                return [prox.tolist()]
            return [(v / norm).tolist()]

    y = np.array([0.1, 0.0])

    model = admm.Model()
    x = admm.Var("x", 2)
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + UnitSphereIndicator(x))
    model.optimize()

    print(" * x: ", np.asarray(x.X))  # Expected: ≈ [1, 0]
    print(" * model.ObjVal: ", model.ObjVal)  # Expected: ≈ 0.405


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_unit_sphere.py


In this concrete example,

.. math::

   \begin{aligned}
   \|y\|_2 &= \sqrt{0.1^2 + 0^2} \\
   &= 0.1.
   \end{aligned}

Since :math:`y \neq 0`, projection onto the unit sphere simply normalizes the vector:

.. math::

   \begin{aligned}
   x^\star &= \frac{y}{\|y\|_2} \\
   &= \frac{[0.1,\;0]}{0.1} \\
   &= [1,\;0].
   \end{aligned}

The point :math:`x^\star` lies on the sphere, so the indicator term is zero:

.. math::

   \delta_{\{\|x\|_2 = 1\}}(x^\star) = 0.

The main point of the example is geometric: the vector is simply normalized onto the sphere. The printed
``model.ObjVal`` reports the final objective value directly.