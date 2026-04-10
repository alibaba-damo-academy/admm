.. include:: ../definition.hrst

.. _udf-example-lhalf:

L1/2 Quasi-Norm
===============

This example shows how to model the classical nonconvex :math:`L_{1/2}` sparse penalty with a UDF. The full
problem is

.. math::

   \min\limits_x \; \tfrac{1}{2}\|x-y\|_2^2 + \lambda \phi(x),
   \qquad
   \phi(x) = \sum_i \sqrt{|x_i|}.

This is a classical nonconvex sparse penalty. It promotes sparsity more aggressively than :math:`L_1`,
but the price is that the model is nonconvex, so the solver acts as a practical local method and the result
should be interpreted as a locally optimal solution or stationary point.

The value returned by :py:meth:`UDFBase.eval` is simply the penalty itself:

.. math::

   f(x) = \phi(x) = \sum_i \sqrt{|x_i|}.

So ``eval`` sums :math:`\sqrt{|x_i|}` over all coordinates:

.. code-block:: python

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.sum(np.sqrt(np.abs(x))))

The proximal operator returned by :py:meth:`UDFBase.argmin` is separable, so it acts coordinatewise:

.. math::

   \operatorname{prox}_{\lambda \phi}(v)
   =
   \arg\min_x \; \lambda \phi(x) + \tfrac{1}{2}\|x-v\|_2^2,

with

.. math::

   \left(\operatorname{prox}_{\lambda \phi}(v)\right)_i
   =
   \begin{cases}
   0, &
   |v_i| \le \tfrac{3}{2}\lambda^{2/3}, \\[6pt]
   \operatorname{sign}(v_i)\dfrac{2|v_i|}{3}
   \left(
   1 + \cos\!\left(
   \dfrac{2\pi}{3}
   - \dfrac{2}{3}\arccos\!\left(
   \dfrac{3\sqrt{3}\lambda}{4|v_i|^{3/2}}
   \right)
   \right)
   \right), &
   |v_i| > \tfrac{3}{2}\lambda^{2/3}.
   \end{cases}

The code below implements that closed-form rule:

.. code-block:: python

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        abs_v = np.abs(v)
        threshold = 1.5 * (lamb ** (2.0 / 3.0))
        prox = np.zeros_like(v)

        active = abs_v > threshold
        if np.any(active):
            phi = np.arccos(
                np.clip((3.0 * np.sqrt(3.0) * lamb) / (4.0 * np.power(abs_v[active], 1.5)), -1.0, 1.0,)
            )
            prox_abs = (2.0 * abs_v[active] / 3.0) * (
                1.0 + np.cos((2.0 * np.pi / 3.0) - (2.0 * phi / 3.0))
            )
            prox[active] = np.sign(v[active]) * prox_abs

        return [prox.tolist()]

For any proper function :math:`f` and scalar :math:`\alpha > 0`,

.. math::

   \operatorname{prox}_{\lambda (\alpha f)}(v)
   =
   \operatorname{prox}_{(\alpha \lambda) f}(v).

So the mathematically standard modeling form is to keep the UDF as the base function and write the
objective coefficient outside, for example ``lam * LHalfNorm(x)``.

The :py:meth:`UDFBase.arguments` method again just tells |ADMM| which symbolic vector is the input to this UDF:

.. code-block:: python

    def arguments(self):
        return [self.arg]

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    class LHalfNorm(admm.UDFBase):
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            return float(np.sum(np.sqrt(np.abs(x))))

        def argmin(self, lamb, arglist):
            v = np.asarray(arglist[0], dtype=float)
            abs_v = np.abs(v)
            threshold = 1.5 * (lamb ** (2.0 / 3.0))
            prox = np.zeros_like(v)

            active = abs_v > threshold
            if np.any(active):
                phi = np.arccos(
                    np.clip((3.0 * np.sqrt(3.0) * lamb) / (4.0 * np.power(abs_v[active], 1.5)), -1.0, 1.0,)
                )
                prox_abs = (2.0 * abs_v[active] / 3.0) * (
                    1.0 + np.cos((2.0 * np.pi / 3.0) - (2.0 * phi / 3.0))
                )
                prox[active] = np.sign(v[active]) * prox_abs

            return [prox.tolist()]

    y = np.array([0.2, 1.0, 2.0])
    lam = 0.5

    model = admm.Model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + lam * LHalfNorm(x))
    model.optimize()

    print(" * x: ", np.asarray(x.X))  # Expected: ≈ [0, 0.70, 1.81]
    print(" * model.ObjVal: ", model.ObjVal)  # Expected: ≈ 1.17405


This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/main/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/udf_lhalf.py


In this concrete example, :math:`\lambda = 0.5`, so the activation threshold is

.. math::

   \begin{aligned}
   \frac{3}{2}\lambda^{2/3}
   &= \frac{3}{2}(0.5)^{2/3} \\
   &\approx 0.94494.
   \end{aligned}

Therefore :math:`|0.2| < 0.94494`, so the first coordinate is driven to zero, while
:math:`|1.0|` and :math:`|2.0|` lie above the threshold and are shrunk by the closed-form proximal map. The
returned point is approximately

.. math::

   x^\star \approx [0,\; 0.701516,\; 1.814402].

The smallest coordinate is eliminated; the larger coordinates are shrunk (not kept
unchanged) by the :math:`L_{1/2}` proximal map.