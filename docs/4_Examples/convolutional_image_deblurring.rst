.. include:: ../definition.hrst

.. _example-convolutional-image-deblurring:

Convolutional Image Deblurring
==============================

Convolutional image deblurring is an inverse problem. We do not observe the
sharp image directly. Instead, we observe a blurred image and try to infer what
sharp image could have produced it.

The optimization model is

.. math::

   \begin{array}{ll}
   \min\limits_U & \lambda \, \mathrm{TV}(U) + \frac{1}{2}\|K * U - B\|_F^2.
   \end{array}

Each symbol has a specific meaning:

- :math:`U \in \mathbb{R}^{m \times n}` is the unknown sharp image we want to recover.
- :math:`K` is the blur kernel.
- :math:`B` is the observed blurred image.
- :math:`K * U` is the image we would observe if :math:`U` were blurred by the
  kernel.
- :math:`\mathrm{TV}(U)` is a total-variation penalty that discourages noisy
  pixel-to-pixel oscillation while still allowing edges.

This is a good tutorial example because it combines two modeling ideas at once:

- :py:func:`conv2d` expresses the forward physical measurement model.
- :py:func:`tv2d` expresses the regularization term that makes the inverse problem
  well behaved.

The blur kernel in this example is a small Gaussian-like averaging stencil:

.. math::

   K =
   \begin{bmatrix}
   1/16 & 2/16 & 1/16 \\
   2/16 & 4/16 & 2/16 \\
   1/16 & 2/16 & 1/16
   \end{bmatrix}.

Its entries are all nonnegative and sum to 1, so it acts like a local weighted
average. The center pixel gets the largest weight, the immediate neighbors get
smaller weights, and the corners get the smallest weights. Applying this kernel
spreads nearby intensity values together, which is exactly what blur means.

**Step 1: Create a synthetic image, a blur kernel, and blurred observations**

We first generate a reproducible piecewise-constant test image and then blur it.
The line ``image_blur = admm.conv2d(image, kernel, "same")`` is how we synthesize
the observed data :math:`B`.

The argument ``"same"`` means the output has the same shape as the input image,
so the blurred image can be compared pixel-by-pixel against any candidate image
after it is blurred in the same way.

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    height = 40
    width = 50
    # Piecewise-constant synthetic image (blocks) — TV regularization is effective here
    image = np.zeros((height, width))
    image[:20, :25] = 0.8
    image[20:, 25:] = 0.6
    image[10:30, 10:40] = 1.0
    image += 0.02 * np.random.randn(height, width)  # slight noise
    kernel = np.array([
        [1 / 16, 2 / 16, 1 / 16],
        [2 / 16, 4 / 16, 2 / 16],
        [1 / 16, 2 / 16, 1 / 16],
    ])
    image_blur = admm.conv2d(image, kernel, "same")
    lam = 0.1

**Step 2: Create the model and the unknown image variable**

The decision variable ``U`` has the same two-dimensional shape as the image.
Each entry of ``U`` represents one pixel value in the reconstructed sharp image.

.. code-block:: python

    model = admm.Model()
    U = admm.Var("U", image.shape)

**Step 3: Build the residual term and the TV regularization term**

The expression ``admm.conv2d(U, kernel, "same")`` is the forward model. It asks:
"If ``U`` were the sharp image, what blurred image would this kernel produce?"
That is why it is the correct model of the measurement process.

We then subtract the actual observed blurred image ``image_blur`` to form the
residual. If that residual is small, the candidate image ``U`` is consistent
with the measured data.

The function ``admm.tv2d(U, p=1)`` adds total-variation regularization. TV does
not try to make the image globally flat. Instead, it penalizes the total amount
of local change across the grid. That makes it useful for deblurring: smooth
regions stay smooth, but important edges can remain sharp instead of being
washed out by a purely quadratic penalty.

.. code-block:: python

    tv = admm.tv2d(U, p=1)
    residual = admm.conv2d(U, kernel, "same") - image_blur
    model.setObjective(lam * tv + 0.5 * admm.sum(admm.square(residual)))

The two objective pieces play different roles:

- ``0.5 * admm.sum(admm.square(residual))`` is the data-fidelity term. It asks
  the blurred version of ``U`` to match the observed image.
- ``lam * tv`` is the regularization term. It stabilizes the inverse problem and
  discourages noisy solutions that fit the blur observation too literally.

Without the residual term, the model would ignore the data. Without the TV term,
many noisy images could explain the same blurred observation almost equally well.
The parameter ``lam`` controls the balance between those two pressures.

**Step 4: Add constraints**

This deblurring model has no explicit constraints, so there are no
``model.addConstr(...)`` calls. The structure comes entirely from the objective:
match the observed blur while keeping the reconstructed image spatially simple in
the TV sense.

**Step 5: Solve and inspect the result**

After optimization, ``model.ObjVal`` reports the best compromise between data
fit and TV regularity, and ``model.StatusString`` reports whether the solver
finished successfully.

.. code-block:: python

    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 9.210929389075202
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS

Complete runnable example:

.. code-block:: python

    import numpy as np
    import admm

    np.random.seed(1)

    height = 40
    width = 50
    # Piecewise-constant synthetic image (blocks) — TV regularization is effective here
    image = np.zeros((height, width))
    image[:20, :25] = 0.8
    image[20:, 25:] = 0.6
    image[10:30, 10:40] = 1.0
    image += 0.02 * np.random.randn(height, width)  # slight noise
    kernel = np.array([
        [1 / 16, 2 / 16, 1 / 16],
        [2 / 16, 4 / 16, 2 / 16],
        [1 / 16, 2 / 16, 1 / 16],
    ])
    image_blur = admm.conv2d(image, kernel, "same")

    lam = 0.1
    model = admm.Model()
    U = admm.Var("U", image.shape)
    tv = admm.tv2d(U, p=1)
    residual = admm.conv2d(U, kernel, "same") - image_blur
    model.setObjective(lam * tv + 0.5 * admm.sum(admm.square(residual)))
    model.optimize()

    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 9.210929389075202
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS


The recovered :math:`U` minimizes :math:`\|K * U - B\|_F^2 + \lambda\,\mathrm{TV}(U)`:
a data-fidelity term plus total-variation regularization for edge-preserving smoothness.
ADMM decomposes this into a least-squares update and a TV proximal step automatically.

On this piecewise-constant test image, the recovery error drops substantially compared to
the blurred observation:

.. list-table:: Recovery quality
   :widths: 50 25
   :header-rows: 1

   * - Metric
     - Value
   * - :math:`\|B - I_{\text{orig}}\|_F` (blurred)
     - 3.59
   * - :math:`\|U^\star - I_{\text{orig}}\|_F` (recovered)
     - 1.08
   * - Error reduction
     - 70%

This example is available as a standalone script in the `examples/ <https://github.com/alibaba-damo-academy/admm/tree/master/examples>`_ folder of the `ADMM repository <https://github.com/alibaba-damo-academy/admm>`_:

.. code-block:: bash

   python examples/image_deblurring.py