Global functions
================

ADMM Python SDK global functions.

Global functions can be applied to expression or constant.

For example:

.. code-block:: py

    >>> print(inrange(numpy.arange(9).reshape(3, 3), 3, 5))
    [[ 0.  0.  0.]
    [inf inf inf]
    [ 0.  0.  0.]]
    >>> print(inrange(Constant(range(9)).reshape(3, 3), 3, 5))
    [[ 0.  0.  0.]
    [inf inf inf]
    [ 0.  0.  0.]]
    >>> print(inrange(Var(3, 3), 3, 5))
    Expr((3, 3), inrange('Var', 'ndarray', 'ndarray'))

For constant input argument, global function returns constant, otherwise it returns expression.

Besides modeling, these functions can also be used to do matrix operations. In this case, it is an alternative to numpy (obviously with less functionality).

The following example demonstrates how to do edge detection for an input image with global functions.

.. code-block:: py

    # Load/save image with python Pillow
    image = Constant(Image.open(sys.argv[1]).convert('L'))
    sobel = Constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    x = corr2d(image, sobel, 'same')
    y = corr2d(image, sobel.T, 'same')

    result = sqrt(square(x) + square(y))

    # Convert Constant to numpy uint8 array
    resimg = Image.fromarray(result.asDense().data.astype('B'))
    resimg.save("_{}".format(sys.argv[1]))

Functions
---------

The following table lists all available global functions:

.. list-table::
    :widths: 15 30

    * - :py:func:`abs`
      - Get the absolute value for an expression.
    * - :py:func:`exp`
      - Return e ^ x.
    * - :py:func:`log`
      - Return log(x).
    * - :py:func:`entropy`
      - Return a tensor, each element indicates the result of x * log(x).
    * - :py:func:`norm`
      - Compute the norm for an expression or constant.
    * - :py:func:`diag`
      - For matrix, retrieve the elements in main diagonal as a vector.
    * - :py:func:`square`
      - Compute x ^ 2 elementwise.
    * - :py:func:`sqrt`
      - Compute x ^ 1/2 elementwise.
    * - :py:func:`log_det`
      - Return a scalar which indicates the log-determinant result of a matrix.
    * - :py:func:`trace`
      - Return a scalar which indicates the sum of the diagonal entries of a matrix.
    * - :py:func:`max`
      - Return the maximum element.
    * - :py:func:`min`
      - Return the minimum element.
    * - :py:func:`sum`
      - Return the sum of all elements.
    * - :py:func:`tv1d`
      - 1D total variation.
    * - :py:func:`tv2d`
      - 2D total variation.
    * - :py:func:`maximum`
      - Element-wise maximum.
    * - :py:func:`minimum`
      - Element-wise minimum.
    * - :py:func:`power`
      - Element-wise power.
    * - :py:func:`logistic`
      - Logistic function.
    * - :py:func:`huber`
      - Huber function.
    * - :py:func:`bathtub`
      - Bathtub function.
    * - :py:func:`squared_bathtub`
      - Squared bathtub function.
    * - :py:func:`kl_div`
      - KL divergence.
    * - :py:func:`conv2d`
      - 2D convolution.
    * - :py:func:`corr2d`
      - 2D correlation.
    * - :py:func:`inrange`
      - Check if elements are in range.
    * - :py:func:`hstack`
      - Stack expressions horizontally (column wise).
    * - :py:func:`vstack`
      - Stack expressions vertically (row wise).
    * - :py:func:`scalene`
      - Asymmetric linear function (different slopes for positive and negative).
    * - :py:func:`vapnik`
      - Vapnik loss: max(norm(x, 2) - epsilon, 0).
    * - :py:func:`squared_hinge`
      - Squared hinge loss: max(1 - x, 0)^2.