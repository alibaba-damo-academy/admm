.. include:: ../definition.hrst

.. _doc-examples:

Examples
========

Use the table below to find the example that best matches your formulation.

.. list-table:: Model pattern lookup
   :widths: 73 27
   :header-rows: 1
   :class: longtable

   * - If your formulation contains
     - Start with
   * - affine objective with linear constraints
     - :ref:`Linear Program <example-linear-program>`
   * - convex quadratic objective with affine constraints
     - :ref:`Quadratic Program <example-quadratic-program>`
   * - PSD matrix variable or semidefinite constraint
     - :ref:`Semidefinite Program <example-semidefinite-program>`
   * - norm constraint such as ``admm.norm(A @ x + b, ord=2) <= c.T @ x + d``
     - :ref:`Second-Order Cone Program <example-second-order-cone-program>`
   * - pure data fitting with squared residuals
     - :ref:`Least Squares <example-least-squares>`
   * - squared residuals plus quadratic shrinkage
     - :ref:`Ridge Regression <example-ridge-regression>`
   * - robust regression with a smooth outlier-resistant loss
     - :ref:`Huber Regression <example-huber-regression>`
   * - smooth loss plus sparsity regularization
     - :ref:`Sparse Logistic Regression <example-sparse-logistic-regression>`
   * - hinge-loss classification with sparse coefficients
     - :ref:`SVM (L1) <example-svm-with-l1-regularization>`
   * - robust regression with asymmetric residual treatment
     - :ref:`Quantile Regression <example-quantile-regression>`
   * - low-rank plus sparse matrix decomposition
     - :ref:`Robust PCA <example-robust-pca>`
   * - PSD matrix model with :py:func:`log_det` and sparsity
     - :ref:`Sparse Inverse Covariance <example-sparse-inverse-covariance-selection>`
   * - maximum-entropy distribution under affine constraints
     - :ref:`Entropy Maximization <example-entropy-maximization>`
   * - sparse event recovery with box-constrained coefficients
     - :ref:`Fault Detection <example-fault-detection>`
   * - logarithmic utility with a single resource budget
     - :ref:`Water Filling <example-water-filling>`
   * - blurred observation with convolution and edge-preserving regularization
     - :ref:`Image Deblurring <example-convolutional-image-deblurring>`
   * - quadratic formulation with budget and nonnegativity constraints
     - :ref:`Portfolio Optimization <example-portfolio-optimization>`
   * - exact sparsity penalty through a custom proximal operator
     - :ref:`L0 Norm <udf-example-l0>`
   * - cardinality budget enforced as a hard sparse-set projection
     - :ref:`L0 Ball Indicator <udf-example-l0-ball>`
   * - stronger-than-L1 nonconvex sparsity regularization
     - :ref:`L1/2 Quasi-Norm <udf-example-lhalf>`
   * - exact group sparsity applied block by block
     - :ref:`Group Sparsity <udf-example-group-sparsity>`
   * - explicit low-rank promotion with a custom singular-value thresholding step
     - :ref:`Matrix Rank Function <udf-example-rank>`
   * - hard rank cap enforced by truncated SVD projection
     - :ref:`Rank-r Indicator <udf-example-rank-r>`
   * - fixed-norm vector structure enforced on the unit sphere
     - :ref:`The Unit-Sphere Indicator <udf-example-unit-sphere>`
   * - orthonormal-column matrix structure enforced by a Stiefel projection
     - :ref:`The Stiefel-Manifold Indicator <udf-example-stiefel>`
   * - simplex feasibility modeled through a custom projection operator
     - :ref:`The Simplex Indicator <udf-example-simplex>`
   * - binary-valued decision vector modeled through a custom projection
     - :ref:`The Binary Indicator <udf-example-binary>`
   * - sparse regression with L0 penalty and linear constraints (UDF + constraints)
     - :ref:`L0-Regularized Regression <udf-example-l0-regression>`

.. _examples-core-convex-forms:

.. rst-class:: landing-page-section-heading

.. rubric:: Core Convex Forms

These examples establish the basic affine, quadratic, and conic formulations used throughout the examples
below.
Use this group when you want the cleanest entry points for standard convex templates before moving to more
application-shaped formulations.

.. list-table:: Examples in this group
   :widths: 34 66
   :header-rows: 1

   * - Example
     - Main structure
   * - :ref:`Linear Program <example-linear-program>`
     - affine objective, linear inequalities, and nonnegativity constraints
   * - :ref:`Quadratic Program <example-quadratic-program>`
     - convex quadratic objective with affine equalities and inequalities
   * - :ref:`Semidefinite Program <example-semidefinite-program>`
     - PSD matrix variable with affine trace constraints
   * - :ref:`Second-Order Cone Program <example-second-order-cone-program>`
     - Euclidean norm constraints coupled with affine structure

.. toctree::
   :maxdepth: 1

   linear_program
   quadratic_program
   semidefinite_program
   second_order_cone_program

.. _examples-data-fitting:

.. rst-class:: landing-page-section-heading

.. rubric:: Data Fitting

These examples focus on regression and classification formulations that arise in statistical modeling and
machine learning.
Use this group when your objective is a sum of residuals or losses, possibly with regularization.

.. list-table:: Examples in this group
   :widths: 34 66
   :header-rows: 1

   * - Example
     - Main structure
   * - :ref:`Least Squares <example-least-squares>`
     - unconstrained minimization of squared residuals
   * - :ref:`Ridge Regression <example-ridge-regression>`
     - least squares with L2 shrinkage
   * - :ref:`Huber Regression <example-huber-regression>`
     - robust regression with Huber loss
   * - :ref:`Sparse Logistic Regression <example-sparse-logistic-regression>`
     - logistic loss with L1 regularization
   * - :ref:`SVM (L1) <example-svm-with-l1-regularization>`
     - hinge loss with sparse coefficients
   * - :ref:`Quantile Regression <example-quantile-regression>`
     - asymmetric absolute loss for quantile estimation

.. toctree::
   :maxdepth: 1

   least_squares
   ridge_regression
   huber_regression
   sparse_logistic_regression
   svm_with_l1_regularization
   quantile_regression

.. _examples-structured-matrix:

.. rst-class:: landing-page-section-heading

.. rubric:: Structured Matrix Problems

These examples involve matrix-valued decision variables with structural constraints such as low rank,
sparsity, or positivity.
Use this group when the decision variable is naturally a matrix and the formulation involves spectral or
entrywise structure.

.. list-table:: Examples in this group
   :widths: 34 66
   :header-rows: 1

   * - Example
     - Main structure
   * - :ref:`Robust PCA <example-robust-pca>`
     - low-rank plus sparse matrix decomposition
   * - :ref:`Sparse Inverse Covariance <example-sparse-inverse-covariance-selection>`
     - sparse inverse covariance estimation with log-determinant

.. toctree::
   :maxdepth: 1

   robust_pca
   sparse_inverse_covariance_selection

.. _examples-application:

.. rst-class:: landing-page-section-heading

.. rubric:: Applications

These examples show how convex templates appear in domain-specific contexts such as information theory,
signal processing, and finance.
Use this group when you want to see how the same mathematical patterns translate into practical models.

.. list-table:: Examples in this group
   :widths: 34 66
   :header-rows: 1

   * - Example
     - Main structure
   * - :ref:`Entropy Maximization <example-entropy-maximization>`
     - maximum-entropy distribution under affine constraints
   * - :ref:`Fault Detection <example-fault-detection>`
     - sparse event recovery with box constraints
   * - :ref:`Water Filling <example-water-filling>`
     - logarithmic utility with a single resource budget
   * - :ref:`Image Deblurring <example-convolutional-image-deblurring>`
     - blurred observation with convolution and edge-preserving regularization
   * - :ref:`Portfolio Optimization <example-portfolio-optimization>`
     - mean-variance allocation with budget and nonnegativity constraints

.. toctree::
   :maxdepth: 1

   entropy_maximization
   fault_detection
   water_filling
   convolutional_image_deblurring
   portfolio_optimization

.. _examples-udf:

.. rst-class:: landing-page-section-heading

.. rubric:: Examples with User-Defined Proximal Functions

These examples show how to extend |ADMM| when the modeling pattern is a strong fit but one proximal term
is not available as a built-in atom. They cover custom sparsity penalties, low-rank and manifold-style
constraints, and projection-based indicators. Most of the nonconvex UDFs below fall outside the disciplined
convex programming rules enforced by tools such as CVXPY. In those cases the solver acts as a practical
local method, and the result should be interpreted as a locally optimal solution or stationary point rather
than a globally optimal one.
Use this group when you need a custom proximal block but still want to stay inside the same symbolic
modeling workflow.

.. list-table:: Examples in this group
   :widths: 45 55
   :header-rows: 1

   * - Example
     - Main structure
   * - :ref:`L0 Norm <udf-example-l0>`
     - exact sparsity penalty via hard thresholding
   * - :ref:`L0 Ball Indicator <udf-example-l0-ball>`
     - cardinality budget enforced by sparse-set projection
   * - :ref:`L1/2 Quasi-Norm <udf-example-lhalf>`
     - nonconvex sparsity promotion stronger than L1
   * - :ref:`Group Sparsity <udf-example-group-sparsity>`
     - exact block sparsity through groupwise proximal updates
   * - :ref:`Matrix Rank Function <udf-example-rank>`
     - explicit low-rank promotion by singular-value thresholding
   * - :ref:`Rank-r Indicator <udf-example-rank-r>`
     - hard rank cap enforced by truncated SVD projection
   * - :ref:`The Unit-Sphere Indicator <udf-example-unit-sphere>`
     - fixed-norm vector feasibility on the unit sphere
   * - :ref:`The Stiefel-Manifold Indicator <udf-example-stiefel>`
     - orthonormal-column matrix feasibility on the Stiefel manifold
   * - :ref:`The Simplex Indicator <udf-example-simplex>`
     - simplex projection for probability-style vectors
   * - :ref:`The Binary Indicator <udf-example-binary>`
     - binary-valued vector feasibility by coordinatewise projection
   * - :ref:`L0-Regularized Regression <udf-example-l0-regression>`
     - combining a UDF with a sensing matrix and linear constraints

.. toctree::
   :maxdepth: 1

   udf_l0_norm
   udf_l0_ball_indicator
   udf_lhalf_norm
   udf_group_sparsity
   udf_matrix_rank
   udf_rank_r_indicator
   udf_unit_sphere_indicator
   udf_stiefel_indicator
   udf_simplex_indicator
   udf_binary_indicator
   udf_l0_regression

For detailed symbol documentation, see the :doc:`../5_API_Document/index`.