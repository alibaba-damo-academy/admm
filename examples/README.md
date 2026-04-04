# ADMM Examples

Standalone, runnable examples for every documented ADMM use case.

## Quick Start

```bash
python examples/portfolio_optimization.py
python examples/least_squares.py
python examples/lasso_regression.py
```

## Available Examples

### Getting Started

| Example | Description |
|---------|-------------|
| [`overview_portfolio.py`](overview_portfolio.py) | Chapter 1 illustrative portfolio (n=20) |
| [`minimal_model.py`](minimal_model.py) | Simplest complete ADMM workflow |
| [`modeling_workflow.py`](modeling_workflow.py) | Full 7-step modeling workflow |
| [`lasso_regression.py`](lasso_regression.py) | Parameter binding with Lasso |
| [`constraints_projection.py`](constraints_projection.py) | Projection onto mixed constraint sets |
| [`udf_intro_l0.py`](udf_intro_l0.py) | UDF introduction: L0 norm walkthrough |

### Core Convex Forms

| Example | Description |
|---------|-------------|
| [`linear_program.py`](linear_program.py) | Linear programming |
| [`quadratic_program.py`](quadratic_program.py) | Quadratic programming |
| [`semidefinite_program.py`](semidefinite_program.py) | Semidefinite programming (PSD variable) |
| [`second_order_cone_program.py`](second_order_cone_program.py) | Second-order cone programming |

### Data Fitting

| Example | Description |
|---------|-------------|
| [`least_squares.py`](least_squares.py) | Linear least squares |
| [`ridge_regression.py`](ridge_regression.py) | Ridge regression (L2 regularization) |
| [`huber_regression.py`](huber_regression.py) | Robust regression with Huber loss |
| [`sparse_logistic_regression.py`](sparse_logistic_regression.py) | Sparse logistic regression |
| [`svm_with_l1.py`](svm_with_l1.py) | SVM with L1 regularization |
| [`quantile_regression.py`](quantile_regression.py) | Quantile regression (asymmetric loss) |

### Structured Matrix Problems

| Example | Description |
|---------|-------------|
| [`robust_pca.py`](robust_pca.py) | Robust PCA (low-rank + sparse) |
| [`sparse_inverse_covariance.py`](sparse_inverse_covariance.py) | Graphical Lasso (sparse precision matrix) |

### Applications

| Example | Description |
|---------|-------------|
| [`entropy_maximization.py`](entropy_maximization.py) | Maximum-entropy distribution |
| [`fault_detection.py`](fault_detection.py) | Sparse fault detection |
| [`water_filling.py`](water_filling.py) | Water-filling power allocation |
| [`image_deblurring.py`](image_deblurring.py) | TV-regularized image deblurring |
| [`portfolio_optimization.py`](portfolio_optimization.py) | Mean-variance portfolio (n=50) |

### User-Defined Proximal Functions (UDF)

| Example | Description |
|---------|-------------|
| [`udf_l0_norm.py`](udf_l0_norm.py) | L0 norm (hard thresholding) |
| [`udf_l0_ball.py`](udf_l0_ball.py) | L0 ball indicator (cardinality constraint) |
| [`udf_lhalf.py`](udf_lhalf.py) | L1/2 quasi-norm (nonconvex sparsity) |
| [`udf_group_sparsity.py`](udf_group_sparsity.py) | Group sparsity (column-wise L0) |
| [`udf_matrix_rank.py`](udf_matrix_rank.py) | Matrix rank (SVD thresholding) |
| [`udf_rank_r.py`](udf_rank_r.py) | Rank-r indicator (truncated SVD) |
| [`udf_unit_sphere.py`](udf_unit_sphere.py) | Unit-sphere indicator |
| [`udf_stiefel.py`](udf_stiefel.py) | Stiefel-manifold indicator |
| [`udf_simplex.py`](udf_simplex.py) | Simplex indicator |
| [`udf_binary.py`](udf_binary.py) | Binary indicator ({0,1} projection) |
| [`udf_l0_regression.py`](udf_l0_regression.py) | L0-regularized nonnegative regression |

## Running All Examples

```bash
for f in examples/*.py; do echo "=== $f ===" && python "$f"; done
```

## Prerequisites

```bash
pip install admm
```

## Learning Path

1. Start with `minimal_model.py` and `least_squares.py`
2. Try `portfolio_optimization.py` and `lasso_regression.py`
3. Explore advanced applications (`robust_pca.py`, `semidefinite_program.py`)
4. Study UDF examples for custom proximal operators
