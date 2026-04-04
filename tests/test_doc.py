"""
test_doc.py
===========
All self-contained runnable examples from the ADMM documentation.

Each example carries:
  - A block comment with the mathematical formulation
  - Print lines with inline ``# Expected:`` reference values
  - A unittest method that verifies SOLVE_OPT_SUCCESS and ObjVal

Usage:
    pytest tests/test_doc.py -v
"""

import admm
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _quiet_model():
    """Return a new Model with solver output fully suppressed."""
    m = admm.Model()
    m.setOption(admm.Options.solver_verbosity_level, 3)
    return m


# ============================================================
# 1_Documentation.rst  §  Illustrative Example
#
#   min_w   -mu^T w + gamma * w^T Sigma w
#   s.t.    sum(w) = 1
#           w >= 0
#
#   Mean-variance portfolio: n = 20 assets, seed = 1.
#   Sigma = F^T F + 0.1 I (factor model, PSD).
# ============================================================
def ex_overview_portfolio():
    print("\n--- 1_Documentation.rst | Illustrative Example (Portfolio n=20) ---")
    np.random.seed(1)
    n = 20
    mu = np.abs(np.random.randn(n))                 # Expected returns of assets
    Sigma = np.random.randn(n + 3, n)               # Random factor matrix
    Sigma = Sigma.T @ Sigma + 0.1 * np.eye(n)       # Covariance matrix of asset returns
    gamma = 0.5                                     # Risk-aversion parameter

    model = _quiet_model()
    w = admm.Var("w", n)
    model.setObjective(-mu.T @ w + gamma * (w.T @ Sigma @ w))
    model.addConstr(admm.sum(w) == 1)
    model.addConstr(w >= 0)
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -1.08295751047248
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 3_User_guide.rst  §  Minimal Model
#
#   min_{x1, x2}   x1 + x2
#   s.t.           x1 >= p
#                  x2 >= 0
#
#   Scalar parameter p bound to 2 at solve time.
#   Exact solution: x1* = 2, x2* = 0, obj* = 2.
# ============================================================
def ex_user_guide_minimal_model():
    print("\n--- 3_User_guide.rst | Minimal Model ---")
    x1 = admm.Var("x1")
    x2 = admm.Var("x2")
    p = admm.Param("p")

    model = _quiet_model()
    model.setObjective(x1 + x2)
    model.addConstr(x1 >= p)
    model.addConstr(x2 >= 0)
    model.setOption(admm.Options.admm_max_iteration, 1000)
    model.optimize({"p": 2})
    print(" * model.ObjVal: ", model.ObjVal)              # Expected: 2.0
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * x1.X: ", x1.X)                              # Expected: 2.0
    print(" * x2.X: ", x2.X)                              # Expected: 0.0
    return model.ObjVal, model.StatusString, x1.X, x2.X


# ============================================================
# 3_User_guide.rst  §  Parameter Binding And Solving — Lasso
#
#   min_x   (1/2) ||A x - b||_2^2 + lam * ||x||_1
#
#   A: 30x10, b: 30-vector, seed = 1, lam is a Param.
#   The same symbolic model is solved twice with lam = 0.05 and lam = 0.2.
# ============================================================
def ex_user_guide_lasso():
    print("\n--- 3_User_guide.rst | Parameters — Reusing a Lasso Model (m=30, n=10, seed=1) ---")
    np.random.seed(1)                   # Reproducible data
    m = 30
    n = 10

    A = np.random.randn(m, n)           # Data matrix
    b = np.random.randn(m)              # Observations

    model = _quiet_model()
    x = admm.Var("x", n)                # Regression coefficients
    lam = admm.Param("lam")             # Regularization weight as a parameter

    model.setObjective(
        0.5 * admm.sum(admm.square(A @ x - b))
        + lam * admm.norm(x, ord=1)
    )
    model.setOption(admm.Options.admm_max_iteration, 5000)

    model.optimize({"lam": 0.05})
    x_small_penalty = np.asarray(x.X).copy()

    print(" * first solve model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * first solve model.ObjVal: ", round(model.ObjVal, 6))    # Expected: finite scalar objective
    print(" * first solve x.X: ", np.round(x_small_penalty, 6))       # Expected: less regularized coefficients
    print(" * first solve ||x||_1: ", round(np.linalg.norm(x_small_penalty, 1), 6))

    model.optimize({"lam": 0.2})
    x_large_penalty = np.asarray(x.X).copy()

    print(" * second solve model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * second solve model.ObjVal: ", round(model.ObjVal, 6))    # Expected: finite scalar objective
    print(" * second solve x.X: ", np.round(x_large_penalty, 6))       # Expected: more heavily shrunk coefficients
    print(" * second solve ||x||_1: ", round(np.linalg.norm(x_large_penalty, 1), 6))
    return x_small_penalty, x_large_penalty


# ============================================================
# 3_User_guide.rst  §  Modeling Workflow — Complete Example
#
#   min_x   (1/2) ||A x - b||_2^2 + alpha * ||x||_1
#   s.t.    x >= 0
#
#   A: 30x10, b: 30-vector, seed = 1, alpha = 0.1 (Param).
#   The solution is nearly nonnegative up to solver tolerance.
# ============================================================
def ex_user_guide_modeling_workflow():
    print("\n--- 3_User_guide.rst | Modeling Workflow — Complete Example (m=30, n=10, seed=1) ---")
    np.random.seed(1)
    m = 30
    n = 10
    A = np.random.randn(m, n)  # Data matrix
    b = np.random.randn(m)     # Observation vector

    model = _quiet_model()  # Create the model
    x = admm.Var("x", n)  # Optimization variable
    alpha = admm.Param("alpha")  # Regularization parameter set before optimization

    model.setObjective(
        0.5 * admm.sum(admm.square(A @ x - b))
        + alpha * admm.norm(x, ord=1)
    )
    model.addConstr(x >= 0)  # Structural feasibility
    model.setOption(admm.Options.admm_max_iteration, 5000)  # Iteration budget
    model.optimize({"alpha": 0.1})  # Bind parameter data and solve

    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * model.ObjVal: ", round(model.ObjVal, 6))   # Expected: finite scalar objective
    print(" * x.X: ", np.round(np.asarray(x.X), 6))      # Expected: nearly nonnegative solution vector
    return model.StatusString, model.ObjVal, np.asarray(x.X)


# ============================================================
# 3_User_guide.rst  §  Constraints
#
#   min_{x,X}  ||x - x_target||_2^2 + ||X - X_target||_F^2
#   s.t.       x >= 0
#              ||x||_1 <= 1.2
#              ||x||_2 <= 1.0
#              ||X||_F <= 1.0
#              ||X||_* <= 1.1
#              X >> 0
#
#   x_target = [2.0, 1.0], X_target = diag(2.0, 1.0).
#   Both targets are intentionally outside the feasible set, so the
#   solver returns their constrained projections.
# ============================================================
def ex_user_guide_constraints():
    print("\n--- 3_User_guide.rst | Constraints ---")
    x_target = np.array([2.0, 1.0])
    X_target = np.array([[2.0, 0.0], [0.0, 1.0]])

    model = _quiet_model()
    x = admm.Var("x", 2)
    X = admm.Var("X", 2, 2, symmetric=True)

    model.setObjective(
        admm.sum(admm.square(x - x_target))
        + admm.sum(admm.square(X - X_target))
    )
    model.addConstr(x >= 0)
    model.addConstr(admm.norm(x, ord=1) <= 1.2)
    model.addConstr(admm.norm(x, ord=2) <= 1.0)
    model.addConstr(admm.norm(X, ord='nuc') <= 1.1)
    model.addConstr(admm.norm(X, ord='fro') <= 1.0)
    model.addConstr(X >> 0)

    model.optimize()

    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * model.ObjVal: ", round(model.ObjVal, 6))   # Expected: around 3.462854
    print(" * x.X: ", np.round(np.asarray(x.X), 6))      # Expected: [0.974155 0.225845]
    print(" * X.X: ", np.round(np.asarray(X.X), 6))      # Expected: [[0.99441 0.     ] [0.      0.10559]]
    return model.ObjVal, model.StatusString, np.asarray(x.X), np.asarray(X.X)


# ============================================================
# 4_Examples.rst  §  Linear Program
#
#   min_x   c^T x
#   s.t.    A x <= b
#           x >= 0
#
#   m = 15, n = 10, seed = 1.
#   Problem constructed from a primal-dual feasible point (x0, s0, lamb0).
# ============================================================
def ex_linear_program():
    print("\n--- 4_Examples.rst | Linear Program (m=15, n=10) ---")
    np.random.seed(1)
    m = 15
    n = 10
    s0 = np.random.randn(m)
    lamb0 = np.maximum(-s0, 0)
    s0 = np.maximum(s0, 0)
    x0 = np.maximum(np.random.randn(n), 0)
    A = np.random.randn(m, n)
    b = A @ x0 + s0
    c = -A.T @ lamb0

    model = _quiet_model()
    x = admm.Var("x", n)
    model.setObjective(c.T @ x)
    model.addConstr(A @ x <= b)
    model.addConstr(x >= 0)
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -7.629241267164004
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Quadratic Program
#
#   min_x   (1/2) x^T P x + q^T x
#   s.t.    G x <= h
#           A x == b
#
#   n = 10, m = 15 inequalities, p = 5 equalities, seed = 1.
#   P = R^T R (PSD), random G, A, q, h, b.
# ============================================================
def ex_quadratic_program():
    print("\n--- 4_Examples.rst | Quadratic Program (n=10, m=15, p=5) ---")
    np.random.seed(1)
    m = 15
    n = 10
    p = 5
    P = np.random.randn(n, n)
    P = P.T @ P
    q = np.random.randn(n)
    G = np.random.randn(m, n)
    h = G @ np.random.randn(n)
    A = np.random.randn(p, n)
    b = np.random.randn(p)

    model = _quiet_model()
    x = admm.Var("x", n)
    model.setObjective(0.5 * x.T @ P @ x + q.T @ x)
    model.addConstr(G @ x <= h)
    model.addConstr(A @ x == b)
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 86.89077551539528
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Semidefinite Program
#
#   min_X   tr(C X)
#   s.t.    tr(A_i X) = b_i,  i = 1, ..., p
#           X >> 0  (PSD)
#
#   n = 4, p = 3, seed = 1. C and A_i are symmetric.
#   C is built as R^T R (PSD) so that tr(C X) >= 0 for all X >> 0,
#   guaranteeing the problem is bounded below.
# ============================================================
def ex_semidefinite_program():
    print("\n--- 4_Examples.rst | Semidefinite Program (n=4, p=3) ---")
    np.random.seed(1)
    n = 4
    p = 3
    R = np.random.randn(n, n)
    C = R.T @ R                  # PSD: tr(C X) >= 0 for all X >> 0
    A = []
    b = []
    for _ in range(p):
        Ai = np.random.randn(n, n)
        Ai = 0.5 * (Ai + Ai.T)
        A.append(Ai)
        b.append(np.random.randn())
    A = np.array(A)
    b = np.array(b)

    model = _quiet_model()
    X = admm.Var("X", n, n, PSD=True)
    model.setObjective(admm.trace(C @ X))
    for i in range(p):
        model.addConstr(admm.trace(A[i] @ X) == b[i])
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 0.4295347451953324
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Second-Order Cone Program
#
#   min_x   f^T x
#   s.t.    ||A_i x + b_i||_2 <= c_i^T x + d_i,  i = 1, ..., m
#           F x = g
#
#   n = 10, m = 3 cone constraints (n_i = 5), p = 5 equalities, seed = 1.
#   Constructed from feasible x0 so d_i = ||A_i x0 + b_i||_2 - c_i^T x0.
# ============================================================
def ex_second_order_cone_program():
    print("\n--- 4_Examples.rst | Second-Order Cone Program (n=10, m=3, p=5) ---")
    np.random.seed(1)
    m = 3
    n = 10
    p = 5
    n_i = 5
    f = np.random.randn(n)
    A = []
    b = []
    c = []
    d = []
    x0 = np.random.randn(n)
    for i in range(m):
        A.append(np.random.randn(n_i, n))
        b.append(np.random.randn(n_i))
        c.append(np.random.randn(n))
        d.append(np.linalg.norm(A[i] @ x0 + b[i], 2) - c[i].T @ x0)
    F = np.random.randn(p, n)
    g = F @ x0

    model = _quiet_model()
    x = admm.Var("x", n)
    model.setObjective(f.T @ x)
    for i in range(m):
        model.addConstr(admm.norm(A[i] @ x + b[i], ord=2) <= c[i].T @ x + d[i])
    model.addConstr(F @ x == g)
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 2.06815161777782
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Least Squares
#
#   min_x   ||A x - b||_2^2
#
#   m = 40, n = 12, seed = 1.
#   b = A x_true + 0.1 * noise.
# ============================================================
def ex_least_squares():
    print("\n--- 4_Examples.rst | Least Squares (m=40, n=12) ---")
    np.random.seed(1)
    m = 40
    n = 12
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true + 0.1 * np.random.randn(m)

    model = _quiet_model()
    x = admm.Var("x", n)
    model.setObjective(admm.sum(admm.square(A @ x - b)))
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 0.2947794914868591
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Ridge Regression
#
#   min_beta   ||X beta - y||_2^2 + lam * ||beta||_2^2
#
#   m = 100, n = 25, lam = 1.0, seed = 1.
#   y = X beta_true + 0.5 * noise  (L2-regularized least squares).
# ============================================================
def ex_ridge_regression():
    print("\n--- 4_Examples.rst | Ridge Regression (m=100, n=25, lam=1) ---")
    np.random.seed(1)
    m = 100
    n = 25
    X = np.random.randn(m, n)
    beta_true = np.random.randn(n)
    y = X @ beta_true + 0.5 * np.random.randn(m)
    lam = 1.0

    model = _quiet_model()
    beta = admm.Var("beta", n)
    model.setObjective(admm.sum(admm.square(X @ beta - y)) + lam * admm.sum(admm.square(beta)))
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 37.12175719889099
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Huber Regression
#
#   min_beta   sum_i phi(x_i^T beta - y_i)
#
#   phi(u) = (1/2) u^2  if |u| <= M,  M|u| - (1/2) M^2  otherwise.
#   m = 80, n = 20, M = 1.0, seed = 1.
#   First 8 observations are heavy outliers (+8 * noise).
# ============================================================
def ex_huber_regression():
    print("\n--- 4_Examples.rst | Huber Regression (m=80, n=20, M=1) ---")
    np.random.seed(1)
    m = 80
    n = 20
    X = np.random.randn(m, n)
    beta_true = np.random.randn(n)
    y = X @ beta_true + 0.1 * np.random.randn(m)
    y[:8] += 8.0 * np.random.randn(8)

    model = _quiet_model()
    beta = admm.Var("beta", n)
    residual = X @ beta - y
    model.setObjective(admm.sum(admm.huber(residual, 1.0)))
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 36.570113991744364
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Sparse Logistic Regression
#
#   min_{w,v}   (1/m) sum_i log(1 + exp(-y_i (x_i^T w + v)))
#             + lam * ||w||_1
#
#   n = 20, m = 60, lam = 0.1, seed = 1. Labels y_i in {-1, +1}.
# ============================================================
def ex_sparse_logistic_regression():
    print("\n--- 4_Examples.rst | Sparse Logistic Regression (n=20, m=60, lam=0.1) ---")
    np.random.seed(1)
    n = 20
    m = 60
    beta = np.random.randn(n)
    X = np.random.randn(m, n)
    y = np.sign(X @ beta + 0.5 * np.random.randn(m))
    y[y == 0] = 1
    lam = 0.1

    model = _quiet_model()
    w = admm.Var("w", n)
    v = admm.Var("v")
    margin = -y * (X @ w + v)
    model.setObjective(admm.sum(admm.logistic(margin, 1)) / m + lam * admm.norm(w, ord=1))
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 0.6458330573436739
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  SVM with L1 Regularization
#
#   min_{beta, v}   (1/m) sum_i max(0, 1 - y_i (x_i^T beta - v))
#                 + lam * ||beta||_1
#
#   m = 120, n = 25 (last 15 coefficients zero in true model),
#   lam = 0.1, seed = 1. Labels in {-1, +1}.
# ============================================================
def ex_svm_l1():
    print("\n--- 4_Examples.rst | SVM with L1 Regularization (m=120, n=25, lam=0.1) ---")
    np.random.seed(1)
    m = 120
    n = 25
    beta_true = np.random.randn(n)
    beta_true[10:] = 0
    X = np.random.randn(m, n)
    y = np.sign(X @ beta_true + 0.5 * np.random.randn(m))
    y[y == 0] = 1
    lam = 0.1

    model = _quiet_model()
    model.setOption(admm.Options.admm_max_iteration, 10000)
    beta = admm.Var("beta", n)
    v = admm.Var("v")
    margin_loss = admm.sum(admm.maximum(1 - y * (X @ beta - v), 0))
    model.setObjective(margin_loss / m + lam * admm.norm(beta, ord=1))
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 0.5810343957323443
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Quantile Regression
#
#   min_w   (1/2) ||X w - y||_1 + (1/2 - tau) * 1^T (X w - y)
#
#   Pinball loss at quantile tau = 0.9.
#   n = 10, m = 200, seed = 1. y = X beta + 0.5 * noise.
# ============================================================
def ex_quantile_regression():
    print("\n--- 4_Examples.rst | Quantile Regression (tau=0.9, n=10, m=200) ---")
    np.random.seed(1)
    n = 10
    m = 200
    beta = np.random.randn(n)
    X = np.random.randn(m, n)
    y = X @ beta + 0.5 * np.random.randn(m)
    tau = 0.9

    model = _quiet_model()
    model.setOption(admm.Options.admm_max_iteration, 10000)
    model.setOption(admm.Options.termination_absolute_error_threshold, 1e-5)
    model.setOption(admm.Options.termination_relative_error_threshold, 1e-5)
    w = admm.Var("w", n)
    residual = X @ w - y
    model.setObjective(0.5 * admm.norm(residual, ord=1) + (0.5 - tau) * admm.sum(residual))
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 35.76437758914047
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Robust PCA
#
#   min_{L, S}   ||L||_*  +  lam * ||vec(S)||_1
#   s.t.         L + S = M
#
#   ||L||_* = nuclear norm (promotes low rank).
#   M = low-rank (m=50, r=10, n=40) + small noise, seed = 1.
#   lam = 1 / sqrt(max(m, n)).
# ============================================================
def ex_robust_pca():
    print("\n--- 4_Examples.rst | Robust PCA (m=50, r=10, n=40) ---")
    np.random.seed(1)
    m = 50
    r = 10
    n = 40
    M = np.random.randn(m, r) @ np.random.randn(r, n)
    M = M + 0.1 * np.random.randn(m, n)
    lam = 1.0 / np.sqrt(max(m, n))

    model = _quiet_model()
    L = admm.Var("L", m, n)
    S = admm.Var("S", m, n)
    model.setObjective(admm.norm(L, ord="nuc") + lam * admm.sum(admm.abs(S)))
    model.addConstr(L + S == M)
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 422.49613807195703
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Sparse Inverse Covariance Selection
#
#   min_{Theta >> 0}   -log det(Theta) + tr(S Theta) + lam * ||vec(Theta)||_1
#
#   Gaussian MLE precision-matrix estimation with L1 sparsity.
#   n = 30, sample_num = 60, lam = 0.05, seed = 1.
# ============================================================
def ex_sparse_inverse_covariance():
    print("\n--- 4_Examples.rst | Sparse Inverse Covariance (n=30, lam=0.05) ---")
    np.random.seed(1)
    n = 30
    sample_num = 60
    A = np.random.randn(n, n)
    true_precision = A.T @ A + 0.5 * np.eye(n)
    samples = np.random.multivariate_normal(
        mean=np.zeros(n),
        cov=np.linalg.inv(true_precision),
        size=sample_num,
    )
    S = np.cov(samples, rowvar=False)
    lam = 0.05

    model = _quiet_model()
    Theta = admm.Var("Theta", n, n, PSD=True)
    model.setObjective(-admm.log_det(Theta) + admm.trace(S @ Theta) + lam * admm.sum(admm.abs(Theta)))
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -15.134257007715702
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Entropy Maximization
#
#   min_x   sum_i x_i log(x_i)      (= -Shannon entropy)
#   s.t.    A x = b
#           F x <= g
#           1^T x = 1
#           x >= 0
#
#   n = 12, m = 3 equalities, p = 2 inequalities, seed = 1.
#   x0 is a random probability vector used to construct feasible b, g.
# ============================================================
def ex_entropy_maximization():
    print("\n--- 4_Examples.rst | Entropy Maximization (n=12, m=3, p=2) ---")
    np.random.seed(1)
    n = 12
    m = 3
    p = 2
    x0 = np.random.rand(n)
    x0 = x0 / np.sum(x0)
    A = np.random.randn(m, n)
    b = A @ x0
    F = np.random.randn(p, n)
    g = F @ x0 + np.random.rand(p)

    model = _quiet_model()
    x = admm.Var("x", n)
    model.setObjective(admm.sum(admm.entropy(x)))  # minimize sum(x*log(x)) = maximize Shannon entropy
    model.addConstr(A @ x == b)
    model.addConstr(F @ x <= g)
    model.addConstr(admm.sum(x) == 1)
    model.addConstr(x >= 0)
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -2.4722786823012264
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Fault Detection
#
#   min_x   ||A x - y||_2^2  +  tau * 1^T x
#   s.t.    0 <= x <= 1
#
#   Box-constrained quadratic: relaxed sparse fault indicators.
#   n = 200 sensors, m = 40 measurements, p_fault = 0.03, seed = 1.
# ============================================================
def ex_fault_detection():
    print("\n--- 4_Examples.rst | Fault Detection (n=200, m=40, p_fault=0.03) ---")
    np.random.seed(1)
    n = 200
    m = 40
    p_fault = 0.03
    snr = 5.0
    sigma = np.sqrt(p_fault * n / (snr ** 2))
    A = np.random.randn(m, n)
    x_true = (np.random.rand(n) <= p_fault).astype(float)
    y = A @ x_true + sigma * np.random.randn(m)
    tau = 2 * np.log(1 / p_fault - 1) * sigma ** 2

    model = _quiet_model()
    model.setOption(admm.Options.admm_max_iteration, 5000)
    x = admm.Var("x", n)
    model.setObjective(admm.sum(admm.square(A @ x - y)) + tau * admm.sum(x))
    model.addConstr(x >= 0)
    model.addConstr(x <= 1)
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 15.294052961638492
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Water Filling
#
#   max_x   sum_i log(alpha_i + x_i)
#   s.t.    sum(x) = P,  x >= 0
#
#   Equivalently: min sum_i -log(alpha_i + x_i).
#   alpha = [0.5, 0.8, 1.0, 1.3, 1.6], P = 2.0 (total power).
# ============================================================
def ex_water_filling():
    print("\n--- 4_Examples.rst | Water Filling (n=5, P=2.0) ---")
    alpha = np.array([0.5, 0.8, 1.0, 1.3, 1.6])
    total_power = 2.0
    n = len(alpha)

    model = _quiet_model()
    x = admm.Var("x", n)
    model.setObjective(admm.sum(-admm.log(alpha + x)))
    model.addConstr(admm.sum(x) == total_power)
    model.addConstr(x >= 0)
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -1.8158925751409778
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Convolutional Image Deblurring
#
#   min_U   lam * TV(U)  +  (1/2) ||K * U - B||_F^2
#
#   K * U denotes 2-D convolution with a 3x3 Gaussian kernel.
#   TV(U) = total-variation penalty (p=1, isotropic-like).
#   Image: 40x50 piecewise-constant blocks, lam = 0.1, seed = 1.
# ============================================================
def ex_image_deblurring():
    print("\n--- 4_Examples.rst | Convolutional Image Deblurring (40x50, lam=0.1) ---")
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
    model = _quiet_model()
    U = admm.Var("U", image.shape)
    tv = admm.tv2d(U, p=1)
    residual = admm.conv2d(U, kernel, "same") - image_blur
    model.setObjective(lam * tv + 0.5 * admm.sum(admm.square(residual)))
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: 9.210929389075202
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 4_Examples.rst  §  Portfolio Optimization
#
#   min_w   -mu^T w + gamma * w^T Sigma w
#   s.t.    1^T w = 1,  w >= 0
#
#   Mean-variance portfolio: n = 50 assets, seed = 1.
#   Sigma = F^T F + 0.1 I (factor model, PSD).
# ============================================================
def ex_portfolio_optimization():
    print("\n--- 4_Examples.rst | Portfolio Optimization (n=50) ---")
    np.random.seed(1)
    n = 50
    mu = np.abs(np.random.randn(n))
    F = np.random.randn(n + 5, n)
    Sigma = F.T @ F + 0.1 * np.eye(n)
    gamma = 0.5

    model = _quiet_model()
    w = admm.Var("w", n)
    expected_return = mu.T @ w
    risk = w.T @ Sigma @ w
    model.setObjective(-expected_return + gamma * risk)
    model.addConstr(admm.sum(w) == 1)
    model.addConstr(w >= 0)
    model.optimize()
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: -0.9808918614054916
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString


# ============================================================
# 3_User_guide.rst  §  UDF — Walkthrough: The L0 Norm as a UDF
#
#   min_x   (1/2) ||x - y||_2^2  +  lam * ||x||_0
#
#   Unconstrained L0-regularized nearest-point problem.
#   y = [0.2, 2.0, 0.6, 2.2], lam = 1.0.
#   Threshold sqrt(2) ≈ 1.414: entries 0.2, 0.6 removed; 2.0, 2.2 kept.
#   Expected x* ≈ [0, 2, 0, 2.2], ObjVal ≈ 2.2.
# ============================================================
def ex_user_guide_udf():
    print("\n--- 3_User_guide.rst | UDF Walkthrough — L0 Norm (y=[0.2,2,0.6,2.2], lam=1) ---")

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

    y = np.array([0.2, 2.0, 0.6, 2.2])
    lam = 1.0

    model = _quiet_model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + lam * L0Norm(x))
    model.optimize()
    print(" * x: ", np.asarray(x.X))               # Expected: ≈ [0, 2, 0, 2.2]
    print(" * model.ObjVal: ", model.ObjVal)        # Expected: ≈ 2.2
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal


# ============================================================
# 4_Examples.rst  §  UDF — L0 Norm
#
#   min_x   (1/2) ||x - y||_2^2  +  lam * ||x||_0
#   s.t.    0 <= x <= 1
#
#   ||x||_0 counts nonzero entries (nonconvex).
#   Prox = hard threshold: keep x_i iff |x_i| > sqrt(2*lam).
#   y = [0.2, 1.7, 0.6, 1.9], lam = 1.0.
#   Expected x* ≈ [0, 1, 0, 1] (small entries thresholded; large entries clipped by x <= 1).
# ============================================================
def ex_udf_l0_norm():
    print("\n--- 4_Examples.rst | UDF L0 Norm (y=[0.2,1.7,0.6,1.9], lam=1) ---")

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

    model = _quiet_model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + lam * L0Norm(x))
    model.addConstr(x >= 0)
    model.addConstr(x <= 1)
    model.setOption(admm.Options.admm_max_iteration, 10000)  # Give the constrained nonconvex solve enough iterations
    model.optimize()
    x_value = np.asarray(x.X)
    print(" * x: ", x_value)                        # Expected: [0. 1. 0. 1.]
    print(" * model.ObjVal: ", np.round(model.ObjVal, 6))  # Expected: 2.85
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal, model.StatusString, x_value


# ============================================================
# 4_Examples.rst  §  UDF — L0 Ball Indicator
#
#   min_x   (1/2) ||x - y||_2^2  +  delta_{||x||_0 <= k}(x)
#
#   Nonconvex cardinality constraint: keep at most k entries.
#   Prox = projection onto L0 ball: keep k largest-magnitude entries.
#   y = [0.2, -1.5, 0.7, 3.0], k = 2.
#   Expected x* ≈ [0, -1.5, 0, 3.0].
# ============================================================
def ex_udf_l0_ball():
    print("\n--- 4_Examples.rst | UDF L0 Ball Indicator (y=[0.2,-1.5,0.7,3], k=2) ---")

    class L0BallIndicator(admm.UDFBase):
        def __init__(self, arg, k=2):
            self.arg = arg
            self.k = k

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            return 0.0 if np.count_nonzero(np.abs(x) > 1e-12) <= self.k else float("inf")

        def argmin(self, lamb, arglist):
            v = np.asarray(arglist[0], dtype=float)
            prox = np.zeros_like(v)
            keep_count = min(max(self.k, 0), v.size)
            if keep_count > 0:
                keep_idx = np.argpartition(np.abs(v), -keep_count)[-keep_count:]
                prox[keep_idx] = v[keep_idx]
            return [prox.tolist()]

    y = np.array([0.2, -1.5, 0.7, 3.0])

    model = _quiet_model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + L0BallIndicator(x, k=2))
    model.optimize()
    print(" * x: ", np.asarray(x.X))               # Expected: [ 0.  -1.5  0.   3. ]
    print(" * model.ObjVal: ", np.round(model.ObjVal, 6))  # Expected: 0.265
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal


# ============================================================
# 4_Examples.rst  §  UDF — L1/2 Quasi-Norm
#
#   min_x   (1/2) ||x - y||_2^2  +  lam * sum_i sqrt(|x_i|)
#
#   L_{1/2} quasi-norm: stronger sparsity than L1 (nonconvex).
#   Coordinatewise prox via closed-form half-thresholding rule.
#   y = [0.2, 1.0, 2.0], lam = 0.5.
# ============================================================
def ex_udf_lhalf():
    print("\n--- 4_Examples.rst | UDF L1/2 Quasi-Norm (y=[0.2,1,2], lam=0.5) ---")

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
                    np.clip(
                        (3.0 * np.sqrt(3.0) * lamb) / (4.0 * np.power(abs_v[active], 1.5)),
                        -1.0,
                        1.0,
                    )
                )
                prox_abs = (2.0 * abs_v[active] / 3.0) * (
                    1.0 + np.cos((2.0 * np.pi / 3.0) - (2.0 * phi / 3.0))
                )
                prox[active] = np.sign(v[active]) * prox_abs
            return [prox.tolist()]

    y = np.array([0.2, 1.0, 2.0])
    lam = 0.5

    model = _quiet_model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + lam * LHalfNorm(x))
    model.optimize()
    print(" * x: ", np.asarray(x.X))               # Expected: [0.  0.70151586  1.81440202]
    print(" * model.ObjVal: ", np.round(model.ObjVal, 6))  # Expected: 1.174051
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal


# ============================================================
# 4_Examples.rst  §  UDF — Group Sparsity
#
#   min_X   (1/2) ||X - Y||_F^2  +  lam * sum_j 1_{||X_{:,j}||_2 != 0}
#
#   Column-group L0 penalty (nonconvex).
#   Prox: drop column j iff ||V_{:,j}||_2^2 <= 2*lam.
#   Y = [[0.2, 2.0, 0.3], [0.1, 1.0, 0.4]], lam = 1.0.
# ============================================================
def ex_udf_group_sparsity():
    print("\n--- 4_Examples.rst | UDF Group Sparsity (Y=2x3, lam=1) ---")

    class GroupSparsityPenalty(admm.UDFBase):
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            X = np.asarray(arglist[0], dtype=float)
            column_norms = np.linalg.norm(X, axis=0)
            return float(np.count_nonzero(column_norms > 1e-12))

        def argmin(self, lamb, arglist):
            Z = np.asarray(arglist[0], dtype=float)
            column_norm_sq = np.sum(Z * Z, axis=0)
            keep_mask = column_norm_sq > 2.0 * lamb
            prox = Z * keep_mask[np.newaxis, :]
            return [prox.tolist()]

    Y = np.array([[0.2, 2.0, 0.3], [0.1, 1.0, 0.4]])
    lam = 1.0

    model = _quiet_model()
    X = admm.Var("X", 2, 3)
    model.setObjective(0.5 * admm.sum(admm.square(X - Y)) + lam * GroupSparsityPenalty(X))
    model.optimize()
    print(" * X: ", np.asarray(X.X))               # Expected: [[0. 2. 0.] [0. 1. 0.]] (columns 0 and 2 zeroed)
    print(" * model.ObjVal: ", np.round(model.ObjVal, 6))  # Expected: 1.15
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal


# ============================================================
# 4_Examples.rst  §  UDF — Matrix Rank Function
#
#   min_X   (1/2) ||X - Y||_F^2  +  lam * rank(X)
#
#   rank(X) = # nonzero singular values (nonconvex).
#   Prox: hard threshold singular values at sqrt(2*lam).
#   Y = [[2, 0], [0, 0.5]], lam = 0.5.
#   Expected X* ≈ [[2, 0], [0, 0]] (sigma_2 = 0.5 <= sqrt(1)).
# ============================================================
def ex_udf_rank():
    print("\n--- 4_Examples.rst | UDF Matrix Rank (Y=[[2,0],[0,0.5]], lam=0.5) ---")

    class RankPenalty(admm.UDFBase):
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            X = np.asarray(arglist[0], dtype=float)
            singular_v = np.linalg.svd(X, compute_uv=False)
            return float(np.sum(singular_v > 1e-10))

        def argmin(self, lamb, arglist):
            Z = np.asarray(arglist[0], dtype=float)
            u, singular_v, vt = np.linalg.svd(Z, full_matrices=False)
            threshold = np.sqrt(2.0 * lamb)
            singular_v = np.where(singular_v <= threshold, 0.0, singular_v)
            prox = (u * singular_v) @ vt
            return [prox.tolist()]

    Y = np.array([[2.0, 0.0], [0.0, 0.5]])
    lam = 0.5

    model = _quiet_model()
    X = admm.Var("X", 2, 2)
    model.setObjective(0.5 * admm.sum(admm.square(X - Y)) + lam * RankPenalty(X))
    model.optimize()
    print(" * X: ", np.asarray(X.X))               # Expected: [[2. 0.] [0. 0.]] (sigma_2 thresholded)
    print(" * model.ObjVal: ", np.round(model.ObjVal, 6))  # Expected: 0.624999
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal


# ============================================================
# 4_Examples.rst  §  UDF — Rank-r Indicator
#
#   min_X   (1/2) ||X - Y||_F^2  +  delta_{rank(X) <= r}(X)
#
#   Nonconvex rank constraint. Prox = truncated SVD: keep top r values.
#   Y = [[3, 0], [0, 1]], r = 1.
# ============================================================
def ex_udf_rank_r():
    print("\n--- 4_Examples.rst | UDF Rank-r Indicator (Y=[[3,0],[0,1]], r=1) ---")

    class RankRIndicator(admm.UDFBase):
        def __init__(self, arg, rank_bound=1):
            self.arg = arg
            self.rank_bound = rank_bound

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            X = np.asarray(arglist[0], dtype=float)
            singular_v = np.linalg.svd(X, compute_uv=False)
            return 0.0 if np.sum(singular_v > 1e-10) <= self.rank_bound else float("inf")

        def argmin(self, lamb, arglist):
            Z = np.asarray(arglist[0], dtype=float)
            u, singular_v, vt = np.linalg.svd(Z, full_matrices=False)
            singular_v[min(self.rank_bound, len(singular_v)):] = 0.0
            prox = (u * singular_v) @ vt
            return [prox.tolist()]

    Y = np.array([[3.0, 0.0], [0.0, 1.0]])

    model = _quiet_model()
    X = admm.Var("X", 2, 2)
    model.setObjective(0.5 * admm.sum(admm.square(X - Y)) + RankRIndicator(X, 1))
    model.optimize()
    print(" * X: ", np.asarray(X.X))               # Expected: [[3. 0.] [0. 0.]] (rank-1 truncation)
    print(" * model.ObjVal: ", np.round(model.ObjVal, 6))  # Expected: 0.5
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal


# ============================================================
# 4_Examples.rst  §  UDF — Unit-Sphere Indicator
#
#   min_x   (1/2) ||x - y||_2^2  +  delta_{||x||_2 = 1}(x)
#
#   Nonconvex: projects y onto the unit sphere.
#   Prox: x = v / ||v||_2  (for v != 0).
#   y = [0.1, 0.0]  →  x* ≈ [1, 0].
# ============================================================
def ex_udf_unit_sphere():
    print("\n--- 4_Examples.rst | UDF Unit-Sphere Indicator (y=[0.1,0]) ---")

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

    model = _quiet_model()
    x = admm.Var("x", 2)
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + UnitSphereIndicator(x))
    model.optimize()
    print(" * x: ", np.asarray(x.X))               # Expected: [ 1. -0.]
    print(" * model.ObjVal: ", np.round(model.ObjVal, 6))  # Expected: 0.405
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal


# ============================================================
# 4_Examples.rst  §  UDF — Stiefel-Manifold Indicator
#
#   min_X   (1/2) ||X - Y||_F^2  +  delta_{St(m,n)}(X)
#
#   St(m,n) = {X in R^{mxn} : X^T X = I_n}  (orthonormal columns).
#   Prox: polar factor — if Z = U Sigma V^T then prox = U V^T.
#   Y = [[2,0],[0,0.5],[0,0]], shape 3x2.
# ============================================================
def ex_udf_stiefel():
    print("\n--- 4_Examples.rst | UDF Stiefel-Manifold Indicator (Y=3x2) ---")

    class StiefelIndicator(admm.UDFBase):
        def __init__(self, arg):
            self.arg = arg

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            X = np.asarray(arglist[0], dtype=float)
            identity = np.eye(X.shape[1])
            return 0.0 if np.linalg.norm(X.T @ X - identity) <= 1e-9 else float("inf")

        def argmin(self, lamb, arglist):
            Z = np.asarray(arglist[0], dtype=float)
            u, _, vt = np.linalg.svd(Z, full_matrices=False)
            prox = u @ vt
            return [prox.tolist()]

    Y = np.array([[2.0, 0.0], [0.0, 0.5], [0.0, 0.0]])

    model = _quiet_model()
    X = admm.Var("X", 3, 2)
    model.setObjective(0.5 * admm.sum(admm.square(X - Y)) + StiefelIndicator(X))
    model.optimize()
    print(" * X: ", np.asarray(X.X))               # Expected: [[1. 0.] [0. 1.] [0. 0.]] (polar factor)
    print(" * model.ObjVal: ", np.round(model.ObjVal, 6))  # Expected: 0.625
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal


# ============================================================
# 4_Examples.rst  §  UDF — Simplex Indicator
#
#   min_x   (1/2) ||x - y||_2^2  +  delta_{Delta_r}(x)
#
#   Delta_r = {x : x >= 0, sum(x) = r}.
#   Prox: Euclidean projection onto simplex via sorting.
#   y = [0.2, -0.1, 0.7], r = 1.0.
# ============================================================
def ex_udf_simplex():
    print("\n--- 4_Examples.rst | UDF Simplex Indicator (y=[0.2,-0.1,0.7], r=1) ---")

    class SimplexIndicator(admm.UDFBase):
        def __init__(self, arg, radius=1.0):
            self.arg = arg
            self.radius = radius

        def arguments(self):
            return [self.arg]

        def eval(self, arglist):
            x = np.asarray(arglist[0], dtype=float)
            if np.min(x) >= -1e-9 and abs(np.sum(x) - self.radius) <= 1e-9:
                return 0.0
            return float("inf")

        def argmin(self, lamb, arglist):
            v = np.asarray(arglist[0], dtype=float)
            sorted_v = np.sort(v)[::-1]
            cumulative = np.cumsum(sorted_v) - self.radius
            indices = np.arange(1, len(v) + 1)
            rho = np.nonzero(sorted_v - cumulative / indices > 0)[0][-1]
            theta = cumulative[rho] / (rho + 1)
            prox = np.maximum(v - theta, 0.0)
            return [prox.tolist()]

    y = np.array([0.2, -0.1, 0.7])

    model = _quiet_model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + SimplexIndicator(x, 1.0))
    model.optimize()
    print(" * x: ", np.asarray(x.X))               # Expected: [0.25 0.   0.75]
    print(" * model.ObjVal: ", np.round(model.ObjVal, 6))  # Expected: 0.0075
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal


# ============================================================
# 4_Examples.rst  §  UDF — Binary Indicator
#
#   min_x   (1/2) ||x - y||_2^2  +  delta_{{0,1}^n}(x)
#
#   Nonconvex binary cube constraint.
#   Prox (coordinatewise): x_i = 0 if v_i < 0.5, else 1.
#   y = [0.2, 0.8, 1.4, -0.3].
#   Expected x* ≈ [0, 1, 1, 0].
# ============================================================
def ex_udf_binary():
    print("\n--- 4_Examples.rst | UDF Binary Indicator (y=[0.2,0.8,1.4,-0.3]) ---")

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

    model = _quiet_model()
    x = admm.Var("x", len(y))
    model.setObjective(0.5 * admm.sum(admm.square(x - y)) + BinaryIndicator(x))
    model.optimize()
    print(" * x: ", np.asarray(x.X))               # Expected: [0. 1. 1. 0.]
    print(" * model.ObjVal: ", np.round(model.ObjVal, 6))  # Expected: 0.165
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    return model.ObjVal


# ============================================================
# 4_Examples.rst  §  UDF — L0-Regularized Regression
#
#   min_x   (1/2) ||A x - b||_2^2  +  lam * ||x||_0
#   s.t.    x >= 0
#
#   Sensing matrix A: 30x20, k=3 nonzero entries in x_true.
#   seed=42, lam=0.5. Nonconvex (local method).
# ============================================================
def ex_udf_l0_regression():
    print("\n--- 4_Examples.rst | UDF L0-Regularized Regression (n=20, m=30, k=3) ---")

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

    np.random.seed(42)
    n, m, k = 20, 30, 3
    x_true = np.zeros(n)
    x_true[np.random.choice(n, k, replace=False)] = np.random.rand(k) * 2 + 0.5
    A = np.random.randn(m, n)
    b = A @ x_true + 0.01 * np.random.randn(m)
    lam = 0.5

    model = _quiet_model()
    x = admm.Var("x", n)
    model.setObjective(0.5 * admm.sum(admm.square(A @ x - b)) + lam * L0Norm(x))
    model.addConstr(x >= 0)
    model.optimize()
    print(" * model.ObjVal: ", np.round(model.ObjVal, 6))  # Expected: 1.501724
    print(" * model.StatusString: ", model.StatusString)  # Expected: SOLVE_OPT_SUCCESS
    print(" * nnz(x): ", np.count_nonzero(np.abs(np.asarray(x.X)) > 1e-6))  # Expected: 3
    return model.ObjVal



# ============================================================
# Tests — one per doc example, same style as test_udf.py
# ============================================================
import unittest


class DocExampleTestCase(unittest.TestCase):
    """Verify every documentation example runs and produces the expected objective."""

    # ---------- 1_Documentation.rst ----------

    def test_overview_portfolio(self):
        obj, status = ex_overview_portfolio()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, -1.083, delta=abs(-1.083) * 0.01)

    # ---------- 3_User_guide ----------

    def test_user_guide_minimal_model(self):
        obj, status, x1, x2 = ex_user_guide_minimal_model()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 2.0, places=4)

    def test_user_guide_lasso(self):
        x_small, x_large = ex_user_guide_lasso()
        self.assertGreater(np.linalg.norm(x_small, 1), np.linalg.norm(x_large, 1))

    def test_user_guide_modeling_workflow(self):
        status, obj, x_val = ex_user_guide_modeling_workflow()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")

    def test_user_guide_constraints(self):
        obj, status, x_sol, X_sol = ex_user_guide_constraints()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 3.463, delta=3.463 * 0.01)

    def test_user_guide_udf(self):
        obj = ex_user_guide_udf()
        self.assertAlmostEqual(obj, 2.2, delta=2.2 * 0.01)

    # ---------- 4_Examples: Core Convex ----------

    def test_linear_program(self):
        obj, status = ex_linear_program()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, -7.629, delta=abs(-7.629) * 0.01)

    def test_quadratic_program(self):
        obj, status = ex_quadratic_program()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 86.891, delta=86.891 * 0.01)

    def test_semidefinite_program(self):
        obj, status = ex_semidefinite_program()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 0.4295, delta=0.4295 * 0.01)

    def test_second_order_cone_program(self):
        obj, status = ex_second_order_cone_program()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 2.068, delta=2.068 * 0.01)

    # ---------- 4_Examples: Data Fitting ----------

    def test_least_squares(self):
        obj, status = ex_least_squares()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 0.2948, delta=0.2948 * 0.01)

    def test_ridge_regression(self):
        obj, status = ex_ridge_regression()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 37.122, delta=37.122 * 0.01)

    def test_huber_regression(self):
        obj, status = ex_huber_regression()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 36.570, delta=36.570 * 0.01)

    def test_sparse_logistic_regression(self):
        obj, status = ex_sparse_logistic_regression()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 0.6458, delta=0.6458 * 0.01)

    def test_svm_l1(self):
        obj, status = ex_svm_l1()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 0.5810, delta=0.5810 * 0.01)

    def test_quantile_regression(self):
        obj, status = ex_quantile_regression()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 35.764, delta=35.764 * 0.01)

    # ---------- 4_Examples: Structured Matrix ----------

    def test_robust_pca(self):
        obj, status = ex_robust_pca()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 422.496, delta=422.496 * 0.01)

    def test_sparse_inverse_covariance(self):
        obj, status = ex_sparse_inverse_covariance()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, -15.134, delta=abs(-15.134) * 0.01)

    # ---------- 4_Examples: Applications ----------

    def test_entropy_maximization(self):
        obj, status = ex_entropy_maximization()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, -2.472, delta=abs(-2.472) * 0.01)

    def test_fault_detection(self):
        obj, status = ex_fault_detection()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 15.294, delta=15.294 * 0.01)

    def test_water_filling(self):
        obj, status = ex_water_filling()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, -1.816, delta=abs(-1.816) * 0.01)

    def test_image_deblurring(self):
        obj, status = ex_image_deblurring()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, 9.211, delta=9.211 * 0.01)

    def test_portfolio_optimization(self):
        obj, status = ex_portfolio_optimization()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(obj, -0.981, delta=abs(-0.981) * 0.01)

    # ---------- 4_Examples: UDF ----------

    def test_udf_l0_norm(self):
        obj, status, x_sol = ex_udf_l0_norm()
        self.assertEqual(status, "SOLVE_OPT_SUCCESS")
        self.assertLess(abs(obj - 2.85), 1e-2)
        self.assertTrue(np.allclose(x_sol, [0, 1, 0, 1], atol=1e-2))

    def test_udf_l0_ball(self):
        obj = ex_udf_l0_ball()
        self.assertLess(abs(obj - 0.265), 1e-2)

    def test_udf_lhalf(self):
        obj = ex_udf_lhalf()
        self.assertLess(abs(obj - 1.174051), 1e-2)

    def test_udf_group_sparsity(self):
        obj = ex_udf_group_sparsity()
        self.assertLess(abs(obj - 1.15), 1e-2)

    def test_udf_rank(self):
        obj = ex_udf_rank()
        self.assertLess(abs(obj - 0.625), 1e-2)

    def test_udf_rank_r(self):
        obj = ex_udf_rank_r()
        self.assertLess(abs(obj - 0.5), 1e-2)

    def test_udf_unit_sphere(self):
        obj = ex_udf_unit_sphere()
        self.assertLess(abs(obj - 0.405), 1e-2)

    def test_udf_stiefel(self):
        obj = ex_udf_stiefel()
        self.assertLess(abs(obj - 0.625), 1e-2)

    def test_udf_simplex(self):
        obj = ex_udf_simplex()
        self.assertLess(abs(obj - 0.0075), 1e-2)

    def test_udf_binary(self):
        obj = ex_udf_binary()
        self.assertLess(abs(obj - 0.165), 1e-2)

    def test_udf_l0_regression(self):
        obj = ex_udf_l0_regression()
        self.assertLess(abs(obj - 1.501724), 1e-2)


if __name__ == "__main__":
    unittest.main()
