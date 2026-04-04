import admm
import math
import numpy as np
import scipy
import scipy.stats as st
import os
import unittest

from scipy.special import xlogy

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _data_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


class ASTProblemTestCase(unittest.TestCase):

    # min   ||A x - b||_2^2
    def test_Least_Squares(self):
        m = 2
        n = 2
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        model = admm.Model()
        x = admm.Var(n)
        cost = (A @ x - b).T @ (A @ x - b)
        model.setObjective(cost)

        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   f^T x
    # s.t.  F x = g
    #       ||A[i] x + b[i]||_2 <= c[i]^T x + d[i],   i = 1, ..., m
    def test_SOCP(self):
        m = 3
        n = 10
        p = 5
        n_i = 5
        np.random.seed(2)
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

        model = admm.Model()
        x = admm.Var("x", n, 1)
        model.setObjective(f.T @ x)
        model.addConstr(F @ x == g.reshape(p, 1))
        for i in range(m):
            model.addConstr(admm.norm(A[i] @ x + b[i], ord=2) <= c[i].T @ x + d[i])

        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   c^T x
    # s.t.  A x <= b
    def test_LP(self):
        m = 15
        n = 10
        np.random.seed(1)
        s0 = np.random.randn(m)
        lamb0 = np.maximum(-s0, 0)
        s0 = np.maximum(s0, 0)
        x0 = np.random.randn(n)
        A = np.random.randn(m, n)
        b = A @ x0 + s0
        c = -A.T @ lamb0

        model = admm.Model()
        x = admm.Var(n)
        model.setObjective(c.T @ x)
        model.addConstr(A @ x <= b)

        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-3)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum_i  max(-R[i] * P[i,:]^T D[i,:], -B[i])
    # s.t.  D >= 0,   D^T 1 <= T,   D 1 >= c
    #
    # where  D in R^(m x n) is the ad display allocation matrix
    def test_AD(self):
        np.random.seed(1)
        m = 5
        n = 24
        SCALE = 10000
        B = np.random.lognormal(mean=8, size=(m, 1)) + 10000
        B = 1000 * np.round(B / 1000)

        P_ad = np.random.uniform(size=(m, 1))
        P_time = np.random.uniform(size=(1, n))
        P = P_ad.dot(P_time)

        T = np.sin(np.linspace(-2 * np.pi / 2, 2 * np.pi - 2 * np.pi / 2, n)) * SCALE
        T += -np.min(T) + SCALE
        c = np.random.uniform(size=(m,))
        c *= 0.6 * T.sum() / c.sum()
        c = 1000 * np.round(c / 1000)
        R = np.array([np.random.lognormal(c.min() / c[i]) for i in range(m)])

        model = admm.Model()
        D = admm.Var("D", m, n)
        Si = [admm.maximum(-R[i] * P[i, :] @ D[i, :].T, -B[i]) for i in range(m)]
        sum_Si = 0
        for i in range(m):
            sum_Si += Si[i]
        model.setObjective(admm.sum(sum_Si))
        model.addConstr(D >= 0)
        model.addConstr(D.T @ np.ones(m) <= T)
        model.addConstr(D @ np.ones(n) >= c)

        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   (1/m) sum_i log(1 + exp(-y_i * x_i^T theta)) + 0.5 * ||theta[:k]||_1
    #
    # where  x_i are quadratic feature pairs of binary inputs z_i
    def test_Model_Fitting(self):
        def pairs(Z):
            m, n = Z.shape
            k = n * (n + 1) // 2
            X = np.zeros((m, k))
            count = 0
            for i in range(n):
                for j in range(i, n):
                    X[:, count] = Z[:, i] * Z[:, j]
                    count += 1
            return X

        np.random.seed(1)
        n = 10
        k = n * (n + 1) // 2
        m = 200
        sigma = 1.9
        DENSITY = 1.0
        theta_true = np.random.randn(n, 1)
        idxs = np.random.choice(range(n), int((1 - DENSITY) * n), replace=False)
        for idx in idxs:
            theta_true[idx] = 0

        Z = np.random.binomial(1, 0.5, size=(m, n))
        Y = np.sign(Z.dot(theta_true) + np.random.normal(0, sigma, size=(m, 1)))
        X = pairs(Z)
        X = np.hstack([X, np.ones((m, 1))])

        model = admm.Model()
        theta = admm.Var(k + 1, 1)
        loss = admm.sum(admm.logistic(-Y * X @ theta, 1))
        reg = admm.norm(theta[:k], ord=1)

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.setObjective(loss * (1 / m) + 0.5 * reg)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x||_1
    # s.t.  A x = b
    def test_Sparse_solution_L1(self):
        np.random.seed(1)
        m = 60
        n = 100
        s = 25
        A = np.random.randn(m, n)
        x0 = np.zeros(n)
        nonzero_indices = np.random.choice(
            n, size=s, replace=False
        )
        x0[nonzero_indices] = np.random.randn(
            s
        )
        b = A.dot(x0)
        model = admm.Model()
        x_l1 = admm.Var(n)
        model.addConstr(A @ x_l1 == b)
        model.setObjective(admm.norm(x_l1, ord=1))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   W^T |x|
    # s.t.  A x = b
    def test_Sparse_solution_L1_Weighted(self):
        np.random.seed(1)
        m = 60
        n = 100
        s = 25
        A = np.random.randn(m, n)
        x0 = np.zeros(n)
        nonzero_indices = np.random.choice(
            n, size=s, replace=False
        )
        x0[nonzero_indices] = np.random.randn(
            s
        )
        b = A.dot(x0)

        model = admm.Model()
        W = np.ones(n)
        x_log = admm.Var(n)
        model.setObjective(W.T @ admm.abs(x_log))
        model.addConstr(A @ x_log == b)

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||X beta - y||_2^2 + lam * ||beta||_2^2
    def test_Ridge_Regression(self):
        def objective_fn(X, Y, beta, lambd):
            return (X @ beta - Y.reshape(50, 1)).T @ (
                X @ beta - Y.reshape(50, 1)
            ) + lambd * beta.T @ beta

        def generate_data(m=100, n=20, sigma=5):
            np.random.seed(1)
            beta_star = np.random.randn(n)
            X = np.random.randn(m, n)
            Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
            return X, Y

        m = 100
        n = 20
        sigma = 5
        X, Y = generate_data(m, n, sigma)
        X_train = X[:50, :]
        Y_train = Y[:50]

        model = admm.Model()
        beta = admm.Var(n, 1)
        model.setObjective(objective_fn(X_train, Y_train, beta, 0.5))

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   |x + z|
    # s.t.  x + z = 100
    def test_abs_sum(self):
        model = admm.Model()
        x = admm.Var("x")
        z = admm.Var("z")
        model.setObjective(admm.abs(x + z))
        model.addConstr(x + z - 100 == 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x + z
    # s.t.  x = 1,   z = x
    def test_presolved_detect_sum(self):
        model = admm.Model()
        x = admm.Var("x")
        z = admm.Var("z")
        model.setObjective(x + z)
        model.addConstr(x == 1)
        model.addConstr(z - x == 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   exp(x)
    # s.t.  x >= 0
    def test_exp_x(self):
        model = admm.Model()
        x = admm.Var("x")
        model.setObjective(admm.exp(x))
        model.addConstr(x >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   4 + x
    # s.t.  x = 0
    def test_presolved_detect_3(self):
        model = admm.Model()
        x = admm.Var("x")
        model.setObjective(4 + x)
        model.addConstr(x == 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(diag(x))
    # s.t.  x = 2
    def test_presolved_diag_x(self):
        model = admm.Model()
        x = admm.Var("x", 2)
        model.setObjective(admm.sum(admm.diag(x)))
        model.addConstr(x == 2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(diag(X))
    # s.t.  X = 2
    def test_presolved_diag_X(self):
        model = admm.Model()
        X = admm.Var("X", 2, 2)
        model.setObjective(admm.sum(admm.diag(X)))
        model.addConstr(X == 2)
        model.setOption(admm.Options.admm_max_iteration, 100000)

        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(diag(X)) + sum(diag(x))
    # s.t.  X = 2,   x = diag(X)
    def test_diag_Diag(self):
        model = admm.Model()
        X = admm.Var("X", 2, 2)
        x = admm.Var("x", 2)
        model.setObjective(admm.sum(admm.diag(X)) + admm.sum(admm.diag(x)))
        model.addConstr(X == 2)
        model.addConstr(x == admm.diag(X))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(X)
    # s.t.  X - Y Q = 0,   Y = P
    def test_mat_multiply(self):
        model = admm.Model()
        X = admm.Var("X", 2, 2)
        Y = admm.Var("Y", 2, 2)
        P = np.array([[1, 1], [1, 1]])
        Q = np.array([[1, 2], [3, 4]])
        model.setObjective(admm.sum(X))
        model.addConstr(X - Y @ Q == 0)
        model.addConstr(Y == P)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   |x + z + 1|
    # s.t.  x = 2,   z >= 2
    def test_abs_sum_x_z_1(self):
        model = admm.Model()
        x = admm.Var("x")
        z = admm.Var("z")
        model.setObjective(admm.abs(x + z + 1))
        model.addConstr(x == 2)
        model.addConstr(z >= 2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x1^2
    # s.t.  x1 >= 1
    def test_square(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        model.setObjective(admm.square(x1))
        model.addConstr(x1 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   |x1+4| + |x2+2| + |x3+3| + |x4+4|
    # s.t.  x1 = 1,   x2 = 2,   x3 = -3,   x4 = -4
    def test_presolved_detect_L1_norm(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        x3 = admm.Var("x3")
        x4 = admm.Var("x4")
        model.setObjective(
            admm.abs(x1 + 4) + admm.abs(x2 + 2) + admm.abs(x3 + 3) + admm.abs(x4 + 4)
        )
        model.addConstr(x1 == 1)
        model.addConstr(x2 == 2)
        model.addConstr(x3 == -3)
        model.addConstr(x4 == -4)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   exp(x)
    # s.t.  x = z
    def test_minimize_exp_x_over_essentially_unconstraint(self):
        model = admm.Model()
        x = admm.Var("x")
        z = admm.Var("z")
        model.setObjective(admm.exp(x))
        model.addConstr(x == z)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1||_2
    # s.t.  ||[x1; x2; x3]||_2 <= x4,   x2 = 1,   x4 = 1
    def test_norm_pattern_detect(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2)
        x2 = admm.Var("x2")
        x3 = admm.Var("x3", 2)
        x4 = admm.Var("x4")
        model.setObjective(admm.sqrt(x1.T @ x1))
        model.addConstr(admm.sqrt(x1.T @ x1 + x2 * x2 + x3.T @ x3) <= x4)
        model.addConstr(x2 == 1)
        model.addConstr(x4 == 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1 + 1||_2
    # s.t.  ||[x1; x2; x3]||_2 <= 1,   x2 = 1
    def test_norm_pattern_detect0(self):
        model = admm.Model()
        x1 = admm.Var("x1", 1)
        x2 = admm.Var("x2")
        x3 = admm.Var("x3", 2)
        model.setObjective(admm.sqrt((x1 + 1).T @ (x1 + 1)))
        model.addConstr(admm.sqrt(x1.T @ x1 + x2 * x2 + x3.T @ x3) <= 1)
        model.addConstr(x2 == 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1 + 1||_2
    # s.t.  ||[x1; x2; x3]||_2 <= 1,   x2 = 1
    def test_norm_pattern_detect_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2)
        x2 = admm.Var("x2")
        x3 = admm.Var("x3", 2)
        model.setObjective(admm.sqrt((x1 + 1).T @ (x1 + 1)))
        model.addConstr(admm.sqrt(x1.T @ x1 + x2 * x2 + x3.T @ x3) <= 1)
        model.addConstr(x2 == 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||[2x1+1; 3x2+1; 4x3+1]||_2
    # s.t.  x2 = 1
    def test_norm_pattern_detect_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2)
        x2 = admm.Var("x2")
        x3 = admm.Var("x3", 2)
        model.setObjective(
            admm.sqrt(
                (2 * x1 + 1).T @ (2 * x1 + 1)
                + (3 * x2 + 1) * (3 * x2 + 1)
                + (4 * x3 + 1).T @ (4 * x3 + 1)
            )
        )
        model.addConstr(x2 == 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   |x1+4| + |x2+2| + |x3+3| + |x4+4|
    # s.t.  x1 = 1,   x2 = 2
    def test_L1norm_detect(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        x3 = admm.Var("x3")
        x4 = admm.Var("x4")
        model.setObjective(
            admm.abs(x1 + 4) + admm.abs(x2 + 2) + admm.abs(x3 + 3) + admm.abs(x4 + 4)
        )
        model.addConstr(x1 == 1)
        model.addConstr(x2 == 2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum((X+1) * log(X+1)) + 30 + sum(Y)
    # s.t.  X - Y A <= 4,   X >= 2,   Y >= 1
    def test_entropy_detect(self):
        model = admm.Model()
        X = admm.Var("X", 2, 3)
        Y = admm.Var("Y", 2, 2)
        A = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum((X + 1) * admm.log(X + 1)) + 30 + admm.sum(Y))
        model.addConstr(X - Y @ A <= 4)
        model.addConstr(X >= 2)
        model.addConstr(Y >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(log(1 + exp(X+1))) + 30 + sum(Y)
    # s.t.  X - Y A <= 4,   X >= 0,   Y >= 1
    def test_logistic_detect(self):
        model = admm.Model()
        X = admm.Var("X", 2, 3)
        Y = admm.Var("Y", 2, 2)
        A = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.log(1 + admm.exp(X + 1))) + 30 + admm.sum(Y))
        model.addConstr(X - Y @ A <= 4)
        model.addConstr(X >= 0)
        model.addConstr(Y >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(X * log(X / Y))
    # s.t.  X <= 2,   Y <= 1
    def test_kl_div_detect(self):
        model = admm.Model()
        X = admm.Var("X", 2, 3)
        Y = admm.Var("Y", 2, 3)
        model.setObjective(admm.sum(X * admm.log(X / Y)))
        model.addConstr(X <= 2)
        model.addConstr(Y <= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(max(1 - x, 0)^2)
    def test_squared_hinge_detect(self):
        model = admm.Model()
        x = admm.Var("x")
        model.setObjective(admm.sum(admm.square(admm.maximum(1 - x, 0))))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x^2
    # s.t.  -1 <= x <= 1
    def test_square_detect(self):
        model = admm.Model()
        x = admm.Var("x")
        model.setObjective(x * x)
        model.addConstr(x >= -1)
        model.addConstr(x <= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum_t  x_t^2 + u_t^2
    # s.t.  x_{t+1} = x_t + u_t + c_t,   x_0 = 0,   u_t >= -1000,   x_t >= 0
    #
    # where  c_t are fixed sinusoidal constants (2-step horizon)
    def test_ADMMpy_exportModel(self):
        model = admm.Model()
        x0 = admm.Var("x0")
        u0 = admm.Var("u0")
        x1 = admm.Var("x1")
        u1 = admm.Var("u1")
        x2 = admm.Var("x2")
        u2 = admm.Var("u2")
        model.setObjective(
            admm.square(x0)
            + admm.square(u0)
            + admm.square(x1)
            + admm.square(u1)
            + admm.square(x2)
            + admm.square(u2)
        )
        model.addConstr(-x0 - u0 + x1 == 0.841471)
        model.addConstr(-x0 <= 0)
        model.addConstr(-u0 <= 1000)
        model.addConstr(-x1 - u1 + x2 == 0.909297)
        model.addConstr(-x1 <= 0)
        model.addConstr(-u1 <= 1000)
        model.addConstr(x0 == 0)
        model.addConstr(-x2 <= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x||_1
    # s.t.  A x = b
    def test_Basis_Pursuit_m40_n50(self):
        m = 50
        A = np.loadtxt(_data_path("Basis_Pursuit_m40_n50_Param_A.csv"), delimiter=",")
        b = np.loadtxt(_data_path("Basis_Pursuit_m40_n50_Param_b.csv"), delimiter=",")

        model = admm.Model()
        x = admm.Var(m)
        model.setObjective(admm.norm(x, ord=1))
        model.addConstr(A @ x == b)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -x1 - y2
    # s.t.  x1 + 2y2 <= 4,   2x1 - y2 <= 2
    def test_ex1(self):
        model = admm.Model()
        x1 = admm.Var()
        y2 = admm.Var()
        model.setObjective(-x1 - y2)
        model.addConstr(x1 + 2 * y2 <= 4)
        model.addConstr(2 * x1 - y2 <= 2)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 1000)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-2)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-2)
        model.setOption(admm.Options.termination_relative_primal_dual_gap, 1e-2)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -(5x1 + 4x2 + 3x3)
    # s.t.  2x1+3x2+x3 <= 5,   4x1+x2+2x3 <= 11,   3x1+4x2+2x3 <= 8,   x >= 0
    def test_ex2(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        x3 = admm.Var()
        model.setObjective(-(5 * x1 + 4 * x2 + 3 * x3))
        model.addConstr(2 * x1 + 3 * x2 + x3 <= 5)
        model.addConstr(4 * x1 + x2 + 2 * x3 <= 11)
        model.addConstr(3 * x1 + 4 * x2 + 2 * x3 <= 8)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.addConstr(x3 >= 0)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-4)
        model.setOption(admm.Options.termination_relative_error_threshold, 5e-4)
        model.setOption(admm.Options.termination_relative_primal_dual_gap, 5e-4)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum of absolute-value, linear, and quadratic terms in x1,...,x5
    # s.t.  x1+2x2+3x3 = 8,   x1-x4+x5 = 4,   x2-x3+2x5 >= 0,   2x1-3x3+4x4 <= 7
    def test_ex3(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        x3 = admm.Var()
        x4 = admm.Var()
        x5 = admm.Var()
        model.setObjective(
            admm.abs(x1 + 1)
            + x1
            + admm.abs(x2 - 1)
            + 2 * x2
            + 3 * admm.abs(x3 - 2)
            + admm.square(x3)
            + 0.5 * admm.abs(x4 - 3)
            + 1.0 / 3 * admm.square(x4 - 1)
            + 5 * admm.abs(x5 - 1)
            + 0.2 * admm.square(x5 + 1)
        )
        model.addConstr(x1 + 2 * x2 + 3 * x3 == 8)
        model.addConstr(x1 - x4 + x5 == 4)
        model.addConstr(x2 - x3 + 2 * x5 >= 0)
        model.addConstr(2 * x1 - 3 * x3 + 4 * x4 <= 7)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum of absolute-value, linear, and quadratic terms in x[0],...,x[4]
    # s.t.  x[0]+2x[1]+3x[2] = 8,   x[0]-x[3]+x[4] = 4,
    #       x[1]-x[2]+2x[4] >= 0,   2x[0]-3x[2]+4x[3] <= 7
    def test_ex3_vec(self):
        model = admm.Model()
        x = admm.Var(5)
        model.setObjective(
            admm.abs(x[0] + 1)
            + x[0]
            + admm.abs(x[1] - 1)
            + 2 * x[1]
            + 3 * admm.abs(x[2] - 2)
            + admm.square(x[2])
            + 0.5 * admm.abs(x[3] - 3)
            + 1.0 / 3 * admm.square(x[3] - 1)
            + 5 * admm.abs(x[4] - 1)
            + 0.2 * admm.square(x[4] + 1)
        )
        model.addConstr(x[0] + 2 * x[1] + 3 * x[2] == 8)
        model.addConstr(x[0] - x[3] + x[4] == 4)
        model.addConstr(x[1] - x[2] + 2 * x[4] >= 0)
        model.addConstr(2 * x[0] - 3 * x[2] + 4 * x[3] <= 7)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum of logistic and quadratic penalty terms in x1,...,x6
    # s.t.  x1+x2+x3+x4+x5+x6 = 10,   x1+x2-x3-x6 >= 0,   x3+x4-x5 <= 3
    def test_ex4(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        x3 = admm.Var()
        x4 = admm.Var()
        x5 = admm.Var()
        x6 = admm.Var()
        model.setObjective(
            admm.logistic(x1 - 1, 1)
            + admm.square(x1 - 2)
            + admm.logistic(2 * x2 - 1, 1)
            + admm.square(x2 - 1)
            + admm.logistic(x3 - 0.5, 1)
            + admm.square(x3)
            + admm.logistic(3 * x4 - 1)
            + 0.5 * admm.square(x4 + 1)
            + admm.logistic(2 * x5 + 1, 1)
            + admm.square(x5 - 1.0 / 3)
            + admm.logistic(x6 + 0.5, 1)
            + 0.2 * admm.square(x6)
        )
        model.addConstr(x1 + x2 + x3 + x4 + x5 + x6 == 10)
        model.addConstr(x1 + x2 - x3 - x6 >= 0)
        model.addConstr(x3 + x4 - x5 <= 3)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum_i  inrange(x_i, l_i, u_i) + (x_i - m_i)^2
    # s.t.  sum(x) = 12,   x1+2x2-x3 <= 1,   2x4-x5 >= 2
    def test_ex5(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        x3 = admm.Var()
        x4 = admm.Var()
        x5 = admm.Var()
        model.setObjective(
            admm.inrange(x1, 0, 1)
            + admm.square(x1 - 0.5)
            + admm.inrange(x2, 1, 2)
            + admm.square(x2 - 1.5)
            + admm.inrange(x3, 2, 3)
            + admm.square(x3 - 2.5)
            + admm.inrange(x4, 3, 4)
            + admm.square(x4 - 3.5)
            + admm.inrange(x5, 4, 5)
            + admm.square(x5 - 4.5)
        )
        model.addConstr(x1 + x2 + x3 + x4 + x5 == 12)
        model.addConstr(x1 + 2 * x2 - x3 <= 1)
        model.addConstr(2 * x4 - x5 >= 2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum of exponential and quadratic terms in x1,...,x5
    # s.t.  x2+x3+x4+x5 = -10,   x1+x2-x3 <= 8,   x2+x3-x5 >= 6,   x1-x2-x4+x5 <= 6
    def test_ex6(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        x3 = admm.Var()
        x4 = admm.Var()
        x5 = admm.Var()
        model.setObjective(
            admm.exp(x1 - 1)
            + admm.exp(x2 - 2)
            + admm.exp(2 * x3 + 3)
            + 5 * x3
            + admm.exp(x4 - 4)
            + admm.exp(x5 + 1)
            + admm.square(x5)
        )
        model.addConstr(x2 + x3 + x4 + x5 == -10)
        model.addConstr(x1 + x2 - x3 <= 8)
        model.addConstr(x2 + x3 - x5 >= 6)
        model.addConstr(x1 - x2 - x4 + x5 <= 6)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum of exponential and quadratic terms in x1,...,x5
    # s.t.  x2+x3+x4+x5 = -10,   x1+x2-x3 <= 8,   x2+x3-x5 >= 6,   x1-x2-x4+x5 <= 6
    def test_ex70(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        x3 = admm.Var()
        x4 = admm.Var()
        x5 = admm.Var()
        model.setObjective(
            admm.exp(x1 - 1)
            + admm.exp(x2 - 2)
            + admm.exp(2 * x3 + 3)
            + 5 * x3
            + admm.exp(x4 - 4)
            + admm.exp(x5 + 1)
            + admm.square(x5)
        )
        model.addConstr(x2 + x3 + x4 + x5 == -10)
        model.addConstr(x1 + x2 - x3 <= 8)
        model.addConstr(x2 + x3 - x5 >= 6)
        model.addConstr(x1 - x2 - x4 + x5 <= 6)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x||_1
    # s.t.  A x = b
    def test_ex8(self):
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = np.array([1, 3])

        model = admm.Model()
        x = admm.Var(4)
        model.setObjective(admm.norm(x, ord=1))
        model.addConstr(A @ x == b)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||A x - b||_2 + ||x||_1
    def test_ex9_norm2(self):
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = np.array([16, 40])

        model = admm.Model()
        x = admm.Var(4)
        model.setObjective(admm.norm(A @ x - b, ord=2) + admm.norm(x, ord=1))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||A x - b||_2 + 0.01 * ||x||_1
    # s.t.  ||A x||_1 <= 2
    def test_ex10_L2_L1_obj_with_L1_constraint(self):
        A = np.array(
            [[1.165, 0.0751, -0.6965, 0.0591], [0.6268, 0.3516, 1.6961, 1.7971]]
        )
        b = np.array([-0.2995, 1.6440])

        model = admm.Model()
        x = admm.Var(4)
        model.setObjective(admm.norm(A @ x - b, ord=2) + 0.01 * admm.norm(x, ord=1))
        model.addConstr(admm.norm(A @ x, ord=1) <= 2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||A x - b||_2 + 0.01 * ||x||_1
    # s.t.  ||A x||_2 <= 2
    def test_ex10_L2_L1_obj_with_L2_constraint(self):
        A = np.array(
            [[1.165, 0.0751, -0.6965, 0.0591], [0.6268, 0.3516, 1.6961, 1.7971]]
        )
        b = np.array([-0.2995, 1.6440])

        model = admm.Model()
        x = admm.Var(4)
        model.setObjective(admm.norm(A @ x - b, ord=2) + 0.01 * admm.norm(x, ord=1))
        model.addConstr(admm.norm(A @ x, ord=2) <= 2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log(x1 + 1) + x2 + 5
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1 >= 0,   x2 >= 0
    def test_neg_log(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        model.setObjective(-admm.log(x1 + 1) + x2 + 5)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x0[0,0] + x1[0,0] + 5
    # s.t.  x0[0,:] - 2x1[0,:] <= 4,   2x0 + x1 <= 20,   x0, x1 >= 0
    def test_ex12_slicing(self):
        model = admm.Model()
        x0 = admm.Var(2, 2)
        x1 = admm.Var(2, 2)
        model.setObjective(x0[0, 0] + x1[0, 0] + 5)
        model.addConstr(x0[0, :] - 2 * x1[0, :] <= 4)
        model.addConstr(2 * x0 + x1 <= 20)
        model.addConstr(x0 >= 0)
        model.addConstr(x1 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   p1^T x0 + p2^T x1 + 5
    # s.t.  x0 + 2x1 <= 4,   2x0 + x1 <= 2,   x0, x1 >= 0
    def test_ex_0306_vec_product(self):
        model = admm.Model()
        x0 = admm.Var(2)
        x1 = admm.Var(2)
        p1 = np.array([1, 1])
        p2 = np.array([1, 1])
        model.setObjective(p1 @ x0 + p2 @ x1 + 5)
        model.addConstr(x0 + 2 * x1 <= 4)
        model.addConstr(2 * x0 + x1 <= 2)
        model.addConstr(x0 >= 0)
        model.addConstr(x1 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(0.5 * A x^T A + A)
    # s.t.  b x = 0
    def test_ex_matrix_1_product(self):
        model = admm.Model()
        A = np.array([[5, 4], [3, 2], [1, 0]])
        b = np.array([[3, 4, 5], [6, 7, 8], [9, 11, 10]])
        x = admm.Var(3, 2)
        model.setObjective(admm.sum(0.5 * A @ x.T @ A + A))
        model.addConstr(b @ x == 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   a^T x[:,0] + b^T x[:,1]
    # s.t.  c x = 100
    def test_ex_matrix_2_matrix_slicing(self):
        model = admm.Model()
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.array([[3, 4, 5], [6, 7, 8]])
        x = admm.Var(3, 2)
        model.setObjective(a @ x[:, 0] + b @ x[:, 1])
        model.addConstr(c @ x - 100 == 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(p1 x1 - p2 x2 + 5)
    # s.t.  x1 + 2x2 <= 4,   2x1 + x2 <= 2,   x1, x2 >= 0
    def test_ex_matrix_3_matrix_product(self):
        model = admm.Model()
        p1 = np.ones((2, 2))
        p2 = np.ones((2, 2))
        x1 = admm.Var(2, 2)
        x2 = admm.Var(2, 2)
        model.setObjective(admm.sum(p1 @ x1 - p2 @ x2 + 5))
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x[0:2])
    # s.t.  x[0] = 2,   x[1] = 6
    def test_index_ex1(self):
        model = admm.Model()
        x = admm.Var(5)
        model.setObjective(admm.sum(x[0:2]))
        model.addConstr(x[0] == 2)
        model.addConstr(x[1] == 6)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   |x1 + x2 + 1|
    # s.t.  x1-x2 >= 5,   x1+x2 >= 5,   x1, x2 >= 0
    def test_linear_expr_ex1(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        model.setObjective(admm.abs(x1 + x2 + 1))
        model.addConstr(x1 - x2 >= 5)
        model.addConstr(x1 + x2 >= 5)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   c^T x
    # s.t.  A x = b
    def test_LP_boyd_ex2(self):
        model = admm.Model()
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        c = np.array([1, 3, 5, 7])
        b = np.array([1, 3])
        x = admm.Var(4)
        model.setObjective(c.T @ x)
        model.addConstr(A @ x == b)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(max(1 - z, 0)) + 0.5 * ||w||_2^2
    # s.t.  z = y * (A w + b)
    #
    # where  SVM hinge-loss formulation with margin variable z
    def test_mapl_vector(self):
        model = admm.Model()
        A = np.array([[0, 1], [1, 0], [1, 2], [2, 1]])
        y = np.array([-1, -1, 1, 1])
        b = admm.Var()
        w = admm.Var(2)
        z = admm.Var(4)
        model.setObjective(admm.sum(admm.maximum(1 - z, 0)) + 0.5 * w.T @ w)
        model.addConstr(z == y * (A @ w + b))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   a^T x[:,0] + b^T x[:,1]
    # s.t.  c x = d,   x >= -100
    def test_mat_ex1(self):
        model = admm.Model()
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.array([[3, 4, 5], [6, 7, 8]])
        d = np.array([[100, 200], [100, 200]])
        x = admm.Var(3, 2)
        model.setObjective(a @ x[:, 0] + b @ x[:, 1])
        model.addConstr(c @ x == d)
        model.addConstr(x >= -100)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-4)
        model.setOption(admm.Options.termination_relative_error_threshold, 5e-4)
        model.setOption(admm.Options.termination_relative_primal_dual_gap, 5e-4)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log det(X) - log det(Y)
    # s.t.  X + Y = 5I,   X, Y in PSD
    def test_mat_log_det_ex(self):
        model = admm.Model()
        x = admm.Var(3, 3, PSD=True)
        y = admm.Var(3, 3, PSD=True)
        model.setObjective(-admm.log_det(x) - admm.log_det(y))
        model.addConstr(x + y == 5 * np.eye(3))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log det(X) - log det(Y)
    # s.t.  X + Y = 5I,   X, Y in PSD
    def test_mat_log_det_ex1(self):
        model = admm.Model()
        x = admm.Var(3, 3, PSD=True)
        y = admm.Var(3, 3, PSD=True)
        model.setObjective(-admm.log_det(x) - admm.log_det(y))
        model.addConstr(x + y == 5 * np.eye(3))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   max(x1 + 1, 2) + x2 + 5
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1 >= 0,   x2 >= 1
    def test_max_fun_shift_error(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        model.setObjective(admm.maximum(x1 + 1, 2) + x2 + 5)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x[0] + x[1]
    # s.t.  x[0] + 2x[1] = 1,   2x[0] - 3x[1] >= 3,   x[1] >= -2
    def test_mergeConstrex1(self):
        model = admm.Model()
        x = admm.Var(2)
        model.setObjective(x[0] + x[1])
        model.addConstr(x[0] + 2 * x[1] == 1)
        model.addConstr(2 * x[0] - 3 * x[1] >= 3)
        model.addConstr(x[1] >= -2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x||_2
    # s.t.  A1 x <= b1,   A2 x <= b2
    def test_norm2ex1(self):
        model = admm.Model()
        A1 = np.array([[0, 1, 1], [-1, -1, 1]])
        b1 = np.array([0, 1])
        A2 = np.array([[0, 0, -1], [1, 0, -1]])
        b2 = np.array([0, -1])
        x = admm.Var("x", 3)
        model.setObjective(admm.norm(x, ord=2))
        model.addConstr(A1 @ x <= b1)
        model.addConstr(A2 @ x <= b2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -p1^T x1 - p2^T x2 + 3
    # s.t.  p3^T x1 + p4^T x2 <= 1,   p5^T x1 + p6^T x2 <= 2,   x1, x2 >= 0
    def test_pytest_ex2(self):
        model = admm.Model()
        x1 = admm.Var(3)
        x2 = admm.Var(4)
        p1 = np.array([1, 2, 3])
        p2 = np.array([4, 5, 6, 7])
        p3 = np.array([1, 1, 1])
        p4 = np.array([1, 1, 1, 1])
        p5 = np.array([1, 4, 7])
        p6 = np.array([2, 5, 6, 8])
        model.setObjective(-p1 @ x1 - p2 @ x2 + 3)
        model.addConstr(p3 @ x1 + p4 @ x2 <= 1)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.addConstr(p5 @ x1 + p6 @ x2 <= 2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   (x1 + x2)^2
    def test_quadratic_ex1(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        model.setObjective(admm.square(x1 + x2))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1 + x2||_2^2
    def test_quadratic_ex2(self):
        model = admm.Model()
        x1 = admm.Var(2)
        x2 = admm.Var(2)
        model.setObjective(x1.T @ x1 + 2 * x1.T @ x2 + x2.T @ x2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x)
    # s.t.  x[1], x[3] >= 0,   x[0], x[2], x[4] >= 1
    def test_rowset_ex1(self):
        model = admm.Model()
        x = admm.Var(5)
        model.setObjective(admm.sum(x))
        model.addConstr(x[1] >= 0)
        model.addConstr(x[3] >= 0)
        model.addConstr(x[0] >= 1)
        model.addConstr(x[2] >= 1)
        model.addConstr(x[4] >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -x - y
    # s.t.  x + 2y <= 4,   2x - y <= 2
    def test_SCALAR_kIdentity_0(self):
        model = admm.Model()
        x = admm.Var()
        y = admm.Var()
        model.setObjective(-x - y)
        model.addConstr(x + 2 * y <= 4)
        model.addConstr(2 * x - y <= 2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   a^T x[:,0] + b^T x[:,1]
    # s.t.  c x = 100
    def test_singleX_ex1(self):
        model = admm.Model()
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.array([[3, 4, 5], [1.2, 33, 100], [6, 7, 8]])
        x = admm.Var(3, 2)
        model.setObjective(a @ x[:, 0] + b @ x[:, 1])
        model.addConstr(c @ x == 100)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log det(X) + tr(S X) + mu * sum(|X|)
    # s.t.  X in PSD
    def test_Sparse_Inverse_Covariance_Selection_n3(self):
        model = admm.Model()
        S = np.array(
            [
                [2.57672, 0.70395, -0.314030],
                [0.70395, 3.62516, -0.45451],
                [-0.31403, -0.454517, 4.10943],
            ]
        )
        mu = 0.1
        x = admm.Var(3, 3, PSD=True)
        model.setObjective(
            -admm.log_det(x) + admm.trace(S @ x) + mu * admm.sum(admm.abs(x))
        )
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(max(1 - z, 0)) + 0.5 * ||w||_2^2
    # s.t.  z = y * (A w + b)
    def test_svm1(self):
        model = admm.Model()
        A = np.array([[0, 1], [1, 2], [1, 0], [2, 1]])
        y = np.array([-1, -1, 1, 1])
        w = admm.Var(2)
        b = admm.Var()
        z = admm.Var(4)
        model.setObjective(admm.sum(admm.maximum(1 - z, 0)) + 0.5 * w.T @ w)
        model.addConstr(z == y * (A @ w + b))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(max(1 - z, 0)) + 0.5 * ||w||_2^2
    # s.t.  z = y * (A w + b)
    def test_svm2(self):
        model = admm.Model()
        A = np.loadtxt(_data_path("data_4_A.csv"), delimiter=",")
        y = np.loadtxt(_data_path("data_4_y.csv"), delimiter=",")
        w = admm.Var(2)
        b = admm.Var()
        z = admm.Var(4)
        model.setObjective(admm.sum(admm.maximum(1 - z, 0)) + 0.5 * w.T @ w)
        model.addConstr(z == y * (A @ w + b))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   tr(X) + tr(Y)
    # s.t.  X = 2,   Y >= 1
    def test_trace_test(self):
        model = admm.Model()
        x = admm.Var(3, 3)
        y = admm.Var(2, 2)
        model.setObjective(admm.trace(x) + admm.trace(y))
        model.addConstr(x == 2)
        model.addConstr(y >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(max(1 - z, 0)) + 0.5 * ||w||_2^2
    # s.t.  z = y * (A w + b)
    def test_VecSVMdump(self):
        model = admm.Model()
        A = np.array(
            [
                [-0.017612, 14.0531],
                [0.423363, 11.0547],
                [0.667394, 12.7415],
                [-0.026632, 10.4277],
                [1.34718, 13.1755],
                [1.17681, 3.16702],
                [0.931635, 1.5895],
                [-0.036453, 2.69099],
                [-0.196949, 0.444165],
                [-1.69345, -0.55754],
            ]
        )
        y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
        w = admm.Var(2)
        b = admm.Var()
        z = admm.Var(10)
        model.setObjective(admm.sum(admm.maximum(1 - z, 0)) + 0.5 * w.T @ w)
        model.addConstr(z == y * (A @ w + b))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log(2x1 + 1) + x2*log(x2) + 0.5*x2^2
    # s.t.  3x1+5x2 = 30,   x1-x2 >= 5,   2x1+1 > 0,   x2 > 0
    def test_wbh03(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        model.setObjective(
            -admm.log(2 * x1 + 1) + admm.entropy(x2) + 0.5 * admm.square(x2)
        )
        model.addConstr(3 * x1 + 5 * x2 == 30)
        model.addConstr(x1 - x2 >= 5)
        model.addConstr(2 * x1 + 1 >= 0.0001)
        model.addConstr(x2 >= 0.0001)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   log(1+exp(2x1+1)) + x1^2 + |2x2-5| + 0.5*x2^2
    # s.t.  2x1+3x2 = 4,   x1-2x2 <= 10
    def test_wbh04(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        model.setObjective(
            admm.logistic(2 * x1 + 1, 1)
            + admm.square(x1)
            + admm.abs(2 * x2 - 5)
            + 0.5 * admm.square(x2)
        )
        model.addConstr(2 * x1 + 3 * x2 == 4)
        model.addConstr(x1 - 2 * x2 <= 10)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   |x1+1|+x1 + |x2-1|+2x2^2 + 3|x3-2|+x3^2 + 0.5|x4-3|+(1/3)(x4-1)^2 + 5|x5-1|+0.2(x5+1)^2
    # s.t.  x1+2x2+3x3 = 8,   x1-x4+x5 = 4,   x2-x3+2x5 >= 0,   2x1-3x3+4x4 <= 7
    def test_wbh13(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        x3 = admm.Var()
        x4 = admm.Var()
        x5 = admm.Var()
        model.setObjective(
            admm.abs(x1 + 1)
            + x1
            + admm.abs(x2 - 1)
            + 2 * admm.square(x2)
            + 3 * admm.abs(x3 - 2)
            + admm.square(x3)
            + 0.5 * admm.abs(x4 - 3)
            + 1.0 / 3 * admm.square(x4 - 1)
            + 5 * admm.abs(x5 - 1)
            + 0.2 * admm.square(x5 + 1)
        )
        model.addConstr(x1 + 2 * x2 + 3 * x3 == 8)
        model.addConstr(x1 - x4 + x5 == 4)
        model.addConstr(x2 - x3 + 2 * x5 >= 0)
        model.addConstr(2 * x1 - 3 * x3 + 4 * x4 <= 7)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   logistic(x1-1)+(x1-2)^2 + logistic(2x2-1)+(x2-1)^2 + logistic(x3-0.5)+x3^2
    #     + logistic(3x4-1)+0.5(x4+1)^2 + logistic(2x5+1)+(x5-1/3)^2 + logistic(x6+0.5)+0.2x6^2
    # s.t.  x1+x2+x3+x4+x5+x6 = 10,   x1+x2-x3-x6 >= 0,   x3+x4-x5 <= 3
    def test_wbh14(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        x3 = admm.Var()
        x4 = admm.Var()
        x5 = admm.Var()
        x6 = admm.Var()
        model.setObjective(
            admm.logistic(x1 - 1, 1)
            + admm.square(x1 - 2)
            + admm.logistic(2 * x2 - 1)
            + admm.square(x2 - 1)
            + admm.logistic(x3 - 0.5, 1)
            + admm.square(x3)
            + admm.logistic(3 * x4 - 1)
            + 0.5 * admm.square(x4 + 1)
            + admm.logistic(2 * x5 + 1, 1)
            + admm.square(x5 - 1.0 / 3)
            + admm.logistic(x6 + 0.5, 1)
            + 0.2 * admm.square(x6)
        )
        model.addConstr(x1 + x2 + x3 + x4 + x5 + x6 == 10)
        model.addConstr(x1 + x2 - x3 - x6 >= 0)
        model.addConstr(x3 + x4 - x5 <= 3)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log(x1)+x1 - log(x2-1) - log(2x3-1) + 2(x3-1)^2 + entropy(x4-1) + (x4-2)^2 + entropy(x5) + 2x5
    # s.t.  x1+2x2+3x3 = 5,   x1+x4-x5 = 0,   x2-x3+x4 <= 6,   x2+x5 <= 4
    def test_wbh15(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        x3 = admm.Var()
        x4 = admm.Var()
        x5 = admm.Var()
        model.setObjective(
            -admm.log(x1)
            + x1
            - admm.log(x2 - 1)
            - admm.log(2 * x3 - 1)
            + 2 * admm.square(x3 - 1)
            + admm.entropy(x4 - 1)
            + admm.square(x4 - 2)
            + admm.entropy(x5)
            + 2 * x5
        )
        model.addConstr(x1 + 2 * x2 + 3 * x3 == 5)
        model.addConstr(x1 + x4 - x5 == 0)
        model.addConstr(x2 - x3 + x4 <= 6)
        model.addConstr(x2 + x5 <= 4)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log(x1) - log(x2)+(x2-1)^2 - log(x3)+(x3-2)^2 - log(x4)+x4^2 - log(x5)+0.5x5^2 - log(x6)
    # s.t.  x1+x2+x3-x4 = 5,   x2+x3+x4 = 4,   x3+x5-x6 >= 5,   x1+x2+2x3 >= 4,   x4+x5+x6 >= 3
    def test_wbh17(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        x3 = admm.Var()
        x4 = admm.Var()
        x5 = admm.Var()
        x6 = admm.Var()
        model.setObjective(
            -admm.log(x1)
            - admm.log(x2)
            + admm.square(x2 - 1)
            - admm.log(x3)
            + admm.square(x3 - 2)
            - admm.log(x4)
            + admm.square(x4)
            - admm.log(x5)
            + 0.5 * admm.square(x5)
            - admm.log(x6)
        )
        model.addConstr(x1 + x2 + x3 - x4 == 5)
        model.addConstr(x2 + x3 + x4 == 4)
        model.addConstr(x3 + x5 - x6 >= 5)
        model.addConstr(x1 + x2 + 2 * x3 >= 4)
        model.addConstr(x4 + x5 + x6 >= 3)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log(x1)
    # s.t.  0.5 <= x1 <= 1
    def test_wbh17_NegLog(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        model.setObjective(-admm.log(x1))
        model.addConstr(x1 <= 1)
        model.addConstr(x1 >= 0.5)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x1^2
    # s.t.  x1 >= 1
    def test_wbh17_Square(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        model.setObjective(admm.square(x1))
        model.addConstr(x1 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum_i  a_i * (x_i - c_i)^2
    # s.t.  sum(x) = 12,   x1+x2-x3 >= 3,   x4+x5-x6 >= 5,   x4-x5+x6 <= 8
    def test_wbh18(self):
        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        x3 = admm.Var()
        x4 = admm.Var()
        x5 = admm.Var()
        x6 = admm.Var()
        model.setObjective(
            admm.square(x1 - 1)
            + 0.5 * admm.square(x2 - 2)
            + 1.0 / 3 * admm.square(x3 - 3)
            + 0.25 * admm.square(x4 - 4)
            + 0.2 * admm.square(x5 - 5)
            + 1.0 / 6 * admm.square(x6 - 6)
        )
        model.addConstr(x1 + x2 + x3 + x4 + x5 + x6 == 12)
        model.addConstr(x1 + x2 - x3 >= 3)
        model.addConstr(x4 + x5 - x6 >= 5)
        model.addConstr(x4 - x5 + x6 <= 8)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -x1 - x2
    # s.t.  x1 + 2x2 <= 4,   2x1 - x2 <= 2
    def test_scalar(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        p = 4
        model.setObjective(-x1 - x2)
        model.addConstr(x1 + 2 * x2 <= p)
        model.addConstr(2 * x1 - x2 <= 2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # max   x1 + x2 + 100
    # s.t.  x1 + 2x2 <= 4,   2x1 - x2 <= 2
    def test_scalar_max(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        p = 4
        model.setObjective(x1 + x2 + 100)
        model.ModelSense = -1
        model.addConstr(x1 + 2 * x2 <= p)
        model.addConstr(2 * x1 - x2 <= 2)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   p1^T x1 + p2^T x2 + 5
    # s.t.  p1^T x1 + 2*p2^T x2 <= 4,   2*p1^T x1 + p2^T x2 <= 2,   x1, x2 >= 0
    def test_vector(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 1)
        x2 = admm.Var("x2", 3, 1)
        p1 = np.ones((1, 2))
        p2 = np.ones((1, 3))

        model.setObjective(p1 @ x1 + p2 @ x2 + 5)
        model.addConstr(p1 @ x1 + 2 * p2 @ x2 <= 4)
        model.addConstr(2 * p1 @ x1 + p2 @ x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(p1 x1 - p2 x2 + 5)
    # s.t.  x1 + 2x2 <= 4,   2x1 + x2 <= 2,   x1, x2 >= 0
    def test_matrix(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2)
        x2 = admm.Var("x2", 2, 2)
        p1 = np.ones((2, 2))
        p2 = np.ones((2, 2))
        model.setObjective(admm.sum(p1 @ x1 - p2 @ x2 + 5))
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x1 p1 - x2 + 5)
    # s.t.  x1 + x2 p2 <= 4,   x1, x2 >= 0
    def test_matrix_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p1 = np.ones((3, 2))
        p2 = np.ones((2, 3))
        model.setObjective(admm.sum(x1 @ p1 - x2 + 5))
        model.addConstr(x1 + x2 @ p2 <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   p1[0,0]*x1[0,0] + p2[0,0]*x2[0,0] + 5
    # s.t.  x1[0,:] - 2x2[0,:] <= 4,   2x1 + x2 <= 20,   x1 >= 0,   x2 >= 1
    def test_slice_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2)
        x2 = admm.Var("x2", 2, 2)
        p1 = np.ones((2, 2))
        p2 = np.ones((2, 2))

        model.setObjective(p1[0, 0] * x1[0, 0] + p2[0, 0] * x2[0, 0] + 5)
        model.addConstr(x1[0, :] - 2 * x2[0, :] <= 4)
        model.addConstr(2 * x1 + x2 <= 20)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   exp(x1 + 1) + x2 + 5
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1 >= 0,   x2 >= 1
    def test_exp(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")

        model.setObjective(admm.exp(x1 + 1) + x2 + 5)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(exp(x1 + 1)) + 30 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 0,   x2 >= 1
    def test_exp_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.exp(x1 + 1)) + 30 + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1 + 3*x2 + 1||_2
    def test_no_constr(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2)
        x2 = admm.Var("x2", 2)
        model.setObjective(admm.norm(x1 + 3 * x2 + 1, ord=2))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   |x1 + 1| + x2 + p
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1 >= 0,   x2 >= 1
    def test_abs(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        p = 5
        model.setObjective(admm.abs(x1 + 1) + x2 + p)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(|x1 + 1|) + 30 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 2,   x2 >= 1
    def test_abs_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.abs(x1 + 1)) + 5 * 6 + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   |x1 + 1| + x2 + 2
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1 >= 0,   x2 >= 1
    def test_abs3(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        model.setObjective(admm.abs(x1 + 1) + x2 + 2)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1 + 1||_1
    # s.t.  x1 - (x2 p)^T <= 4,   x1 >= 0,   x2 >= 1
    def test_norm1_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3)
        x2 = admm.Var("x2", 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.norm(x1 + 1, ord=1))
        model.addConstr(x1 - (x2 @ p).T <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1 + 1||_F + 5 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 2,   x2 >= 1
    def test_norm_fro(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.norm(x1 + 1, ord="fro") + 5) + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1 + 1||_* + 5 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 2,   x2 >= 1
    #
    # where  ||·||_* denotes the nuclear norm (sum of singular values)
    def test_norm_nuc(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.norm(x1 + 1, ord="nuc") + 5) + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||A - Y x||_F
    # s.t.  x >= 0
    def test_norm_fro_2(self):
        model = admm.Model()
        x = admm.Var("x", 2, 3)
        Y = np.array([[1, 2], [1, 2], [1, 2]])
        A = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])
        model.setObjective(admm.norm(A - Y @ x, "fro"))
        model.addConstr(x >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x1 + x2)
    # s.t.  x2 >= 1,   ||x1||_inf <= 1
    def test_linf(self):
        model = admm.Model()
        x1 = admm.Var("x1", 1)
        x2 = admm.Var("x2", 1)
        model.setObjective(admm.sum(x1 + x2))
        model.addConstr(x2 >= 1)
        model.addConstr(admm.norm(x1, ord=np.inf) <= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x1 + x2)
    # s.t.  x2 >= 1,   ||x1||_inf <= 1
    def test_linf_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2)
        x2 = admm.Var("x2", 2)
        model.setObjective(admm.sum(x1 + x2))
        model.addConstr(x2 >= 1)
        model.addConstr(admm.norm(x1, ord=admm.inf) <= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x1 + x2)
    # s.t.  x2 >= 1,   |x1| <= 1
    def test_linf_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2)
        x2 = admm.Var("x2", 2)
        model.setObjective(admm.sum(x1 + x2))
        model.addConstr(x2 >= 1)
        model.addConstr(admm.abs(x1) <= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1 + 1||_inf
    # s.t.  x1 - (x2 p)^T <= 4,   x1 >= 0,   x2 >= 1
    def test_linf_3(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3)
        x2 = admm.Var("x2", 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.norm(x1 + 1, ord=np.inf))
        model.addConstr(x1 - (x2 @ p).T <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1 + 1||_2
    # s.t.  x1 - (x2 p)^T <= 4,   x1 >= 2,   x2 >= 1
    def test_norm2_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, 1)
        x2 = admm.Var("x2", 1, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.norm(x1 + 1, ord=2))
        model.addConstr(x1 - (x2 @ p).T <= 4)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   max(x1 + 1, 2) + x2 + p
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1 >= 0,   x2 >= 1
    def test_maximum(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        p = 5
        model.setObjective(admm.maximum(x1 + 1, 2) + x2 + p)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -min(x1 + 1, 2) + x2 + p
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1 >= 0,   x2 >= 1
    def test_minimum(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        p = 5
        model.setObjective(-admm.minimum(x1 + 1, 2) + x2 + p)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(max(x1 + 1, 8))
    # s.t.  x1 - p^T x2 <= 4,   x1 >= 0,   x2 >= 1
    def test_maximum_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3)
        x2 = admm.Var("x2", 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.maximum(x1 + 1, 8)))
        model.addConstr(x1 - p.T @ x2 <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(-min(x1 + 1, 8))
    # s.t.  x1 - p^T x2 <= 4,   x1 >= 0,   x2 >= 1
    def test_minimum_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3)
        x2 = admm.Var("x2", 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(-admm.minimum(x1 + 1, 8)))
        model.addConstr(x1 - p.T @ x2 <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(max_i(x1 + 1))
    # s.t.  x1 - p^T x2 <= 4,   x1 >= 0,   x2 >= 1
    def test_max_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3)
        x2 = admm.Var("x2", 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.max(x1 + 1)))
        model.addConstr(x1 - p.T @ x2 <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   max_i(-(x1 + 1))
    # s.t.  x1 - p^T x2 <= 4,   x1 <= 0,   x2 <= 1
    def test_min_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3)
        x2 = admm.Var("x2", 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.max(-(x1 + 1)))
        model.addConstr(x1 - p.T @ x2 <= 4)
        model.addConstr(x1 <= 0)
        model.addConstr(x2 <= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(max_ij(x1 + 1)) + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 0,   x2 >= 1
    def test_max_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.max(x1 + 1)) + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   max_ij(-(x1 + 1)) + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 <= 0,   x2 <= 1
    def test_min_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.max(-(x1 + 1)) + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 <= 0)
        model.addConstr(x2 <= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-2)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum((x1+1)*log(x1+1)) + 30 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 2,   x2 >= 1
    def test_entropy_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.entropy(x1 + 1)) + 30 + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   huber(x1 + 1, 1) + x2 + 5
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1 >= 0,   x2 >= 1
    def test_huber(self):
        model = admm.Model()
        x1 = admm.Var("x1", 1)
        x2 = admm.Var("x2", 1)
        model.setObjective(admm.huber(x1 + 1, 1) + x2 + 5)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(huber(x1 + 1, 1)) + 30 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 2,   x2 >= 1
    def test_huber_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.huber(x1 + 1, 1)) + 30 + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(huber(x1, 2)) + sum(x2)
    # s.t.  x1 + x2 <= 6,   x1 >= 2,   x2 >= 1
    def test_huber_3(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2)
        x2 = admm.Var("x2", 2, 2)
        model.setObjective(admm.sum(admm.huber(x1, 2)) + admm.sum(x2))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(huber(x1 + 1)) + 30 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 2,   x2 >= 1
    def test_huber_default_w(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.huber(x1 + 1)) + 30 + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   (x1 + 1)^2 + x2 + 5
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1 >= 0,   x2 >= 1
    def test_square_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 1)
        x2 = admm.Var("x2", 1)
        model.setObjective(admm.square(x1 + 1) + x2 + 5)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum((x1 + 1)^2)
    # s.t.  x1 - p^T x2 <= 4,   x1 >= 0,   x2 >= 1
    def test_square_3(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3)
        x2 = admm.Var("x2", 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.square(x1 + 1)))
        model.addConstr(x1 - p.T @ x2 <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log(x1 + 1) + x2 + p
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1, x2 >= 0
    def test_log_1(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        p = 5
        model.setObjective(-admm.log(x1 + 1) + x2 + p)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(-log(x1 + 1)) + 30 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 0,   x2 >= 1
    def test_log_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(-admm.log(x1 + 1)) + 30 + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   log(1 + exp(x1 + 1)) + x2 + p
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1, x2 >= 0
    def test_logistic(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        p = 5
        model.setObjective(admm.logistic(x1 + 1, 1) + x2 + p)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   log(1 + exp(x1 + 1)) + x2 + 5
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1, x2 >= 0
    def test_logistic_1(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        model.setObjective(admm.logistic(x1 + 1, 1) + x2 + 5)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   (x1 + 1)^2 + x2 + p
    # s.t.  x1+2x2 <= 4,   2x1+x2 <= 2,   x1 >= 1,   x2 >= 0
    def test_power(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        p = 5
        model.setObjective(admm.power(x1 + 1, 2) + x2 + p)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 1)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum((x1 + 1)^2) + 30 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 0,   x2 >= 1
    def test_power_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.power(x1 + 1, 2)) + 30 + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(-sqrt(x1))
    # s.t.  x1 + x2 <= 6,   x1 >= 2,   x2 >= 1
    def test_sqrt_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2)
        x2 = admm.Var("x2", 2, 2)
        model.setObjective(admm.sum(-admm.sqrt(x1)))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   tr(x1)
    # s.t.  x1 + x2 <= 6,   x1 >= 2,   x2 >= 1
    def test_trace(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2)
        x2 = admm.Var("x2", 2, 2)
        model.setObjective(admm.trace(x1))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   tr(x1) + sum(x2)
    # s.t.  x1 + x2 <= 6,   x1 >= 2,   x2 >= 1
    def test_trace_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, 3)
        x2 = admm.Var("x2", 3, 3)
        model.setObjective(admm.trace(x1) + admm.sum(x2))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(kl_div(x, y))
    # s.t.  x <= 2,   y <= 1
    #
    # where  kl_div(x, y) = x*log(x/y) - x + y
    def test_kl_div_1(self):
        model = admm.Model()
        x = admm.Var("x", 2, 3)
        y = admm.Var("y", 2, 3)
        model.setObjective(admm.sum(admm.kl_div(x, y)))
        model.addConstr(x <= 2)
        model.addConstr(y <= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   tr(C X)
    # s.t.  tr(A[i] X) = b[i],   i = 1,...,p,   X in PSD
    def test_PSD_0(self):
        n = 3
        p = 3
        np.random.seed(1)
        C = np.random.randn(n, n)

        A = []
        b = []
        for i in range(p):
            A.append(np.random.randn(n, n))
            b.append(np.random.randn())

        C = 0.5 * (C + C.T)
        A = [0.5 * (Ai + Ai.T) for Ai in A]

        model = admm.Model()
        X = admm.Var("X", n, n, PSD=True)
        for i in range(p):
            model.addConstr(admm.trace(A[i] @ X) == b[i])
        model.setObjective(admm.trace(C @ X))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   tr(C X)
    # s.t.  tr(A X) = b,   X in PSD
    def test_PSD_01(self):
        n = 2
        C = np.array([[1, 2], [3, 4]])
        A = np.array([[1, 2], [3, 4]])
        b = 100

        model = admm.Model()
        X = admm.Var("X", n, n, PSD=True)
        model.addConstr(admm.trace(A @ X) == b)
        model.setObjective(admm.trace(C @ X))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x1) + sum(x2)
    # s.t.  x1 + x2 <= 6,   x1[0,2] >= 2,   x1 >= 1,   x2 >= 1,   x1 in PSD
    def test_PSD_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, 3, PSD=True)
        x2 = admm.Var("x2", 3, 3)
        model.setObjective(admm.sum(x1) + admm.sum(x2))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x1[0, 2] >= 2)
        model.addConstr(x1 >= 1)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x1) + sum(x2)
    # s.t.  x1 + x2 <= 6,   x1[0,2] >= 2,   x1 >= 1,   x2 >= 1,   x1 symmetric
    def test_symmetric_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, 3, symmetric=True)
        x2 = admm.Var("x2", 3, 3)
        model.setObjective(admm.sum(x1) + admm.sum(x2))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x1[0, 2] >= 2)
        model.addConstr(x1 >= 1)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   f^T x
    # s.t.  F x = g,   ||A*x + b||_2 <= c^T x + d
    def test_SOC_1(self):
        n = 2
        m = 3
        F = np.ones((m, n))
        f = np.ones(n)
        g = 2 * np.ones(m)
        A, b, d = 2, 3, 6
        c = 5 * np.ones(n)

        model = admm.Model()
        x = admm.Var("x", n)
        model.setObjective(f.T @ x)
        model.addConstr(F @ x == g)
        model.addConstr(admm.norm(A * x + b, ord=2) <= c.T @ x + d)

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   5 * f^T x
    # s.t.  F x = g,   3*||A*x + b||_2 <= 2*c^T x + d
    def test_SOC_2(self):
        n = 2
        m = 3
        F = np.ones((m, n))
        f = np.ones(n)
        g = 2 * np.ones(m)
        A, b, d = 2, 3, 6
        c = 5 * np.ones(n)

        model = admm.Model()
        x = admm.Var("x", n)
        model.setObjective(5 * f.T @ x)
        model.addConstr(F @ x == g)
        model.addConstr(3 * admm.norm(A * x + b, ord=2) <= 2 * c.T @ x + d)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   0.1 * tv2d(U) + 0.9 * ||conv(U, K) - b||_F^2
    #
    # where  b = conv(image, K) is a blurred observation, U is the recovered image
    def test_conv2d(self):
        np.random.seed(1)
        height = 40
        width = 50
        image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8) / 255
        kernel = np.array(
            [[1 / 9, 2 / 9, 1 / 9], [2 / 9, 4 / 9, 2 / 9], [1 / 9, 2 / 9, 1 / 9]]
        )
        image_blur = admm.conv2d(image, kernel, "same")

        model = admm.Model()
        U = admm.Var("U", image.shape)
        tv = admm.tv2d(U, p=1)
        data_fidelity = admm.conv2d(U, kernel, "same") - image_blur
        model.setObjective(0.1 * tv + 0.9 * admm.sum(admm.square(data_fidelity)))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   tv2d(U)
    # s.t.  U * mask = image * mask
    def test_TV_inpainting(self):
        np.random.seed(1)
        height = 4
        width = 4
        image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8) / 255.0
        index_set = np.random.rand(height, width)
        a = 0.6
        index_set[index_set >= a] = 1
        index_set[index_set < a] = 1

        model = admm.Model()
        U = admm.Var("U", image.shape)
        model.setObjective(admm.tv2d(U, p=1))
        model.addConstr(U * index_set == image * index_set)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   47 * huber(50*x1 + p2 + 81, 1)
    # s.t.  9*x1 + 40 >= -55
    def test_huber_4(self):
        model = admm.Model()
        x1 = admm.Var("x1", 1)
        p2 = 60
        model.setObjective(admm.sum(47 * admm.huber(50 * x1 + p2 + 81, 1)))
        model.addConstr(9 * x1 + 40 >= -55)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1||_2 + sum(max(x1 + x2 + 1, 1))
    # s.t.  x1 <= -1
    def test_0612_ex_20(self):
        n = 2
        model = admm.Model()
        x1 = admm.Var("x1", n, n)
        x2 = admm.Var("x2", n, n)
        model.setObjective(
            admm.norm(x1, ord=2) + admm.sum(admm.maximum(x1 + x2 + 1, 1))
        )
        model.addConstr(x1 <= -1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||A x - b||_2 + ||x||_1
    def test_ex11_norm2(self):
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = np.array([16, 40])
        model = admm.Model()
        x = admm.Var("x", 4)
        model.setObjective(admm.norm(A @ x - b, ord=2) + admm.norm(x, ord=1))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(0.5 * A x^T A + A)
    # s.t.  b x = 0
    def test_ex1_matrix_1(self):
        model = admm.Model()
        A = np.array([[5, 4], [3, 2], [1, 0]])
        b = np.array([[3, 4, 5], [6, 7, 10], [-6, 2, 2.3]])
        x = admm.Var("x", 3, 2)
        model.setObjective(admm.sum(0.5 * A @ x.T @ A + A))
        model.addConstr(b @ x == 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum((x1 + p1 + 2) * log(x1 + p1 + 2))
    # s.t.  x1 >= 0
    def test_constr_check(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2, nonneg=True)
        p1 = np.array([[-30, 68], [-66, -44]])
        model.setObjective(admm.sum(admm.entropy(1 * x1 + p1 + 2)))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(5 * inrange(96*x1 + x3 + 38, -86.65, 9.09)) + sum(p1)
    # s.t.  mixed linear constraints on x1, x2, x3
    def test_in_range(self):

        model = admm.Model()
        x1 = admm.Var("x1", 4, 4, symmetric=True)
        x2 = admm.Var("x2", 4, nonpos=True)
        x3 = admm.Var("x3", 4, 4, symmetric=True)
        p1 = [-77, 65, -77, 82]
        model.setObjective(
            admm.sum(5 * admm.inrange(96 * x1 + x3 + 38, -86.65, 9.09)) + admm.sum(p1)
        )
        model.addConstr(33 * x1 + 63 >= -2)
        model.addConstr(-52 * x2 + 2 <= 96)
        model.addConstr(4 * x3 + 59 >= 23)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   76 * ||9*x1 + 74||_1
    # s.t.  -2*x1 + 4 >= -63,   x1 >= 0
    def test_inf_by_price(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, nonneg=True)
        model.setObjective(76 * admm.norm(9 * x1 + 74, ord=1))
        model.addConstr(-2 * x1 + 4 >= -63)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum((60*x1 + 56)^2)
    # s.t.  -25*x1 + 46 >= -32,   x1 >= 0
    def test_convergence_slow(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, nonneg=True)
        model.setObjective(admm.sum(admm.square(60 * x1 + 56)))
        model.addConstr(-25 * x1 + 46 >= -32)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||9*x2 + 2||_2
    # s.t.  -53*x2 + 62 >= 52
    def test_Norm2(self):
        model = admm.Model()
        x2 = admm.Var("x2", 3, 3)
        model.setObjective(admm.norm(9 * x2 + 2, ord=2))
        model.addConstr(-53 * x2 + 62 >= 52)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   squared_hinge(x)
    #
    # where  squared_hinge(x) = max(1 - x, 0)^2
    def test_squared_hinge(self):
        model = admm.Model()
        x = admm.Var("x")
        model.setObjective(admm.squared_hinge(x))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(47 * (2*x1 + x2 + 14)^2)
    # s.t.  81*x1 + 23 >= -94,   -32*x2 + 92 >= -83
    def test_consistent(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2, symmetric=True)
        x2 = admm.Var("x2", 2, 2, nonneg=True)
        model.setObjective(admm.sum(47 * admm.square(2 * x1 + x2 + 14)))
        model.addConstr(81 * x1 + 23 >= -94)
        model.addConstr(-32 * x2 + 92 >= -83)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   37 * ||70*x1 + x3 + 50||_* + sum(40 * -sqrt(6*x3 + 75))
    # s.t.  -44*x1 + 49 <= -97,   27*x3 + 56 >= -15
    def test_prebatch(self):
        model = admm.Model()
        x1 = admm.Var("x1", 5, 3)
        x3 = admm.Var("x3", 5, 3)
        model.setObjective(
            37 * admm.norm(70 * x1 + x3 + 50, ord="nuc")
            + admm.sum(40 * -admm.sqrt(6 * x3 + 75))
        )
        model.addConstr(-44 * x1 + 49 <= -97)
        model.addConstr(27 * x3 + 56 >= -15)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x)
    # s.t.  A x = p
    def test_my_0627(self):
        model = admm.Model()
        x = admm.Var(2, 1)
        p = np.array([[12], [2313]])
        A = np.array([[6, 8], [21, 7]])
        expr = admm.sum(x)
        model.setObjective(expr)
        model.addConstr(A @ x == p)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   c^T x
    # s.t.  A[i,:] x <= b[i],   i = 1,...,m
    def test_Linear_Program_Comparison(self):
        m = 15
        n = 10
        np.random.seed(1)
        s0 = np.random.randn(m)
        lamb0 = np.maximum(-s0, 0)
        s0 = np.maximum(s0, 0)
        x0 = np.random.randn(n)
        A = np.random.randn(m, n)
        b = A @ x0 + s0
        c = -A.T @ lamb0

        model = admm.Model()
        x_admm = admm.Var(n)
        model.setObjective(c.T @ x_admm)
        for i in range(m):
            model.addConstr(A[i, :] @ x_admm <= b[i])

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   0.5 * x^T P x + q^T x
    # s.t.  G[i,:] x <= h[i],   i = 1,...,m,   A[i,:] x = b[i],   i = 1,...,p
    def test_Quadratic_Program_Comparison(self):
        m = 15
        n = 10
        p = 5
        np.random.seed(1)
        P = np.random.randn(n, n)
        P = P.T @ P
        q = np.random.randn(n)
        G = np.random.randn(m, n)
        h = G @ np.random.randn(n)
        A = np.random.randn(p, n)
        b = np.random.randn(p)

        model = admm.Model()
        x = admm.Var(n)
        objective = 0.5 * x.T @ P @ x + q.T @ x
        model.setObjective(objective)

        for i in range(m):
            model.addConstr(G[i, :] @ x <= h[i])

        for i in range(p):
            model.addConstr(A[i, :] @ x == b[i])

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(C * X)
    # s.t.  sum(A[i] * X) = b[i],   i = 1,...,p,   X in PSD
    def test_SDP_Comparison(self):
        n = 3
        p = 3
        np.random.seed(1)
        C = np.random.rand(n, n)

        A = []
        b = []
        for i in range(p):
            A.append(np.random.randn(n, n))
            b.append(np.random.randn())

        model = admm.Model()
        X_admm = admm.Var("X", n, n, PSD=True)

        for i in range(p):
            model.addConstr(admm.sum(A[i] * X_admm) == b[i])

        model.setObjective(admm.sum(C * X_admm))
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   f^T x
    # s.t.  F x = g,   ||A[i] x + b[i]||_2 <= c[i]^T x + d[i],   i = 1,...,m
    def test_SOCP_Comparison(self):
        m = 3
        n = 10
        p = 5
        n_i = 5
        np.random.seed(2)
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

        model = admm.Model()
        x_admm = admm.Var("x", n, 1)

        model.setObjective(f.T @ x_admm)

        model.addConstr(F @ x_admm == g.reshape(p, 1))

        for i in range(m):
            model.addConstr(
                admm.norm(A[i] @ x_admm + b[i], ord=2) <= c[i].T @ x_admm + d[i]
            )

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||A x - b||_2^2
    def test_Least_Squares_Comparison(self):
        m = 20
        n = 15
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        model = admm.Model()
        x_admm = admm.Var("x", n, 1)
        cost = admm.sum(admm.square(A @ x_admm - b))
        model.setObjective(cost)

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum_i  max(-R[i] * P[i,:]^T D[i,:], -B[i])
    # s.t.  D >= 0,   D^T 1 <= T,   D 1 >= c
    def test_Advertising_Comparison(self):
        np.random.seed(1)
        m = 5
        n = 24
        SCALE = 10000
        B = np.random.lognormal(mean=8, size=(m, 1)) + 10000
        B = 1000 * np.round(B / 1000)

        P_ad = np.random.uniform(size=(m, 1))
        P_time = np.random.uniform(size=(1, n))
        P = P_ad.dot(P_time)

        T = np.sin(np.linspace(-2 * np.pi / 2, 2 * np.pi - 2 * np.pi / 2, n)) * SCALE
        T += -np.min(T) + SCALE
        c = np.random.uniform(size=(m,))
        c *= 0.6 * T.sum() / c.sum()
        c = 1000 * np.round(c / 1000)
        R = np.array([np.random.lognormal(c.min() / c[i]) for i in range(m)])

        model = admm.Model()
        D = admm.Var("D", m, n)
        Si = [admm.maximum(-R[i] * P[i, :] @ D[i, :].T, -B[i]) for i in range(m)]
        sum_Si = 0
        for i in range(m):
            sum_Si += Si[i]
        model.setObjective(admm.sum(sum_Si))
        model.addConstr(D >= 0)
        model.addConstr(D.T @ np.ones(m) <= T)
        model.addConstr(D @ np.ones(n) >= c)

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   w^T Sigma w
    # s.t.  sum(w) = 1,   mu^T w >= 0.1,   ||w||_1 <= 2
    def test_Portfolio_Optimization_Comparison(self):
        np.random.seed(2)
        n = 5
        mu = np.abs(np.random.randn(n, 1)) / 15
        Sigma = np.random.uniform(-0.15, 0.8, size=(n, n))
        Sigma_nom = Sigma.T.dot(Sigma)

        model = admm.Model()
        w = admm.Var(n)
        ret = mu.T @ w
        risk = w.T @ Sigma_nom @ w
        model.setObjective(risk)
        model.addConstr(admm.sum(w) == 1)
        model.addConstr(ret >= 0.1)
        model.addConstr(admm.norm(w, ord=1) <= 2)

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # max   w^T Sigma_risk w          (over Sigma_risk, Delta;  w = (1/n)*1 fixed)
    # s.t.  Sigma_risk = Sigma_nom + Delta,   diag(Delta) = 0,   |Delta_ij| <= 0.2,   Sigma_risk in PSD
    def test_Worst_Case_Risk_Comparison(self):
        np.random.seed(2)
        n = 5
        Sigma = np.random.uniform(-0.15, 0.8, size=(n, n))
        Sigma_nom = Sigma.T.dot(Sigma)

        w_fixed = np.ones(n) / n

        model_risk = admm.Model()
        Sigma_risk = admm.Var((n, n), PSD=True)
        Delta = admm.Var((n, n), symmetric=True)
        risk = w_fixed.T @ Sigma_risk @ w_fixed
        model_risk.setObjective(-risk)
        model_risk.addConstr(Sigma_risk == Sigma_nom + Delta)
        model_risk.addConstr(admm.diag(Delta) == 0)
        model_risk.addConstr(admm.abs(Delta) <= 0.2)

        model_risk.setOption(admm.Options.admm_max_iteration, 1000000)
        model_risk.setOption(admm.Options.solver_verbosity_level, 3)
        model_risk.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model_risk.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model_risk.optimize()

        self.assertEqual(model_risk.StatusString, "SOLVE_OPT_SUCCESS")

    # min   (1/m) sum_i log(1 + exp(-y_i * x_i^T theta)) + 0.5 * ||theta[:k]||_1
    def test_Logistic_Regression_L1_Comparison(self):
        def pairs(Z):
            m, n = Z.shape
            k = n * (n + 1) // 2
            X = np.zeros((m, k))
            count = 0
            for i in range(n):
                for j in range(i, n):
                    X[:, count] = Z[:, i] * Z[:, j]
                    count += 1
            return X

        np.random.seed(1)
        n = 10
        k = n * (n + 1) // 2
        m = 200
        sigma = 1.9
        DENSITY = 1.0
        theta_true = np.random.randn(n, 1)
        idxs = np.random.choice(range(n), int((1 - DENSITY) * n), replace=False)
        for idx in idxs:
            theta_true[idx] = 0

        Z = np.random.binomial(1, 0.5, size=(m, n))
        Y = np.sign(Z.dot(theta_true) + np.random.normal(0, sigma, size=(m, 1)))
        X = pairs(Z)
        X = np.hstack([X, np.ones((m, 1))])

        model = admm.Model()
        theta = admm.Var(k + 1, 1)
        loss = admm.sum(admm.logistic(-Y * X @ theta, 1))
        reg = admm.norm(theta[:k], ord=1)

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.setObjective(loss * (1 / m) + 0.5 * reg)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   gamma * 0.5 * w^T Sigma w - mu^T w
    # s.t.  sum(w) = 1,   w >= 0
    def test_Parameter_Handling_Comparison(self):
        np.random.seed(1)
        n = 3
        mu = np.abs(np.random.randn(n, 1))
        Sigma = np.random.randn(n, n)
        Sigma = Sigma.T.dot(Sigma)

        gamma_values = [0.1, 1.0, 10.0]

        for gamma_val in gamma_values:
            model = admm.Model()
            w = admm.Var(n)
            gamma = admm.Param("gamma")
            ret = mu.T @ w
            risk = w.T @ Sigma @ w
            model.setObjective(gamma * 0.5 * risk - ret)
            model.addConstr(admm.sum(w) == 1)
            model.addConstr(w >= 0)

            model.setOption(admm.Options.admm_max_iteration, 1000)
            model.setOption(admm.Options.solver_verbosity_level, 3)
            model.optimize({"gamma": gamma_val})

            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||X^T beta - y||_2
    def test_Robust_Regression_Comparison(self):
        np.random.seed(1)
        n = 300
        SAMPLES = int(1.5 * n)
        beta_true = 5 * np.random.normal(size=(n, 1))
        X = np.random.randn(n, SAMPLES)
        Y = np.zeros((SAMPLES, 1))
        v = np.random.normal(size=(SAMPLES, 1))

        TESTS = 10
        p_vals = np.linspace(0, 0.15, num=TESTS)

        for p in p_vals:
            factor = 2 * np.random.binomial(1, 1 - p, size=(SAMPLES, 1)) - 1
            Y = factor * X.T.dot(beta_true) + v

            model_lsq = admm.Model()
            beta_lsq = admm.Var(n, 1)
            cost_lsq = admm.norm(X.T @ beta_lsq - Y, ord=2)
            model_lsq.setObjective(cost_lsq)
            model_lsq.setOption(admm.Options.solver_verbosity_level, 3)
            model_lsq.optimize()

            self.assertEqual(model_lsq.StatusString, "SOLVE_OPT_SUCCESS")

    # min   lam * ||beta||_1 - (1/m) * (y^T (X beta) - sum(log(1 + exp(X beta))))
    def test_Logistic_Regression_L1_Error_Comparison(self):
        np.random.seed(1)
        n = 50
        m = 50

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        beta_true = np.array([1, 0.5, -0.5] + [0] * (n - 3))
        X = (np.random.random((m, n)) - 0.5) * 10
        Y = np.round(sigmoid(X @ beta_true + np.random.randn(m) * 0.5))

        trials = 10
        lambda_vals = np.logspace(-2, 0, trials)

        model = admm.Model()
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)

        beta = admm.Var(n, 1)

        for i in range(trials):
            lamda = lambda_vals[i]

            log_likelihood = Y.T @ (X @ beta) - admm.sum(admm.logistic(X @ beta, 1))
            model.setObjective(
                lamda * admm.norm(beta, ord=1) - log_likelihood * (1 / m)
            )
            model.setOption(admm.Options.admm_max_iteration, 10000)
            model.setOption(admm.Options.solver_verbosity_level, 3)
            model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
            model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
            model.optimize()

            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   0.5 * sum(|y - X w|) + (tau - 0.5) * sum(y - X w)
    #
    # where  tau in {0.1, 0.5, 0.9} is the quantile level
    def test_Quantile_Regression_Comparison(self):
        TRAIN_LEN = 400
        SKIP_LEN = 100
        TEST_LEN = 50
        m = 10
        TOTAL_LEN = TRAIN_LEN + SKIP_LEN + TEST_LEN

        seed = 1
        np.random.seed(seed)
        x0 = np.random.randn(m)
        x = np.zeros(TOTAL_LEN)
        x[:m] = x0
        for i in range(m + 1, TOTAL_LEN):
            x[i] = 1.8 * x[i - 1] - 0.82 * x[i - 2] + np.random.normal()
        x = np.exp(0.05 * x + 0.05 * np.random.normal(size=TOTAL_LEN))

        tau_vals = [0.1, 0.5, 0.9]
        train_end = TRAIN_LEN + SKIP_LEN
        y_train = x[m:train_end]
        X_train = np.lib.stride_tricks.sliding_window_view(x, m + 1)[: train_end - m]
        ones_train = np.ones_like(y_train)
        X_aug = np.column_stack([X_train, ones_train])

        for tau_val in tau_vals:
            model = admm.Model()
            w = admm.Var(m + 2)

            model.setOption(admm.Options.admm_max_iteration, 10000)
            model.setOption(admm.Options.solver_verbosity_level, 3)
            model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
            model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)

            tau = tau_val
            r = y_train - (X_aug @ w)
            model.setObjective(0.5 * admm.sum(admm.abs(r)) + (tau - 0.5) * admm.sum(r))
            model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||X beta - y||_2^2 + lam * ||beta||_2^2
    def test_Ridge_Regression_Comparison(self):
        def objective_fn(X, Y, beta, lambd):
            return (X @ beta - Y.reshape(50, 1)).T @ (X @ beta - Y.reshape(50, 1)) + lambd * beta.T @ beta

        def generate_data(m=100, n=20, sigma=5):
            np.random.seed(1)
            beta_star = np.random.randn(n)
            X = np.random.randn(m, n)
            Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
            return X, Y

        m = 100
        n = 20
        sigma = 5

        X, Y = generate_data(m, n, sigma)
        X_train = X[:50, :]
        Y_train = Y[:50]

        lambd_values = np.logspace(-2, 3, 10)

        model = admm.Model()
        beta = admm.Var(n, 1)
        model.setOption(admm.Options.admm_max_iteration, 5000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)

        for v in lambd_values:
            lambd = v

            model.setObjective(objective_fn(X_train, Y_train, beta, lambd))
            model.optimize()

            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   (1/m) * sum(max(1 - y*(X beta - v), 0)) + lam * ||beta||_1
    def test_SVM_L1_Regularization_Comparison(self):
        np.random.seed(1)
        n = 20
        m = 1000
        DENSITY = 0.2
        beta_true = np.random.randn(n, 1)
        idxs = np.random.choice(range(n), int((1 - DENSITY) * n), replace=False)
        for idx in idxs:
            beta_true[idx] = 0
        offset = 0
        sigma = 45
        X = np.random.normal(0, 5, size=(m, n))
        Y = np.sign(X.dot(beta_true) + offset + np.random.normal(0, sigma, size=(m, 1)))

        trials = 3
        lambda_vals = np.logspace(-2, 0, trials)

        model = admm.Model()
        beta = admm.Var(n, 1)
        v = admm.Var()
        lambd = admm.Param("lambd")
        loss = admm.sum(admm.maximum(1 - Y * (X @ beta - v), 0))
        reg = admm.norm(beta, ord=1)
        model.setObjective(loss * (1 / m) + lambd * reg)
        model.setOption("solver_verbosity_level", 3)
        model.setOption("admm_max_iteration", 50000)
        model.setOption("termination_absolute_error_threshold", 1e-3)
        model.setOption("termination_relative_error_threshold", 1e-6)

        for i in range(trials):
            model.optimize({"lambd": lambda_vals[i]})

            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||w||_2
    # s.t.  A_tar w = [1; 0],   ||A_s[i,:] w||_2 <= sidelobe_limit,   i = 1,...,ns
    #
    # where  w in R^(2n) is the real/imaginary beamforming weight vector
    def test_Antenna_Array_Beamforming_Comparison(self):
        ARRAY_GEOMETRY = "2D_RANDOM"
        lambda_wl = 1
        theta_tar = 60
        min_sidelobe = -20
        max_half_beam = 50

        if ARRAY_GEOMETRY == "2D_RANDOM":
            np.random.seed(1)
            n = 36
            L = 5
            loc = L * np.random.random((n, 2))
        elif ARRAY_GEOMETRY == "1D_UNIFORM_LINE":
            n = 30
            d = 0.45 * lambda_wl
            loc = np.hstack(
                (d * np.array(range(0, n)).reshape(-1, 1), np.zeros((n, 1)))
            )
        elif ARRAY_GEOMETRY == "2D_UNIFORM_LATTICE":
            m = 6
            n = m**2
            d = 0.45 * lambda_wl
            loc = np.zeros((n, 2))
            for x in range(m):
                for y in range(m):
                    loc[m * y + x, :] = [x, y]
            loc = loc * d
        else:
            raise Exception("Undefined array geometry")

        theta = np.arange(1, 361).reshape(-1, 1)
        A = np.kron(np.cos(np.pi * theta / 180), loc[:, 0].T) + np.kron(
            np.sin(np.pi * theta / 180), loc[:, 1].T
        )
        A = np.exp(2 * np.pi * 1j / lambda_wl * A)
        ind_closest = np.argmin(np.abs(theta - theta_tar))
        Atar = A[ind_closest, :]
        Atar_R = Atar.real
        Atar_I = Atar.imag
        neg_Atar_I = -Atar_I
        Atar_RI = np.block([[Atar_R, neg_Atar_I], [Atar_I, Atar_R]])
        realones_ri = np.array([1.0, 0.0])

        halfbeam_bot_admm = 1
        halfbeam_top_admm = max_half_beam
        while halfbeam_top_admm - halfbeam_bot_admm > 1:
            halfbeam_cur = np.ceil((halfbeam_top_admm + halfbeam_bot_admm) / 2.0)
            ind = np.nonzero(
                np.squeeze(
                    np.array(
                        np.logical_or(
                            theta <= (theta_tar - halfbeam_cur),
                            theta >= (theta_tar + halfbeam_cur),
                        )
                    )
                )
            )
            As = A[ind[0], :]
            As_R = As.real
            As_I = As.imag
            neg_As_I = -As_I
            As_RI_top = np.block([As_R, neg_As_I])
            As_RI_bot = np.block([As_I, As_R])
            model_admm = admm.Model()
            w_ri_admm = admm.Var(2 * n)
            model_admm.addConstr(Atar_RI @ w_ri_admm == realones_ri)
            for i in range(As.shape[0]):
                As_ri_row = np.vstack((As_RI_top[i, :], As_RI_bot[i, :]))
                model_admm.addConstr(
                    admm.norm(As_ri_row @ w_ri_admm, ord=2) <= 10 ** (min_sidelobe / 20)
                )
            model_admm.setObjective(0)
            model_admm.setOption(admm.Options.solver_verbosity_level, 3)
            model_admm.setOption(admm.Options.admm_max_iteration, 5000)
            model_admm.setOption(
                admm.Options.termination_absolute_error_threshold, 1e-2
            )
            model_admm.setOption(
                admm.Options.termination_relative_error_threshold, 1e-2
            )
            model_admm.setOption(admm.Options.penalty_param_auto, 1)
            model_admm.optimize()

            if model_admm.status == 1:
                halfbeam_top_admm = halfbeam_cur
            else:
                halfbeam_bot_admm = halfbeam_cur
        halfbeam_admm = halfbeam_top_admm
        if (theta_tar + halfbeam_admm) % 360 > (theta_tar - halfbeam_admm) % 360:
            upper_beam_admm = (theta_tar + halfbeam_admm) % 360
            lower_beam_admm = (theta_tar - halfbeam_admm) % 360
            ind = np.nonzero(
                np.squeeze(
                    np.array(
                        np.logical_or(
                            theta <= lower_beam_admm, theta >= upper_beam_admm
                        )
                    )
                )
            )
        else:
            upper_beam_admm = (theta_tar + halfbeam_admm) % 360
            lower_beam_admm = (theta_tar - halfbeam_admm) % 360
            ind = np.nonzero(
                np.squeeze(
                    np.array(
                        np.logical_and(
                            theta <= lower_beam_admm, theta >= upper_beam_admm
                        )
                    )
                )
            )
        As = A[ind[0], :]
        As_R = As.real
        As_I = As.imag
        neg_As_I = -As_I
        As_RI_top = np.block([As_R, neg_As_I])
        As_RI_bot = np.block([As_I, As_R])
        model_admm = admm.Model()
        w_ri_admm = admm.Var(2 * n)
        model_admm.addConstr(Atar_RI @ w_ri_admm == realones_ri)
        for i in range(As.shape[0]):
            As_ri_row = np.vstack((As_RI_top[i, :], As_RI_bot[i, :]))
            model_admm.addConstr(
                admm.norm(As_ri_row @ w_ri_admm, ord=2) <= 10 ** (min_sidelobe / 20)
            )
        model_admm.setObjective(admm.norm(w_ri_admm, ord=2))
        model_admm.setOption(admm.Options.solver_verbosity_level, 3)
        model_admm.setOption(admm.Options.admm_max_iteration, 20000)
        model_admm.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model_admm.setOption(admm.Options.termination_relative_error_threshold, 1e-3)
        model_admm.setOption(admm.Options.penalty_param_auto, 1)
        model_admm.optimize()

        self.assertEqual(model_admm.StatusString, "SOLVE_OPT_SUCCESS")

    # max   I(x) = c^T x - sum(entropy(p x) / log 2)
    # s.t.  sum(x) = 1,   x >= 0
    #
    # where  x is the input distribution, p is the channel transition matrix
    def test_Channel_Capacity_Comparison(self):

        def channel_capacity_ADMM(N, M, p, sum_x=1):
            """
            Channel capacity using ADMM solver
            """
            if N * M == 0:
                print(
                    "The range of both input and output values must be greater than zero"
                )
                return "failed", np.nan, np.nan

            model = admm.Model()
            x = admm.Var(N)

            y = p @ x

            c = np.sum(np.array((xlogy(p, p) / math.log(2))), axis=0)
            I = c @ x + admm.sum(-admm.entropy(y) * (1 / math.log(2)))

            model.setObjective(-I)
            model.addConstr(admm.sum(x) == sum_x)
            model.addConstr(x >= 0)

            model.setOption(admm.Options.admm_max_iteration, 10000)
            model.setOption(admm.Options.solver_verbosity_level, 3)
            model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
            model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
            model.optimize()

            return model.StatusString
        test_cases = [
            {
                "P": np.array([[0.75, 0.25], [0.25, 0.75]]),
                "n": 2,
                "m": 2,
            },
            {
                "P": np.array([[0.8, 0.0], [0.0, 0.8], [0.2, 0.2]]),
                "n": 2,
                "m": 3,
            },
            {
                "P": np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]]),
                "n": 3,
                "m": 3,
            },
        ]
        for case in test_cases:
            P = case["P"]
            n = case["n"]
            m = case["m"]
            self.assertEqual(channel_capacity_ADMM(n, m, P), "SOLVE_OPT_SUCCESS")

    # min   sum(x * log(x))
    # s.t.  A x = b,   F x <= g
    def test_Entropy_Maximization_Comparison(self):
        n = 20
        m = 10
        p = 5

        np.random.seed(0)
        tmp = np.random.rand(n)
        A = np.random.randn(m, n)
        b = A.dot(tmp)
        F = np.random.randn(p, n)
        g = F.dot(tmp) + np.random.rand(p)

        model = admm.Model()
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        x = admm.Var(n)
        model.setObjective(admm.sum(admm.entropy(x)))
        model.addConstr(A @ x == b)
        model.addConstr(F @ x <= g)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||A x - y||_2^2 + tau * sum(x)
    # s.t.  0 <= x <= 1
    def test_Sparse_Binary_Recovery_Comparison(self):
        n = 2000
        m = 200
        p = 0.01
        snr = 5
        np.random.seed(0)
        sigma = np.sqrt(p * n / (snr**2))
        A = np.random.randn(m, n)

        x_true = (np.random.rand(n) <= p).astype(int)
        v = sigma * np.random.randn(m)

        y = A.dot(x_true) + v

        model = admm.Model()
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        x = admm.Var(n)
        tau = 2 * admm.log(1 / p - 1) * sigma**2
        model.setObjective(admm.sum(admm.square(A @ x - y)) + tau * admm.sum(x))
        model.addConstr(0 <= x)
        model.addConstr(x <= 1)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||X_obs c - y_obs||_2^2
    # s.t.  X_cens c >= D
    #
    # where  D is the censoring threshold, obs/cens partition the observations
    def test_Censored_Regression_Comparison(self):
        n = 30
        M = 50
        K = 200

        np.random.seed(n * M * K)
        X = np.random.randn(K * n).reshape(K, n)
        c_true = np.random.rand(n)

        y = X.dot(c_true) + 0.3 * np.sqrt(n) * np.random.randn(K)

        order = np.argsort(y)
        y_ordered = y[order]
        X_ordered = X[order, :]

        D = (y_ordered[M - 1] + y_ordered[M]) / 2.0

        model = admm.Model()
        X_uncensored = X_ordered[:M, :]
        c = admm.Var(n)
        model.setObjective(
            (X_uncensored @ c - y_ordered[:M]).T @ (X_uncensored @ c - y_ordered[:M])
        )
        model.addConstr(X_ordered[M:, :] @ c >= D)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||A - Y X||_F
    # s.t.  X >= 0  (odd iters),   Y >= 0  (even iters)
    #
    # where  alternating minimization over X and Y with k latent factors
    def test_Non_Negative_Matrix_Factorization_Comparison(self):
        np.random.seed(0)

        m = 10
        n = 10
        k = 4
        A = np.random.rand(m, k).dot(np.random.rand(k, n))

        Y_init = np.random.rand(m, k) + np.random.rand(m, k)

        Y = Y_init.copy()

        MAX_ITERS = 10
        residual_admm = np.zeros(MAX_ITERS)

        Y_admm = Y_init.copy()
        X_admm = None

        for iter_num in range(1, 1 + MAX_ITERS):
            model = admm.Model()
            if iter_num % 2 == 1:
                X = admm.Var(k, n)
                model.addConstr(X >= 0)
                model.setObjective(admm.norm(A - Y_admm @ X, "fro"))
            else:
                Y = admm.Var(m, k)
                model.addConstr(Y >= 0)
                model.setObjective(admm.norm(A - Y @ X_admm, "fro"))

            model.setOption(admm.Options.admm_max_iteration, 10000)
            model.setOption(admm.Options.solver_verbosity_level, 3)
            model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
            model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
            model.optimize()

            residual_admm[iter_num - 1] = model.ObjVal
            if iter_num % 2 == 1:
                X_admm = X.X
            else:
                Y_admm = Y.X
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||w||_F^2 + tau * ||v||_F^2
    # s.t.  x[:,t+1] = A x[:,t] + B w[:,t],   y[:,t] = C x[:,t] + v[:,t]
    #
    # where  w is process noise, v is measurement noise (Kalman smoother)
    def test_Vehicle_Tracking_Kalman_Filter_Comparison(self):
        n = 1000
        T = 50
        delt = np.linspace(0, T, n, endpoint=True, retstep=True)[1]
        gamma = 0.05

        A = np.zeros((4, 4))
        B = np.zeros((4, 2))
        C = np.zeros((2, 4))

        A[0, 0] = 1
        A[1, 1] = 1
        A[0, 2] = (1 - gamma * delt / 2) * delt
        A[1, 3] = (1 - gamma * delt / 2) * delt
        A[2, 2] = 1 - gamma * delt
        A[3, 3] = 1 - gamma * delt

        B[0, 0] = delt**2 / 2
        B[1, 1] = delt**2 / 2
        B[2, 0] = delt
        B[3, 1] = delt

        C[0, 0] = 1
        C[1, 1] = 1

        sigma = 20
        p = 0.20
        np.random.seed(6)

        x = np.zeros((4, n + 1))
        x[:, 0] = [0, 0, 0, 0]
        y = np.zeros((2, n))

        w = np.random.randn(2, n)
        v = np.random.randn(2, n)

        np.random.seed(0)
        indx = np.random.rand(n) <= p
        v[:, indx] = sigma * np.random.randn(2, n)[:, indx]

        for t in range(n):
            y[:, t] = C.dot(x[:, t]) + v[:, t]
            x[:, t + 1] = A.dot(x[:, t]) + B.dot(w[:, t])

        model = admm.Model()
        x_admm = admm.Var(4, n + 1)
        w_admm = admm.Var(2, n)
        v_admm = admm.Var(2, n)

        tau = 0.08

        obj = admm.sum(w_admm * w_admm) + tau * admm.sum(v_admm * v_admm)
        model.setObjective(obj)

        for t in range(n):
            model.addConstr(x_admm[:, t + 1] == A @ x_admm[:, t] + B @ w_admm[:, t])
            model.addConstr(y[0, t] == C[0, :] @ x_admm[:, t] + v_admm[0, t])
            model.addConstr(y[1, t] == C[1, :] @ x_admm[:, t] + v_admm[1, t])

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x)
    # s.t.  0.1 <= x[i] <= 1.0,   i = 0, 1
    def test_Simple_Wire_Sizing_Comparison(self):
        model = admm.Model()
        x = admm.Var("x", 2)

        model.setObjective(admm.sum(x))

        model.addConstr(x[0] >= 0.1)
        model.addConstr(x[1] >= 0.1)
        model.addConstr(x[0] <= 1.0)
        model.addConstr(x[1] <= 1.0)

        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(-log(alpha + x))
    # s.t.  sum(x) = sum_x,   x >= 0
    def test_Water_Filling_Comparison(self):
        np.random.seed(0)

        test_cases = [
            {
                "n": 3,
                "alpha": np.array([0.8, 1.0, 1.2]),
                "sum_x": 1.0,
            },
            {
                "n": 5,
                "alpha": np.array([0.5, 0.7, 1.0, 1.3, 1.5]),
                "sum_x": 2.0,
            },
            {
                "n": 4,
                "alpha": np.array([0.3, 0.9, 1.1, 0.6]),
                "sum_x": 1.5,
            },
        ]

        for case in test_cases:
            n = case["n"]
            alpha = case["alpha"]
            sum_x = case["sum_x"]

            model = admm.Model()
            x = admm.Var(n)
            alpha_param = admm.Param("alpha", n)

            model.setObjective(admm.sum(-admm.log(alpha_param + x)))

            model.addConstr(admm.sum(x) - sum_x == 0)
            model.addConstr(x >= 0)

            model.setOption(admm.Options.admm_max_iteration, 5000)
            model.setOption(admm.Options.solver_verbosity_level, 3)
            model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
            model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
            model.optimize({"alpha": alpha})

            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   gamma * w^T Sigma w - mu^T w
    # s.t.  sum(w) = 1,   w >= 0
    def test_Mean_Variance_Portfolio_Comparison(self):
        np.random.seed(0)

        def generate_test_data(n=5, seed=1):
            """Generate synthetic portfolio data"""
            np.random.seed(seed)
            mu = np.abs(np.random.randn(n, 1))
            Sigma = np.random.randn(n, n)
            Sigma = Sigma.T @ Sigma
            return mu, Sigma

        test_cases = [
            {"n": 5, "gamma_vals": [0.1, 1.0, 10.0]},
            {"n": 8, "gamma_vals": [0.5, 5.0, 50.0]},
            {"n": 10, "gamma_vals": [1.0, 10.0, 100.0]},
        ]

        for case in test_cases:
            n = case["n"]
            gamma_vals = case["gamma_vals"]

            mu, Sigma = generate_test_data(n=n, seed=1)

            for gamma in gamma_vals:
                model = admm.Model()
                w = admm.Var((n, 1))

                ret = mu.T @ w
                risk = w.T @ Sigma @ w

                objective = gamma * admm.sum(risk) - admm.sum(ret)
                model.setObjective(objective)

                model.addConstr(admm.sum(w) == 1)
                model.addConstr(w >= 0)

                model.setOption(admm.Options.admm_max_iteration, 5000)
                model.setOption(admm.Options.solver_verbosity_level, 3)
                model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
                model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
                model.optimize()

                self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   1000 * (1/(T(T-1))) * sum(|D w|)
    # s.t.  sum(w) = 1,   w >= 0
    #
    # where  D contains all pairwise return differences across T periods
    def test_Gini_Mean_Difference_Portfolio_Comparison(self):
        np.random.seed(0)

        def generate_test_data(T=30, N=5, seed=123):
            """Generate synthetic return data for testing"""
            rs = np.random.RandomState(seed)

            cov = rs.rand(N, N) * 1.5 - 0.5
            cov = cov @ cov.T / 1000 + np.diag(rs.rand(N) * 0.7 + 0.3) / 1000
            mean = np.zeros(N) + 1 / 1000

            returns = st.multivariate_normal.rvs(
                mean=mean, cov=cov, size=T, random_state=rs
            )

            D = np.array([]).reshape(0, N)
            for j in range(0, returns.shape[0] - 1):
                D = np.concatenate((D, returns[j + 1 :] - returns[j, :]), axis=0)

            return D

        test_cases = [
            {"T": 20, "N": 3},
            {"T": 30, "N": 4},
            {"T": 50, "N": 5},
        ]

        for case in test_cases:
            T = case["T"]
            N = case["N"]

            D = generate_test_data(T, N, seed=123)

            model = admm.Model()
            w = admm.Var((N, 1))

            all_pairs_ret_diff = D @ w

            model.addConstr(w >= 0)
            model.addConstr(admm.sum(w) == 1)

            risk = admm.sum(admm.abs(all_pairs_ret_diff)) * (1 / ((T - 1) * T))
            model.setObjective(risk * 1000)

            model.setOption(admm.Options.admm_max_iteration, 20000)
            model.setOption(admm.Options.solver_verbosity_level, 3)
            model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
            model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
            model.optimize()

            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   1000 * ||g||_2
    # s.t.  sum(x) = 1,   x >= 0,   g[i] = tr(B[i] X),   Z = [[X, x]; [x^T, 1]] in PSD
    #
    # where  B[i] are cokurtosis factor matrices, X = x x^T is the rank-1 SDP relaxation
    def test_Approximate_Kurtosis_Portfolio_Comparison(
        self,
    ):
        np.random.seed(0)

        def kurt_matrix(Y):
            """Calculate the cokurtosis matrix S_4"""
            P = Y
            T, n = P.shape
            mu = np.mean(P, axis=0).reshape(1, -1)
            mu = np.repeat(mu, T, axis=0)
            x = P - mu
            ones = np.ones((1, n))
            z = np.kron(ones, x) * np.kron(x, ones)
            S4 = 1 / T * z.T @ z
            return S4

        def block_vec_pq(A, p, q):
            """Calculate block vectorization operator"""
            mp, nq = A.shape
            if mp % p == 0 and nq % q == 0:
                m = int(mp / p)
                n = int(nq / q)
                bvec_A = np.empty((0, p * q))
                for j in range(n):
                    Aj = np.empty((0, p * q))
                    for i in range(m):
                        Aij = (
                            A[i * p : (i + 1) * p, j * q : (j + 1) * q]
                            .reshape(-1, 1, order="F")
                            .T
                        )
                        Aj = np.vstack([Aj, Aij])
                    bvec_A = np.vstack([bvec_A, Aj])
            return bvec_A

        test_cases = [
            {"n": 4, "T": 200, "K_factor": 2},
            {"n": 6, "T": 300, "K_factor": 2},
            {"n": 8, "T": 600, "K_factor": 2},
        ]

        for case in test_cases:
            n = case["n"]
            T = case["T"]
            K_factor = case["K_factor"]
            K = K_factor * n

            np.random.seed(42)
            Y = np.random.randn(T, n) * 0.02

            Sigma_4 = kurt_matrix(Y)

            A = block_vec_pq(Sigma_4, n, n)

            s_A, V_A = np.linalg.eig(A)
            s_A = np.clip(s_A.real, 0, np.inf)
            sort_idx = np.argsort(s_A)[::-1]
            s_A = s_A[sort_idx]
            V_A = V_A[:, sort_idx]

            Bi = []
            for i in range(K):
                B = s_A[i] ** 0.5 * V_A[:, i]
                B = B.reshape((n, n), order="F").real
                B = 0.5 * (B + B.T)
                Bi.append(B)

            model = admm.Model()

            x = admm.Var((n, 1))
            X = admm.Var(n, n, symmetric=True)
            g = admm.Var((K, 1))
            Z = admm.Var(n + 1, n + 1, PSD=True)

            risk = admm.norm(g, ord=2)
            model.setObjective(risk * 1000)

            model.addConstr(admm.sum(x) == 1.0)
            model.addConstr(x[:, 0] >= 0)

            for i in range(K):
                model.addConstr(g[i, 0] == admm.trace(Bi[i] @ X))

            model.addConstr(Z[0:n, 0:n] == X)
            model.addConstr(Z[0:n, n : n + 1] == x)
            model.addConstr(Z[n : n + 1, 0:n] == x.T)
            model.addConstr(Z[n, n] == 1.0)

            model.setOption(admm.Options.admm_max_iteration, 200000)
            model.setOption(admm.Options.termination_absolute_error_threshold, 1e-6)
            model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
            model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x^T Sigma x
    # s.t.  sum(x) = 1,   x >= 0
    def test_Minimum_Variance_Portfolio_Comparison(self):
        np.random.seed(0)

        test_cases = [
            {"n": 5},
            {"n": 8},
            {"n": 12},
        ]

        for case in test_cases:
            n = case["n"]

            np.random.seed(42)
            A = np.random.randn(n, n)
            Sigma = (
                A @ A.T + np.eye(n) * 0.0001
            )
            Sigma = Sigma / 1000

            model = admm.Model()
            x = admm.Var(n)
            model.setObjective(x.T @ Sigma @ x)
            model.addConstr(admm.sum(x) == 1)
            model.addConstr(x >= 0)

            model.setOption(admm.Options.admm_max_iteration, 10000)
            model.setOption(admm.Options.solver_verbosity_level, 3)
            model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
            model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
            model.optimize()

            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   w^T Sigma w
    # s.t.  sum(w) = 1,   w >= 0
    def test_Risk_Portfolio_Comparison(self):
        np.random.seed(0)

        test_cases = [
            {"n": 5},
            {"n": 8},
            {"n": 10},
        ]

        for case in test_cases:
            n = case["n"]

            np.random.seed(42)
            A = np.random.randn(n, n)
            Sigma = (
                A @ A.T + np.eye(n) * 0.0001
            )
            Sigma = Sigma / 1000

            model = admm.Model()
            x = admm.Var(n)
            model.setObjective(x.T @ Sigma @ x)
            model.addConstr(admm.sum(x) == 1)
            model.addConstr(x >= 0)

            model.setOption(admm.Options.admm_max_iteration, 10000)
            model.setOption(admm.Options.solver_verbosity_level, 3)
            model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
            model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
            model.optimize()

            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -mu^T x
    # s.t.  sum(x) = 1,   x >= 0,   x[0] + x[1] <= exposure_limit
    def test_Maximum_Return_Portfolio_Comparison(self):
        np.random.seed(0)

        test_cases = [
            {"n": 5, "exposure_limit": 0.8},
            {"n": 8, "exposure_limit": 0.6},
            {
                "n": 10,
                "exposure_limit": 0.7,
            },
        ]

        for case in test_cases:
            n = case["n"]
            exposure_limit = case["exposure_limit"]

            np.random.seed(42)
            mu = np.random.randn(n) * 0.001
            mu = mu + np.random.randn(n) * 0.0005

            model = admm.Model()
            x = admm.Var(n)
            model.setObjective(-mu @ x)
            model.addConstr(admm.sum(x) == 1)
            model.addConstr(x >= 0)
            model.addConstr(x[0] + x[1] <= exposure_limit)

            model.setOption(admm.Options.admm_max_iteration, 10000)
            model.setOption(admm.Options.solver_verbosity_level, 3)
            model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
            model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
            model.optimize()

            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x||_1
    # s.t.  A x = b
    def test_Sparse_Signal_Recovery_Comparison(self):
        np.random.seed(1)

        m = 60
        n = 100
        s = 25

        A = np.random.randn(m, n)
        x0 = np.zeros(n)
        nonzero_indices = np.random.choice(
            n, size=s, replace=False
        )
        x0[nonzero_indices] = np.random.randn(
            s
        )
        b = A.dot(x0)

        model_l1 = admm.Model()
        x_l1 = admm.Var(n)
        model_l1.addConstr(A @ x_l1 == b)
        model_l1.setObjective(admm.norm(x_l1, ord=1))
        model_l1.setOption(admm.Options.solver_verbosity_level, 3)
        model_l1.setOption(admm.Options.admm_max_iteration, 200000)
        model_l1.setOption(admm.Options.termination_absolute_error_threshold, 1e-6)
        model_l1.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model_l1.optimize()

        self.assertEqual(model_l1.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log det(S) + tr(S Y)
    # s.t.  sum(|S|) <= alpha,   S in PSD
    def test_Sparse_Inverse_Covariance_Selection_Comparison(self):
        np.random.seed(0)

        n = 6
        N = 100

        A = np.random.randn(n, n)
        A[scipy.sparse.rand(n, n, 0.85).todense().nonzero()] = 0
        Strue = A.dot(A.T) + 0.05 * np.eye(n)

        R = np.linalg.inv(Strue)

        y_sample = scipy.linalg.sqrtm(R).dot(np.random.randn(n, N))

        Y = np.cov(y_sample)

        alphas = [1.0, 0.1, 0.01]

        for alpha in alphas:
            model = admm.Model()
            S = admm.Var("S", n, n, PSD=True)

            obj = -admm.log_det(S) + admm.trace(S @ Y)
            model.setObjective(obj)
            model.addConstr(admm.sum(admm.abs(S)) <= alpha)

            model.setOption(admm.Options.admm_max_iteration, 10000)
            model.setOption(admm.Options.solver_verbosity_level, 3)
            model.setOption(admm.Options.termination_absolute_error_threshold, 1e-5)
            model.setOption(admm.Options.termination_relative_error_threshold, 1e-5)
            model.optimize()

            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   0.5 * ||y - x||_2^2 + lam * tv1d(x[1:] - x[:-1])
    #
    # where  tv1d is the 1D total-variation (L1 trend filtering) of the signal
    def test_L1_Trend_Filtering_SP500_Comparison(self):
        try:
            y = np.loadtxt(
                open("CompareCvxpy/Advanced_Applications/data/snp500.txt", "rb"),
                delimiter=",",
                skiprows=1,
            )
        except FileNotFoundError:
            print("S&P 500 data file not found, using synthetic data")
            np.random.seed(42)
            n = 300
            y = (
                np.cumsum(np.random.normal(0, 0.01, n))
                + np.sin(np.linspace(0, 6 * np.pi, n)) * 0.1
            )

        if y.size > 300:
            idx = np.linspace(0, y.size - 1, 300).astype(int)
            y = y[idx]
        n = y.size

        vlambda = 20

        model = admm.Model()
        x = admm.Var(n)
        dx = x[1:] - x[:-1]
        objective = 0.5 * admm.sum(admm.square(y - x)) + vlambda * admm.tv1d(dx, 1, 1)
        model.setObjective(objective)

        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_primal_dual_gap, 1e-3)
        model.setOption("penalty_param_auto", 1)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   0.5 * ||y - x||_2^2 + lam * tv1d(x[1:] - x[:-1])
    #
    # where  tv1d is the 1D total-variation (L1 trend filtering) of the signal
    def test_L1_Trend_Filtering_synthetic_Comparison(self):
        np.random.seed(42)
        n = 300
        y = (
            np.cumsum(np.random.normal(0, 0.01, n))
            + np.sin(np.linspace(0, 6 * np.pi, n)) * 0.1
        )

        n = y.size

        vlambda = 20

        model = admm.Model()
        x = admm.Var(n)

        dx = x[1:] - x[:-1]
        objective = 0.5 * admm.sum(admm.square(y - x)) + vlambda * admm.tv1d(dx, 1, 1)
        model.setObjective(objective)

        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-3)
        model.setOption(admm.Options.termination_relative_primal_dual_gap, 1e-3)
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(kl_div(alpha*W, alpha*(W + beta*P)))
    # s.t.  P, W >= 0,   sum(P) = P_tot,   sum(W) = W_tot
    #
    # where  P is power, W is bandwidth, kl_div is the generalized KL divergence
    def test_Optimal_Power_Allocation_Comparison(self):
        def optimal_power_ADMM(n, a_val, b_val, P_tot=1.0, W_tot=1.0):
            n = len(a_val)
            if n != len(b_val):
                print("alpha and beta vectors must have same length!")
                return "failed", np.nan, np.nan, np.nan
            model = admm.Model()

            P = admm.Var(n)
            W = admm.Var(n)
            alpha = admm.Param("alpha", n)
            beta = admm.Param("beta", n)
            alpha_value = np.array(a_val)
            beta_value = np.array(b_val)

            R = admm.kl_div(alpha * W, alpha * (W + beta * P))
            model.setObjective(admm.sum(R))
            model.addConstr(P >= 0)
            model.addConstr(W >= 0)
            model.addConstr(admm.sum(P) - P_tot == 0.0)
            model.addConstr(admm.sum(W) - W_tot == 0.0)

            model.setOption(admm.Options.solver_verbosity_level, 3)
            model.setOption(admm.Options.admm_max_iteration, 1000000)
            model.setOption(admm.Options.termination_absolute_error_threshold, 1e-8)
            model.setOption(admm.Options.termination_relative_error_threshold, 1e-8)
            model.optimize({"alpha": alpha_value, "beta": beta_value})

            return model.StatusString
        np.random.seed(42)
        test_cases = [
            {
                "num": 2,
                "a": np.array([0.1, 0.2]),
                "b": np.array([0.1, 0.2]),
                "P_tot": 1,
                "W_tot": 1.0,
            },
            {
                "num": 3,
                "a": np.array([0.1, 0.2, 0.3]),
                "b": np.array([0.1, 0.2, 0.3]),
                "P_tot": 0.5,
                "W_tot": 1.0,
            },
            {
                "num": 5,
                "a": np.arange(10, 15, 1) / (1.0 * 5),
                "b": np.arange(10, 15, 1) / (1.0 * 5),
                "P_tot": 0.5,
                "W_tot": 1.0,
            },
            {
                "num": 4,
                "a": np.array([0.1, 0.3, 0.5, 0.7]),
                "b": np.array([0.2, 0.4, 0.6, 0.8]),
                "P_tot": 1.0,
                "W_tot": 2.0,
            },
            {
                "num": 5,
                "a": np.arange(10, 5 + 10) / (1.0 * 5),
                "b": np.arange(10, 5 + 10) / (1.0 * 5),
                "P_tot": 0.5,
                "W_tot": 1,
            },
            {
                "num": 20,
                "a": np.arange(10, 20 + 10) / (1.0 * 20),
                "b": np.arange(10, 20 + 10) / (1.0 * 20),
                "P_tot": 0.5,
                "W_tot": 1,
            },
            {
                "num": 50,
                "a": np.arange(10, 50 + 10) / (1.0 * 50),
                "b": np.arange(10, 50 + 10) / (1.0 * 50),
                "P_tot": 0.5,
                "W_tot": 1,
            },
        ]
        for case in test_cases:
            status_admm = optimal_power_ADMM(
                case["num"], case["a"], case["b"], case["P_tot"], case["W_tot"]
            )
            self.assertEqual(status_admm, "SOLVE_OPT_SUCCESS")

    # min   sum_{t=0}^{T-1}  ||x[:,t+1]||_2^2 + ||u[:,t]||_2^2
    # s.t.  x[:,t+1] = A x[:,t] + B u[:,t],   ||u[:,t]||_inf <= 1,   x[:,T] = 0
    def test_Control_Comparison(self):
        np.random.seed(1)
        n = 8
        m = 2
        T = 50
        alpha = 0.2
        beta = 3
        A = np.eye(n) - alpha * np.random.rand(n, n)
        B = np.random.randn(n, m)
        x_0 = beta * np.random.randn(n)

        model = admm.Model()
        x_ADMM = admm.Var(n, T + 1)
        u_ADMM = admm.Var(m, T)
        cost = 0

        for t in range(T):
            cost += admm.sum(x_ADMM[:, t + 1] * x_ADMM[:, t + 1]) + admm.sum(
                u_ADMM[:, t] * u_ADMM[:, t]
            )
            model.addConstr(x_ADMM[:, t + 1] == A @ x_ADMM[:, t] + B @ u_ADMM[:, t])
            model.addConstr(admm.norm(u_ADMM[:, t], ord=np.inf) <= 1)

        model.addConstr(x_ADMM[:, T] == 0)
        model.addConstr(x_ADMM[:, 0] == x_0[:])
        model.setObjective(cost)

        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-6)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
