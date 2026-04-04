import admm
import numpy as np
import os
import unittest
import math

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
        model.setOption(admm.Options.admm_max_iteration, 10000000)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum_{t=0}^{T-1} (||x_{t+1}||_2^2 + ||u_t||_2^2)
    # s.t.  x_{t+1} = A x_t + B u_t,   t = 0, ..., T-1
    #       ||u_t||_inf <= 1,   x_T = 0,   x_0 = x_0
    def test_Control(self):
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
        model.addConstr(x_ADMM[:, 0] == x_0)
        model.setObjective(cost)

        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-6)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertLess(math.fabs(model.ObjVal - 2515.32), 1e-1)
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
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-6)
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
        model.setObjective(loss * (1 / m) + 0.5 * reg)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 0.676865), 0.01)
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
        nonzero_indices = np.random.choice(n, size=s, replace=False)
        x0[nonzero_indices] = np.random.randn(s)
        b = A.dot(x0)
        model = admm.Model()
        x_l1 = admm.Var(n)
        model.addConstr(A @ x_l1 == b)
        model.setObjective(admm.norm(x_l1, ord=1))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 50000)
        model.optimize()

        err = np.absolute(x_l1.X - x0).sum()
        self.assertLess(err, 1e-2)
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
        nonzero_indices = np.random.choice(n, size=s, replace=False)
        x0[nonzero_indices] = np.random.randn(s)
        b = A.dot(x0)

        model = admm.Model()
        W = np.ones(n)
        x_log = admm.Var(n)
        model.setObjective(W.T @ admm.abs(x_log))
        model.addConstr(A @ x_log == b)
        model.optimize()

        err = np.absolute(x_log.X - x0).sum()
        self.assertLess(err, 1e-2)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x^2 + |x| + 1
    # s.t.  x >= 100
    def test_bug_detect_checkUnbound(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        model.setObjective(admm.power(x1, 2) + admm.abs(x1) + 1)
        model.addConstr(x1 >= 100)
        model.setOption(admm.Options.solver_verbosity_level, 3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 100), 1e-3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 2), 1e-3)
        self.assertEqual(model.status, 1)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   exp(x)
    # s.t.  x >= 0
    def test_exp_x(self):

        model = admm.Model()
        x = admm.Var("x")
        model.setObjective(admm.exp(x))
        model.addConstr(x >= 0)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 1), 1e-3)
        self.assertEqual(model.status, 1)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   4 + x
    # s.t.  x = 0
    def test_presolved_detect_3(self):

        model = admm.Model()
        x = admm.Var("x")
        model.setObjective(4 + x)
        model.addConstr(x == 0)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 4), 1e-3)
        self.assertEqual(model.status, 1)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(diag(x))
    # s.t.  x = 2
    def test_presolved_diag_x(self):

        model = admm.Model()
        x = admm.Var("x", 2)
        model.setObjective(admm.sum(admm.diag(x)))
        model.addConstr(x == 2)
        model.setOption(admm.Options.admm_max_iteration, 100000)

        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 4), 1e-3)
        self.assertEqual(model.status, 1)
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

        self.assertLess(math.fabs(model.objval - 4), 1e-3)
        self.assertEqual(model.status, 1)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 8), 1e-3)
        self.assertEqual(model.status, 1)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(diag(x)) + tr(X)
    # s.t.  x = 2,   X = diag(x)
    def test_diag_Diag1(self):
        model = admm.Model()
        x = admm.Var("x", 2)
        X = admm.Var("X", 2, 2)
        model.setObjective(admm.sum(admm.diag(x)) + admm.trace(X))
        model.addConstr(x == 2)
        model.addConstr(X == admm.diag(x))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 8), 1e-3)
        self.assertEqual(model.status, 1)

    # min   sum(diag(X)) + sum(diag(x))
    # s.t.  X = 2,   x = diag(X)
    def test_diag_Diag2(self):
        X = admm.Var("X", 2, 2)
        x = admm.Var("x", 2)
        model = admm.Model()
        model.setObjective(admm.sum(admm.diag(X)) + admm.sum(admm.diag(x)))
        model.addConstr(X == 2)
        model.addConstr(x == admm.diag(X))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 8), 1e-3)
        self.assertEqual(model.status, 1)

    # min   sum(diag(x))
    # s.t.  x = 2
    def test_diag_Diag3(self):
        model = admm.Model()
        x = admm.Var("x", 2)
        model.setObjective(admm.sum(admm.diag(x)))
        model.addConstr(x == 2)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 4), 1e-3)
        self.assertEqual(model.status, 1)

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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 1)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(X)
    # s.t.  X = Y Q,   Y = P
    def test_mat_multiply1(self):
        model = admm.Model()
        X = admm.Var("X", 2, 2)
        Y = admm.Var("Y", 2, 2)
        P = np.array([[1, 1], [1, 1]])
        Q = np.array([[1, 2], [3, 4]])
        model.setObjective(admm.sum(X))
        model.addConstr(X == Y @ Q)
        model.addConstr(Y == P)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 1)

    # min   |x + z + 1|
    # s.t.  x = 2,   z >= 2
    def test_abs_sum_x_z_1(self):

        model = admm.Model()
        x = admm.Var("x")
        z = admm.Var("z")
        model.setObjective(admm.abs(x + z + 1))
        model.addConstr(x == 2)
        model.addConstr(z >= 2)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 1)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x1^2
    # s.t.  x1 >= 1
    def test_square(self):

        model = admm.Model()
        x1 = admm.Var("x1")
        model.setObjective(admm.square(x1))
        model.addConstr(x1 >= 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 1)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 9), 1e-1)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -|x + z|
    # s.t.  x = 1
    # Check detection of unboundedness (concave objective, free variable z).
    def test_unbound_detect_abs_sum(self):
        model = admm.Model()
        x = admm.Var("x")
        z = admm.Var("z")
        model.setObjective(-admm.abs(x + z))
        model.addConstr(x == 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 3)

    # min   -exp(x1 + x2)
    # s.t.  x1 + 2 >= 1,   2 x2 + 1 <= 1
    # Check detection of unboundedness (concave objective).
    def test_unbound_detect_exp_sum(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        model.setObjective(-admm.exp(x1 + x2))
        model.addConstr(x1 + 2 >= 1)
        model.addConstr(2 * x2 + 1 <= 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 3)

    # min   -log(x)
    # s.t.  x >= 1
    # Check detection of unboundedness (x -> +inf).
    def test_unbound_detect_log(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        model.setObjective(-admm.log(x1))
        model.addConstr(x1 >= 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 3)

    # min   log(x1 + x2)
    # s.t.  2 x1 + 3 >= 0
    # Check detection of unboundedness (x2 free, x1+x2 -> 0+ makes log -> -inf).
    def test_unbound_detect_log_sum(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        model.setObjective(admm.log(x1 + x2))
        model.addConstr(2 * x1 + 3 >= 0)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 3)

    # min   sqrt(-x) - log(z)
    # s.t.  x - z = 0
    # Check detection of infeasibility (sqrt(-x) requires x <= 0, log(z) requires z > 0).
    def test_infeasible_detect_sqrt_x_minus_log_x(self):
        model = admm.Model()
        x = admm.Var("x")
        z = admm.Var("z")
        model.setObjective(admm.sqrt(-x) - admm.log(z))
        model.addConstr(x - z == 0)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertIn(model.status, (2, 4))

    # min   3 x + 1
    # s.t.  x <= 0,   z <= 1,   x + z = 1
    # Check detection of infeasibility (conflicting linear constraints).
    def test_infeasible_detect_conflict_linear_constraint(self):
        model = admm.Model()
        x = admm.Var("x")
        z = admm.Var("z")
        model.setObjective(3 * x + 1)
        model.addConstr(x + 0 <= 0)
        model.addConstr(z <= 1)
        model.addConstr(x + z == 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 2)

    # min   0
    # s.t.  x1 + 2 x2 = 1,   2 x1 + 4 x2 = 4
    # Check detection of infeasibility (inconsistent parallel equality constraints).
    def test_infeasible_detect_conflict_linear_constraint1(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        model.setObjective(0)
        model.addConstr(x1 + 2 * x2 == 1)
        model.addConstr(2 * x1 + 4 * x2 == 4)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 2)

    # min   0
    # s.t.  x1 + 2 x2 = 2,   x1 >= 3,   x2 >= 0
    # Check detection of infeasibility (x1 >= 3 contradicts x1 + 2 x2 = 2, x2 >= 0).
    def test_infeasible_detect_conflict_linear_constraint2(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        model.setObjective(0)
        model.addConstr(x1 + 2 * x2 == 2)
        model.addConstr(x1 >= 3)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 2)

    # min   -log(x1 + x2)
    # s.t.  x1 = 1,   x2 <= -2
    # Check detection of infeasibility (x1 + x2 <= -1 < 0 for log domain).
    def test_infeasible_detect_log_sum(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        model.setObjective(-admm.log(x1 + x2))
        model.addConstr(x1 == 1)
        model.addConstr(x2 <= -2)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 2)

    # min   exp(x)
    # s.t.  x = z
    def test_minimize_exp_x_over_essentially_unconstraint(self):

        model = admm.Model()
        x = admm.Var("x")
        z = admm.Var("z")
        model.setObjective(admm.exp(x))
        model.addConstr(x == z)
        model.setOption(admm.Options.admm_max_iteration, 1000000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.status, 1)
        self.assertLess(math.fabs(model.objval - 0), 1e-3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval), 1e-3)
        constraint_val = np.sqrt(x1.X.T @ x1.X + x2.X**2 + x3.X.T @ x3.X)
        self.assertLessEqual(constraint_val, x4.X + 1e-3)
        self.assertLess(math.fabs(x2.X - 1), 1e-3)
        self.assertLess(math.fabs(x4.X - 1), 1e-3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertTrue(all(math.fabs(val) < 1e-3 for val in x1.X))
        self.assertLess(math.fabs(x2.X - 1), 1e-3)
        self.assertTrue(all(math.fabs(val) < 1e-3 for val in x3.X))
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(x1.X.T @ x1.X + x2.X * x2.X + x3.X.T @ x3.X, 1.001)
        self.assertLess(math.fabs(x2.X - 1), 1e-3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertTrue(all(math.fabs(val + 0.5) < 1e-3 for val in x1.X))
        self.assertLess(math.fabs(x2.X - 1), 1e-3)
        self.assertTrue(all(math.fabs(val + 0.25) < 1e-3 for val in x3.X))
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 9), 1e-1)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 53.77502), 1e-1)
        self.assertLess(math.fabs(X.X[0][0] - 2), 0.5)
        self.assertLess(math.fabs(Y.X[0][0] - 1), 1e-3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 41.87965263776742), 1e-1)
        self.assertLess(math.fabs(X.X[0][0] - 0), 1e-3)
        self.assertLess(math.fabs(Y.X[0][0] - 1), 1e-3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(max(1 - x, 0)^2)
    def test_squared_hinge_detect(self):

        model = admm.Model()
        x = admm.Var("x")
        model.setObjective(admm.sum(admm.square(admm.maximum(1 - x, 0))))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval), 1e-3)
        self.assertLess(math.fabs(x.X - 1), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x^2
    # s.t.  -1 <= x <= 1
    def test_square_detect(self):

        model = admm.Model()
        x = admm.Var("x")
        model.setObjective(x * x)
        model.addConstr(x >= -1)
        model.addConstr(x <= 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval), 1e-3)
        self.assertLess(math.fabs(x.X), 1e-3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 1000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 1.0616) / 1.0616, 1e-3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 2000)
        model.optimize()

        self.assertLess(
            math.fabs(model.objval - 3.182313603032398) / 3.182313603032398, 1e-3
        )
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -x1 - y2
    # s.t.  x1 + 2 y2 <= 4,   2 x1 - y2 <= 2
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

    # min   -(5 x1 + 4 x2 + 3 x3)
    # s.t.  2 x1 + 3 x2 + x3 <= 5,   4 x1 + x2 + 2 x3 <= 11
    #       3 x1 + 4 x2 + 2 x3 <= 8,   x >= 0
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
        model.setOption(admm.Options.admm_max_iteration, 10000000)
        model.optimize()

        self.assertLess(model.objval, -12.9)
        self.assertGreater(model.objval, -13.1)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(model.objval, 14.2)
        self.assertGreater(model.objval, 14)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(model.objval, 14.2)
        self.assertGreater(model.objval, 14)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(model.objval, 18.1)
        self.assertGreater(model.objval, 18)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(model.objval, 0.051)
        self.assertGreater(model.objval, 0.049)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   exp(x1-1) + exp(x2-2) + exp(2 x3+3) + 5 x3 + exp(x4-4) + exp(x5+1) + x5^2
    # s.t.  x2 + x3 + x4 + x5 = -10,   x1 + x2 - x3 <= 8
    #       x2 + x3 - x5 >= 6,   x1 - x2 - x4 + x5 <= 6
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(model.objval, 13.23)
        self.assertGreater(model.objval, 13.10)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   exp(x1-1) + exp(x2-2) + exp(2 x3+3) + 5 x3 + exp(x4-4) + exp(x5+1) + x5^2
    # s.t.  x2 + x3 + x4 + x5 = -10,   x1 + x2 - x3 <= 8
    #       x2 + x3 - x5 >= 6,   x1 - x2 - x4 + x5 <= 6
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(model.objval, 13.23)
        self.assertGreater(model.objval, 13.10)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(model.objval, 0.51)
        self.assertGreater(model.objval, 0.49)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||A x - b||_2 + ||x||_1
    def test_ex9_norm2(self):
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = np.array([16, 40])

        model = admm.Model()
        x = admm.Var(4)
        model.setObjective(admm.norm(A @ x - b, ord=2) + admm.norm(x, ord=1))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 6) / 6.0, 1e-3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 0.009412217910640164), 1e-4)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 0.009412217910640164), 1e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log(x1 + 1) + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 0
    def test_neg_log(self):

        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        model.setObjective(-admm.log(x1 + 1) + x2 + 5)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(
            math.fabs(model.objval - 4.30685282018775) / 4.30685282018775, 1e-3
        )
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x0[0,0] + x1[0,0] + 5
    # s.t.  x0[0,:] - 2 x1[0,:] <= 4,   2 x0 + x1 <= 20,   x0 >= 0,   x1 >= 0
    def test_ex12_slicing(self):

        model = admm.Model()
        x0 = admm.Var(2, 2)
        x1 = admm.Var(2, 2)
        model.setObjective(x0[0, 0] + x1[0, 0] + 5)
        model.addConstr(x0[0, :] - 2 * x1[0, :] <= 4)
        model.addConstr(2 * x0 + x1 <= 20)
        model.addConstr(x0 >= 0)
        model.addConstr(x1 >= 0)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 5) / 5, 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   p1^T x0 + p2^T x1 + 5
    # s.t.  x0 + 2 x1 <= 4,   2 x0 + x1 <= 2,   x0 >= 0,   x1 >= 0
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 5) / 5, 2e-4)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 15) / 15.0, 1e-4)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 200) / 200, 2e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(p1 x1 - p2 x2 + 5)
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 0
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 4) / 4, 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x[0] + x[1]
    # s.t.  x[0] = 2,   x[1] = 6
    def test_index_ex1(self):

        model = admm.Model()
        x = admm.Var(5)
        model.setObjective(admm.sum(x[0:2]))
        model.addConstr(x[0] == 2)
        model.addConstr(x[1] == 6)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 8) / 8, 1e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   |x1 + x2 + 1|
    # s.t.  x1 - x2 >= 5,   x1 + x2 >= 5,   x1 >= 0,   x2 >= 0
    def test_linear_expr_ex1(self):

        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        model.setObjective(admm.abs(x1 + x2 + 1))
        model.addConstr(x1 - x2 >= 5)
        model.addConstr(x1 + x2 >= 5)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(model.objval, 6.1)
        self.assertGreater(model.objval, 5.999)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 1.5) / 1.5, 1e-4)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 1) / 1, 1e-2)
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
        model.setOption(admm.Options.admm_max_iteration, 10000000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 300) / 300, 2e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log det(X) - log det(Y)
    # s.t.  X + Y = 5 I,   X in S+^3,   Y in S+^3
    def test_mat_log_det_ex(self):

        model = admm.Model()
        x = admm.Var(3, 3, PSD=True)
        y = admm.Var(3, 3, PSD=True)
        model.setObjective(-admm.log_det(x) - admm.log_det(y))
        model.addConstr(x + y == 5 * np.eye(3))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(
            math.fabs(model.objval + 5.497744403407283) / 5.497744403407283, 1e-4
        )
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log det(X) - log det(Y)
    # s.t.  X + Y = 5 I,   X in S+^3,   Y in S+^3
    def test_mat_log_det_ex1(self):

        model = admm.Model()
        x = admm.Var(3, 3, PSD=True)
        y = admm.Var(3, 3, PSD=True)
        model.setObjective(-admm.log_det(x) - admm.log_det(y))
        model.addConstr(x + y == 5 * np.eye(3))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(
            math.fabs(model.objval + 5.497744403407283) / 5.497744403407283, 1e-4
        )
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   max(x1 + 1, 2) + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 1
    def test_max_fun_shift_error(self):

        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        model.setObjective(admm.maximum(x1 + 1, 2) + x2 + 5)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 8) / 8, 1e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x[0] + x[1]
    # s.t.  x[0] + 2 x[1] = 1,   2 x[0] - 3 x[1] >= 3,   x[1] >= -2
    def test_mergeConstrex1(self):

        model = admm.Model()
        x = admm.Var(2)
        model.setObjective(x[0] + x[1])
        model.addConstr(x[0] + 2 * x[1] == 1)
        model.addConstr(2 * x[0] - 3 * x[1] >= 3)
        model.addConstr(x[1] >= -2)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(
            math.fabs(model.objval - 1.1428571445959526) / 1.1428571445959526, 1e-4
        )
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 1) / 1, 2e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -p1^T x1 - p2^T x2 + 3
    # s.t.  p3^T x1 + p4^T x2 <= 1,   p5^T x1 + p6^T x2 <= 2,   x1 >= 0,   x2 >= 0
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval + 1) / (-1), 2e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   (x1 + x2)^2
    def test_quadratic_ex1(self):

        model = admm.Model()
        x1 = admm.Var()
        x2 = admm.Var()
        model.setObjective(admm.square(x1 + x2))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval), 1e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1 + x2||_2^2
    def test_quadratic_ex2(self):

        model = admm.Model()
        x1 = admm.Var(2)
        x2 = admm.Var(2)
        model.setObjective(x1.T @ x1 + 2 * x1.T @ x2 + x2.T @ x2)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval), 1e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x)
    # s.t.  x[1] >= 0,   x[3] >= 0,   x[0] >= 1,   x[2] >= 1,   x[4] >= 1
    def test_rowset_ex1(self):

        model = admm.Model()
        x = admm.Var(5)
        model.setObjective(admm.sum(x))
        model.addConstr(x[1] >= 0)
        model.addConstr(x[3] >= 0)
        model.addConstr(x[0] >= 1)
        model.addConstr(x[2] >= 1)
        model.addConstr(x[4] >= 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 3), 1e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -x - y
    # s.t.  x + 2 y <= 4,   2 x - y <= 2
    def test_SCALAR_kIdentity_0(self):

        model = admm.Model()
        x = admm.Var()
        y = admm.Var()
        model.setObjective(-x - y)
        model.addConstr(x + 2 * y <= 4)
        model.addConstr(2 * x - y <= 2)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval + 2.8) / 2.8, 1e-3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 200) / 200, 2e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log det(X) + tr(S X) + mu * ||X||_1
    # s.t.  X in S+^3
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 6.6894390191919975), 1e-1)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 1), 2e-4)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 1), 2e-4)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 8) / 8, 1e-4)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 0.03692370402861033), 2e-4)
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
        model.optimize()

        self.assertLess(math.fabs(model.objval + 3.32) / 3.32, 3e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   log(1 + exp(2 x1 + 1)) + x1^2 + |2 x2 - 5| + 0.5 x2^2
    # s.t.  2 x1 + 3 x2 = 4,   x1 - 2 x2 <= 10
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(model.objval, 4)
        self.assertGreater(model.objval, 3.98)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum of absolute-value, linear, and quadratic terms in x1,...,x5
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(
            math.fabs(model.objval - 13.849107142857145) / 13.849107142857145, 3e-4
        )
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum of logistic and quadratic penalty terms in x1,...,x6
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log(x1) + x1 - log(x2-1) - log(2 x3-1) + 2(x3-1)^2 + entropy(x4-1) + (x4-2)^2 + entropy(x5) + 2 x5
    # s.t.  x1 + 2 x2 + 3 x3 = 5,   x1 + x4 - x5 = 0,   x2 - x3 + x4 <= 6,   x2 + x5 <= 4
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.optimize()

        self.assertLess(
            math.fabs(model.objval - 7.420222523587223) / 7.420222523587223, 3e-4
        )
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.optimize()

        self.assertLess(
            math.fabs(model.objval - 3.5516120024347133) / 3.5516120024347133, 3e-4
        )
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   log(x)
    # s.t.  0.5 <= x <= 1
    def test_wbh17_Log(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        model.setObjective(admm.log(x1))
        model.addConstr(x1 <= 1)
        model.addConstr(x1 >= 0.5)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertLess(math.fabs(x1.X - 0.5), 1e-2)
        self.assertEqual(model.status, 1)

    # min   -log(x)
    # s.t.  0.5 <= x <= 1
    def test_wbh17_NegLog(self):

        model = admm.Model()
        x1 = admm.Var("x1")
        model.setObjective(-admm.log(x1))
        model.addConstr(x1 <= 1)
        model.addConstr(x1 >= 0.5)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x1^2
    # s.t.  x1 >= 1
    def test_wbh17_Square(self):

        model = admm.Model()
        x1 = admm.Var("x1")
        model.setObjective(admm.square(x1))
        model.addConstr(x1 >= 1)
        model.optimize()

        self.assertEqual(model.status, 1)
        self.assertLess(math.fabs(model.objval - 1), 3e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   (x1-1)^2 + 0.5(x2-2)^2 + (x3-3)^2/3 + 0.25(x4-4)^2 + 0.2(x5-5)^2 + (x6-6)^2/6
    # s.t.  x1+x2+x3+x4+x5+x6 = 12,   x1+x2-x3 >= 3
    #       x4+x5-x6 >= 5,   x4-x5+x6 <= 8
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 1000000)
        model.optimize()

        self.assertLess(
            math.fabs(model.objval - 6.098039215686274) / 6.098039215686274, 3e-4
        )
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   (x1-1)^2 + 0.5(x2-2)^2 + (x3-3)^2/3 + 0.25(x4-4)^2 + 0.2(x5-5)^2 + (x6-6)^2/6
    # s.t.  x1+x2+x3+x4+x5+x6 = 12,   x1+x2+x3 >= 5,   x2+x3+x4 <= 4
    #       x4+x5+x6 >= 8,   x1+x5+x6 <= 7
    # Check detection of infeasibility (over-constrained system).
    def test_wbh19(self):
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
        model.addConstr(x1 + x2 + x3 >= 5)
        model.addConstr(x2 + x3 + x4 <= 4)
        model.addConstr(x4 + x5 + x6 >= 8)
        model.addConstr(x1 + x5 + x6 <= 7)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 1000000)
        model.optimize()

        self.assertIn(model.status, (2, 4))

    # min   -x1 - x2
    # s.t.  x1 + 2 x2 <= 4,   2 x1 - x2 <= 2
    def test_scalar(self):

        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        p = 4
        model.setObjective(-x1 - x2)
        model.addConstr(x1 + 2 * x2 <= p)
        model.addConstr(2 * x1 - x2 <= 2)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100)
        model.optimize()

        self.assertLess(math.fabs(model.objval + 2.8), 1e-3)
        self.assertLess(math.fabs(x1.X - 1.6), 1e-3)
        self.assertLess(math.fabs(x2.X - 1.2), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # max   x1 + x2 + 100
    # s.t.  x1 + 2 x2 <= 4,   2 x1 - x2 <= 2
    def test_scalar_max(self):

        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        p = 4
        model.setObjective(x1 + x2 + 100)
        model.ModelSense = -1
        model.addConstr(x1 + 2 * x2 <= p)
        model.addConstr(2 * x1 - x2 <= 2)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 102.8), 1e-3)
        self.assertLess(math.fabs(x1.X - 1.6), 1e-3)
        self.assertLess(math.fabs(x2.X - 1.2), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   1^T x1 + 1^T x2 + 5
    # s.t.  1^T x1 + 2 * 1^T x2 <= 4,   2 * 1^T x1 + 1^T x2 <= 2,   x1 >= 0,   x2 >= 0
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 5), 1e-3)
        self.assertLess(math.fabs(x1.X[0][0] - 0), 1e-3)
        self.assertLess(math.fabs(x2.X[0][0] - 0), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(p1 x1 - p2 x2 + 5)
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 0
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 4), 1e-3)
        self.assertLess(math.fabs(x1.X[0][0] - 0), 1e-3)
        self.assertLess(math.fabs(x2.X[0][0] - 2), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x1 p1 - x2 + 5)
    # s.t.  x1 + x2 p2 <= 4,   x1 >= 0,   x2 >= 0
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 12), 1e-2)
        self.assertLess(math.fabs(x1.X[0][0] - 0), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x1[0,0] + x2[0,0] + 5
    # s.t.  x1[0,:] - 2 x2[0,:] <= 4,   2 x1 + x2 <= 20,   x1 >= 0,   x2 >= 1
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 6), 1e-2)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   exp(x1 + 1) + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 1
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 8.71828), 2e-3)
        self.assertLess(math.fabs(x1.X - 0), 1e-3)
        self.assertLess(math.fabs(x2.X - 1), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 50.30969), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 0), 0.5)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1 + 3 x2 + 1||_2
    def test_no_constr(self):

        model = admm.Model()
        x1 = admm.Var("x1", 2)
        x2 = admm.Var("x2", 2)
        model.setObjective(admm.norm(x1 + 3 * x2 + 1, ord=2))
        model.optimize()

        self.assertLess(math.fabs(model.objval), 1e-3)
        constraint_val = np.linalg.norm(x1.X + 3 * x2.X + 1, ord=2)
        self.assertLess(constraint_val, 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   |x1 + 1| + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 1
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 7), 1e-2)
        self.assertLess(math.fabs(x1.X - 0), 1e-2)
        self.assertLess(math.fabs(x2.X - 1), 1e-3)
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
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        objval = np.sum(np.abs(x1.X + 1)) + 5 * 6 + np.sum(x2.X)
        self.assertLess(math.fabs(objval - 52), 1e-2)
        self.assertLess(math.fabs(x1.X[0][0] - 2), 1e-2)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-2)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   |x1 + 1| + x2 + 2
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 1
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 4), 1e-2)
        self.assertLess(math.fabs(x1.X - 0), 1e-2)
        self.assertLess(math.fabs(x2.X - 1), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 3), 1e-3)
        self.assertLess(math.fabs(x1.X[0] - 0), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 16.348469), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 2), 1e-3)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x1 + 1||_* + 5 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 2,   x2 >= 1
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 16.348486936028), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 2), 1e-3)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||A - Y x||_F
    # s.t.  x >= 0
    def test_norm_fro_2(self):

        model = admm.Model()
        x = admm.Var("x", 2, 3)
        Y = np.array([[1, 2], [1, 2], [1, 2]])
        A = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])
        model.setObjective(admm.norm(A - Y @ x, ord="fro"))
        model.addConstr(x >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 14.696938456699), 1e-3)
        self.assertLess(math.fabs(x.X[0][0] - 1.59992253), 1e-3)
        self.assertLess(math.fabs(x.X[0][1] - 1.99990317), 1e-3)
        self.assertLess(math.fabs(x.X[0][2] - 2.3998838), 1e-3)
        self.assertLess(math.fabs(x.X[1][0] - 3.19984507), 1e-3)
        self.assertLess(math.fabs(x.X[1][1] - 3.99980633), 1e-3)
        self.assertLess(math.fabs(x.X[1][2] - 4.7997676), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   x1 + x2
    # s.t.  x2 >= 1,   ||x1||_inf <= 1
    def test_linf(self):

        model = admm.Model()
        x1 = admm.Var("x1", 1)
        x2 = admm.Var("x2", 1)
        model.setObjective(admm.sum(x1 + x2))
        model.addConstr(x2 >= 1)
        model.addConstr(admm.norm(x1, ord=np.inf) <= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval), 1e-3)
        self.assertLess(math.fabs(x1.X[0] + 1), 1e-3)
        self.assertLess(math.fabs(x2.X[0] - 1), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 0), 1e-3)
        self.assertLess(math.fabs(x1.X[0] + 1), 1e-3)
        self.assertLess(math.fabs(x2.X[0] - 1), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 0), 1e-3)
        self.assertLess(math.fabs(x1.X[0] + 1), 1e-3)
        self.assertLess(math.fabs(x2.X[0] - 1), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 1), 2e-3)
        self.assertLess(math.fabs(x1.X[0] - 0), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 5.1961), 1e-3)
        self.assertLess(math.fabs(x1.X[0][0] - 2), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   max(x1 + 1, 2) + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 1
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 8), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -min(x1 + 1, 2) + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 1
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 4.5), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 24), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval + 24), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
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
        model.setOption(admm.Options.admm_max_iteration, 100)
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval + 1), 1e-3)
        self.assertLess(math.fabs(x1.X[0] - 0), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(max_i(x1 + 1)) + sum(x2)
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
        model.optimize()

        self.assertLess(math.fabs(model.objval - 5), 1e-2)
        self.assertLess(math.fabs(x1.X[0][0] - 0), 1e-3)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   max_i(-(x1 + 1)) + sum(x2)
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
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   entropy(x1 + 1) + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 1
    def test_entropy(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        p = 5
        model.setObjective(admm.entropy(x1 + 1) + x2 + p)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 6), 1e-2)
        self.assertLess(math.fabs(x1.X - 0), 1e-3)
        self.assertLess(math.fabs(x2.X - 1), 1e-3)

    # min   sum(entropy(x1 + 1)) + 30 + sum(x2)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 53.77502), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 2), 0.5)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   1 / (x1 + 1) + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 1
    def test_inverse(self):
        model = admm.Model()
        x1 = admm.Var("x1", 1, 1)
        x2 = admm.Var("x2", 1, 1)
        model.setObjective(1 / (x1 + 1) + x2 + 5)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 6.6666666), 1e-3)
        self.assertLess(math.fabs(np.asarray(x1.X).reshape(-1)[0] - 0.5), 1e-3)
        self.assertLess(math.fabs(np.asarray(x2.X).reshape(-1)[0] - 1), 1e-3)

    # min   huber(x1 + 1, 1) + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 1
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 6.5), 1e-2)
        self.assertLess(math.fabs(x1.X[0] - 0), 1e-2)
        self.assertLess(math.fabs(x2.X[0] - 1), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 49), 1e-1)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 12), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 2), 0.5)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 0.5)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 49), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 2), 0.5)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   (x1 + 1)^2 + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 1
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 7), 1e-2)
        self.assertLess(math.fabs(x1.X[0] - 0), 1e-3)
        self.assertLess(math.fabs(x2.X[0] - 1), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 3), 1e-2)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -log(x1 + 1) + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 0
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 4.30685), 1e-3)
        self.assertLess(math.fabs(x1.X - 1), 1e-3)
        self.assertLess(math.fabs(x2.X - 0), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   log(1 + exp(x1 + 1)) + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 0
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   log(1 + exp(x1 + 1)) + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 0
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   (x1 + 1)^2 + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 1,   x2 >= 0
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 9), 1e-2)
        self.assertLess(math.fabs(x1.X - 1), 1e-2)
        self.assertLess(math.fabs(x2.X - 0), 1e-3)
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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 40.0001081402754), 1e-2)
        self.assertLess(math.fabs(x1.X[0][0] - 0), 1e-3)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum((x1 + 1)^(4/3)) + 30 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 0,   x2 >= 1
    def test_power_3(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(admm.power(x1 + 1, 4 / 3)) + 30 + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 39.999999961), 1e-2)
        self.assertLess(math.fabs(x1.X[0][0] - 0), 1e-3)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-3)

    # min   sum(-(x1 + 1)^(1/2)) + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 0,   x2 >= 1
    def test_power_4(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(-admm.power(x1 + 1, 1 / 2)) + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval + 23.1591143) / 23.1591143, 1e-2)

    # min   sum(-(x1 + 1)^(1/3)) + 30 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 0,   x2 >= 1
    def test_power_5(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(-admm.power(x1 + 1, 1 / 3)) + 30 + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 20.2919757), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 9), 1e-1)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-1)

    # min   sum(-(x1 + 1)^(1/5)) + 30 + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 0,   x2 >= 1
    def test_power_6(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(-admm.power(x1 + 1, 1 / 5)) + 30 + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 24.152281335), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 9), 1e-1)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-1)

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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval + 8.944208230), 1e-2)
        self.assertLess(math.fabs(x1.X[0][0] - 5), 1e-2)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-2)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(-sqrt(x1 + 1)) + sum(x2)
    # s.t.  x1 - x2 p <= 4,   x1 >= 0,   x2 >= 1
    def test_sqrt_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 3)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([[1, 2, 3], [4, 5, 6]])
        model.setObjective(admm.sum(-admm.sqrt(x1 + 1)) + admm.sum(x2))
        model.addConstr(x1 - x2 @ p <= 4)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.solver_message_print_interval, 100)
        model.optimize()

        self.assertLess(math.fabs(model.objval + 23.1591143) / 23.1591143, 1e-2)

    # min   delta_{[1,3]}(x1 + 1) + x2 + 5
    # s.t.  x1 + 2 x2 <= 4,   2 x1 + x2 <= 2,   x1 >= 0,   x2 >= 0
    def test_indicator(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x2 = admm.Var("x2")
        p = 5
        model.setObjective(admm.inrange(x1 + 1, 1, 3) + x2 + p)
        model.addConstr(x1 + 2 * x2 <= 4)
        model.addConstr(2 * x1 + x2 <= 2)
        model.addConstr(x1 >= 0)
        model.addConstr(x2 >= 0)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 5), 1e-3)

    # min   -log det(x1)
    # s.t.  x1 + x2 <= 6,   x1 >= 2,   x2 >= 1
    def test_logdet_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2)
        x2 = admm.Var("x2", 2, 2)
        model.setObjective(-admm.log_det(x1))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval + 3.04452), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 5), 0.5)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 1e-3)

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
        model.setOption(admm.Options.penalty_param_auto, 1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 4), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 2), 0.5)
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
        model.optimize()

        self.assertLess(math.fabs(model.objval - 15), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 2), 0.5)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 0.5)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(diag(x1))
    # s.t.  x1 + x2 <= 6,   x1 >= 2,   x2 >= 1
    def test_diag_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2)
        x2 = admm.Var("x2", 2, 2)
        model.setObjective(admm.sum(admm.diag(x1)))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 4), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 2), 0.5)

    # min   sum(diag(x1)) + sum(x2)
    # s.t.  x1 + x2 <= 6,   x1 >= 2,   x2 >= 1
    def test_diag_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, 3)
        x2 = admm.Var("x2", 3, 3)
        model.setObjective(admm.sum(admm.diag(x1)) + admm.sum(x2))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 15), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 2), 0.5)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 0.5)

    # min   sum(diag(p) + x1 + x2)
    # s.t.  x1 + x2 <= 6,   x1 >= 2,   x2 >= 1
    def test_diag_3(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2)
        x2 = admm.Var("x2", 2, 2)
        p = np.array([1, 2])
        model.setObjective(admm.sum(admm.diag(p) + x1 + x2))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x1 >= 2)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 15), 1e-1)
        self.assertLess(math.fabs(x1.X[0][0] - 2), 0.5)
        self.assertLess(math.fabs(x2.X[0][0] - 1), 0.5)

    # min   sum(kl_div(x, y))
    # s.t.  x <= 2,   y <= 1
    def test_kl_div_1(self):
        model = admm.Model()
        x = admm.Var("x", 2, 3)
        y = admm.Var("y", 2, 3)
        model.setObjective(admm.sum(admm.kl_div(x, y)))
        model.addConstr(x <= 2)
        model.addConstr(y <= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval + 2.207276643), 1e-3)
        self.assertLess(math.fabs(x.X[0][0] - 0.36785), 1e-3)
        self.assertLess(math.fabs(y.X[0][0] - 1), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x1) + sum(x2)
    # s.t.  x1 + x2 <= 6,   x1 >= -1,   x2 >= 1,   x1 diagonal
    def test_diag_4(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, 3, diag=True)
        x2 = admm.Var("x2", 3, 3)
        model.setObjective(admm.sum(x1) + admm.sum(x2))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x1 >= -1)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 6), 1e-2)
        x1_ans = np.diag([-1, -1, -1])
        x2_ans = np.ones((3, 3))
        for i in range(3):
            for j in range(3):
                self.assertLess(math.fabs(x1.X[i][j] - x1_ans[i][j]), 2e-3)
                self.assertLess(math.fabs(x2.X[i][j] - x2_ans[i][j]), 2e-3)

    # min   sum(x1) + sum(x2)
    # s.t.  x1 + x2 <= 6,   x1 >= -1,   x2 >= 1,   x1 >= 0  (nonneg)
    def test_nonneg_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, 3, nonneg=True)
        x2 = admm.Var("x2", 3, 3)
        model.setObjective(admm.sum(x1) + admm.sum(x2))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x1 >= -1)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 9), 1e-2)
        x1_ans = np.zeros((3, 3))
        x2_ans = np.ones((3, 3))
        for i in range(3):
            for j in range(3):
                self.assertLess(math.fabs(x1.X[i][j] - x1_ans[i][j]), 2e-3)
                self.assertLess(math.fabs(x2.X[i][j] - x2_ans[i][j]), 2e-3)

    # min   -sum(x1) + sum(x2)
    # s.t.  x1 + x2 >= 6,   x2 >= 7,   x1 <= 0  (nonpos)
    def test_nonpos_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, 3, nonpos=True)
        x2 = admm.Var("x2", 3, 3)
        model.setObjective(-admm.sum(x1) + admm.sum(x2))
        model.addConstr(x1 + x2 >= 6)
        model.addConstr(x2 >= 7)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 63), 1e-2)
        x1_ans = np.zeros((3, 3))
        x2_ans = 7 * np.ones((3, 3))
        for i in range(3):
            for j in range(3):
                self.assertLess(math.fabs(x1.X[i][j] - x1_ans[i][j]), 2e-3)
                self.assertLess(math.fabs(x2.X[i][j] - x2_ans[i][j]), 2e-3)

    # min   sum(x1) + sum(x2)
    # s.t.  x1 + x2 >= 6,   x1 >= -1,   x2 >= 7,   x1 >= 0  (nonneg)
    def test_pos_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, 3, nonneg=True)
        x2 = admm.Var("x2", 3, 3)
        model.setObjective(admm.sum(x1) + admm.sum(x2))
        model.addConstr(x1 + x2 >= 6)
        model.addConstr(x1 >= -1)
        model.addConstr(x2 >= 7)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 63), 2e-2)
        x1_ans = np.zeros((3, 3))
        x2_ans = 7 * np.ones((3, 3))
        for i in range(3):
            for j in range(3):
                self.assertLess(math.fabs(x1.X[i][j] - x1_ans[i][j]), 2e-3)
                self.assertLess(math.fabs(x2.X[i][j] - x2_ans[i][j]), 2e-3)

    # min   -sum(x1) + sum(x2)
    # s.t.  x1 + x2 <= 6,   x2 >= 1,   x1 <= 0  (nonpos)
    def test_neg_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, 3, nonpos=True)
        x2 = admm.Var("x2", 3, 3)
        model.setObjective(-admm.sum(x1) + admm.sum(x2))
        model.addConstr(x1 + x2 <= 6)
        model.addConstr(x2 >= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 9), 1e-2)
        x1_ans = np.zeros((3, 3))
        x2_ans = np.ones((3, 3))
        for i in range(3):
            for j in range(3):
                self.assertLess(math.fabs(x1.X[i][j] - x1_ans[i][j]), 2e-3)
                self.assertLess(math.fabs(x2.X[i][j] - x2_ans[i][j]), 2e-3)

    # min   tr(C X)
    # s.t.  tr(A[i] X) = b[i],   i = 1, ..., p,   X in S+^n
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
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.setOption(admm.Options.solver_verbosity_level, 1)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   tr(C X)
    # s.t.  tr(A X) = b,   X in S+^2
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
        model.setOption(admm.Options.solver_verbosity_level, 1)
        model.optimize()

        constraint_val = abs(admm.trace(A @ X.X).asScalar() - b)
        self.assertLess(constraint_val, 1e-3)
        eigenvals = np.linalg.eigvals(X.X)
        self.assertTrue(np.all(eigenvals >= -1e-6))
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x1) + sum(x2)
    # s.t.  x1 + x2 <= 6,   x1[0,2] >= 2,   x1 >= 1,   x2 >= 1,   x1 in S+^3
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
        model.optimize()

        self.assertLess(math.fabs(model.objval - 22), 1e-2)
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
        model.optimize()

        self.assertLess(math.fabs(model.objval - 20), 1e-2)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(x1) + sum(x2)
    # s.t.  x1 + x2 >= -6,   x1[0,2] <= -5,   x1 <= -1,   x2 <= -1,   x1 in S-^3
    def test_NSD_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, 3, NSD=True)
        x2 = admm.Var("x2", 3, 3)
        model.setObjective(admm.sum(x1) + admm.sum(x2))
        model.addConstr(x1 + x2 >= -6)
        model.addConstr(x1[0, 2] <= -5)
        model.addConstr(x1 <= -1)
        model.addConstr(x2 <= -1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval + 54), 2e-2)
        self.assertLess(math.fabs(x1.X[0][0] + 5), 2e-2)
        self.assertLess(math.fabs(x1.X[0][2] + 5), 2e-2)
        self.assertLess(math.fabs(x1.X[2][0] + 5), 2e-2)
        self.assertLess(math.fabs(x1.X[2][2] + 5), 2e-2)

    # min   sum(x1) + sum(x2)
    # s.t.  x1 + x2 >= -6,   x1[0,1] <= -5,   x1 <= -1,   x2 <= -1,   x1 in S-^2
    def test_NSD_2(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2, NSD=True)
        x2 = admm.Var("x2", 2, 2)
        model.setObjective(admm.sum(x1) + admm.sum(x2))
        model.addConstr(x1 + x2 >= -6)
        model.addConstr(x1[0, 1] <= -5)
        model.addConstr(x1 <= -1)
        model.addConstr(x2 <= -1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval + 24), 1e-2)
        x1_ans = np.array([[-5.0, -5.0], [-5.0, -5.0]])
        x2_ans = np.array([[-1.0, -1.0], [-1.0, -1.0]])
        for i in range(2):
            for j in range(2):
                self.assertLess(math.fabs(x1.X[i][j] - x1_ans[i][j]), 2e-2)
                self.assertLess(math.fabs(x2.X[i][j] - x2_ans[i][j]), 2e-2)

    # min   f^T x
    # s.t.  F x = g,   ||A x + b||_2 <= c^T x + d
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

        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 2), 1e-3)
        self.assertLess(math.fabs(x.X[0] - 1), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   5 f^T x
    # s.t.  F x = g,   3 ||A x + b||_2 <= 2 c^T x + d
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
        model.optimize()

        self.assertLess(math.fabs(model.objval - 10), 2e-3)
        self.assertLess(math.fabs(x.X[0] - 1), 2e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(diag(x))
    # s.t.  x >= 2
    def test_diag_with_lowbound(self):
        model = admm.Model()
        x = admm.Var("x", 2)
        model.setObjective(admm.sum(admm.diag(x)))
        model.addConstr(x >= 2)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 4), 1e-3)

    # min   sum(diag(x))
    # s.t.  x >= 2,   x in R^1
    def test_diag_with_lowbound_1(self):
        model = admm.Model()
        k = 1
        x = admm.Var("x", k)
        model.setObjective(admm.sum(admm.diag(x)))
        model.addConstr(x >= 2)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 2 * k), 1e-3)

    # min   sum(diag(x))
    # s.t.  x = 2
    def test_diag_with_EQ(self):
        model = admm.Model()
        x = admm.Var("x", 5)
        model.setObjective(admm.sum(admm.diag(x)))
        model.addConstr(x == 2)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 10), 1e-3)

    # min   sum(diag(x)) + tr(X)
    # s.t.  x = 1,   X = diag(x)
    def test_diag_trace(self):
        model = admm.Model()
        x = admm.Var("x", 5)
        X = admm.Var("X", 5, 5)
        model.setObjective(admm.sum(admm.diag(x)) + admm.trace(X))
        model.addConstr(x == 1)
        model.addConstr(X == admm.diag(x))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 10), 1e-3)

    # min   sum(diag(X)) + sum(diag(x))
    # s.t.  X >= 2,   x = diag(X)
    def test_diag_trace_1(self):
        model = admm.Model()
        X = admm.Var("X", 2, 2)
        x = admm.Var("x", 2)
        model.setObjective(admm.sum(admm.diag(X)) + admm.sum(admm.diag(x)))
        model.addConstr(X >= 2)
        model.addConstr(x == admm.diag(X))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 8), 1e-3)

    # min   ||x1||_1
    # s.t.  x1 <= -1,   x1 in R^{3 x 2}
    def test_only_2_3(self):
        model = admm.Model()
        x1 = admm.Var("x1", (3, 2))
        model.setObjective(admm.norm(x1, ord=1))
        model.addConstr(x1 <= -1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 3), 1e-3)

    # min   ||x1||_1
    # s.t.  x1 <= -1,   x1 in R^3
    def test_only_2_3_1(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3)
        model.setObjective(admm.norm(x1, ord=1))
        model.addConstr(x1 <= -1)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 3), 1e-3)

    # min   100 * max_i(-x1)
    # s.t.  x1 <= -2
    def test_example_4(self):
        model = admm.Model()
        x1 = admm.Var("x1", (4, 1))

        model.setObjective(100 * admm.max(-x1))
        model.addConstr(x1 <= -2)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 200), 4e-2)

    # min   0.1 * tv2d(U) + 0.9 * ||conv2d(U, k) - b||_F^2
    #
    # where  tv2d(U) is the 2D total variation of image U, k is a blur kernel
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 400000)
        model.setOption(admm.Options.termination_absolute_error_threshold, 1e-6)
        model.setOption(admm.Options.termination_relative_error_threshold, 1e-6)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   tv2d(U)
    # s.t.  U * mask = image * mask
    #
    # where  tv2d(U) is the 2D total variation, mask is the observed pixel set
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
        model.setOption(admm.Options.solver_verbosity_level, 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 8.984313725), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   47 * huber(50 x1 + 141, 1)
    # s.t.  9 x1 + 40 >= -55
    def test_huber_4(self):
        model = admm.Model()
        x1 = admm.Var("x1", 1)
        p2 = 60
        model.setObjective(admm.sum(47 * admm.huber(50 * x1 + p2 + 81, 1)))
        model.addConstr(9 * x1 + 40 >= -55)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 0), 1e-3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        py_eval = admm.norm(x1.X, ord=2) + admm.sum(admm.maximum(x1.X + x2.X + 1, 1))
        self.assertLess(math.fabs(py_eval.asScalar() - model.objval), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   |x[0]+1| + x[0] + |x[1]-1| + 2 x[1] + 3|x[2]-2| + x[2]^2
    #       + 0.5|x[3]-3| + (x[3]-1)^2/3 + 5|x[4]-1| + 0.2(x[4]+1)^2
    # s.t.  x[0] + 2 x[1] + 3 x[2] = 8,   x[0] - x[3] + x[4] = 4
    #       x[1] - x[2] + 2 x[4] >= 0,   2 x[0] - 3 x[2] + 4 x[3] <= 7
    def test_ex3_vec_1(self):
        model = admm.Model()
        x = admm.Var("x", 5)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(model.objval, 14.2)
        self.assertGreater(model.objval, 14)

    # min   ||A x - b||_2 + ||x||_1
    def test_ex11_norm2(self):
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = np.array([16, 40])
        model = admm.Model()
        x = admm.Var("x", 4)
        model.setObjective(admm.norm(A @ x - b, ord=2) + admm.norm(x, ord=1))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 6) / 6.0, 1e-3)
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 20000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 15) / 15.0, 1e-4)
        self.assertEqual(model.status, 1)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(entropy(x1 + p1 + 2))
    # s.t.  x1 >= 0  (nonneg)
    def test_constr_check(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2, nonneg=True)
        p1 = np.array([[-30, 68], [-66, -44]])
        model.setObjective(admm.sum(admm.entropy(1 * x1 + p1 + 2)))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertLess(abs(model.objval - 296.29) / 296.29, 1e-2)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(5 * delta_{[-86.65, 9.09]}(96 x1 + x3 + 38)) + sum(p1)
    # s.t.  33 x1 + 63 >= -2,   -52 x2 + 2 <= 96,   4 x3 + 59 >= 23
    #       x1 symmetric,   x2 nonpos,   x3 symmetric
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertEqual(model.status, 1)
        self.assertLess(math.fabs(model.objval + 7), 1e-2)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   76 * ||9 x1 + 74||_1
    # s.t.  -2 x1 + 4 >= -63,   x1 >= 0  (nonneg)
    def test_inf_by_price(self):
        model = admm.Model()
        x1 = admm.Var("x1", 3, nonneg=True)
        model.setObjective(76 * admm.norm(9 * x1 + 74, ord=1))
        model.addConstr(-2 * x1 + 4 >= -63)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertEqual(model.status, 1)
        self.assertLess(math.fabs(model.objval - 16872), 1e-1)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum((60 x1 + 56)^2)
    # s.t.  -25 x1 + 46 >= -32,   x1 >= 0  (nonneg)
    def test_convergence_slow(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, nonneg=True)
        model.setObjective(admm.sum(admm.square(60 * x1 + 56)))
        model.addConstr(-25 * x1 + 46 >= -32)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertEqual(model.status, 1)
        self.assertLess(math.fabs(model.objval - 6272) / 6272, 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||9 x2 + 2||_2
    # s.t.  -53 x2 + 62 >= 52
    def test_Norm2(self):
        model = admm.Model()
        x2 = admm.Var("x2", 3, 3)
        model.setObjective(admm.norm(9 * x2 + 2, ord=2))
        model.addConstr(-53 * x2 + 62 >= 52)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertEqual(model.status, 1)
        self.assertLess(math.fabs(model.objval), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   -sqrt(x1)
    # s.t.  x3 = 68
    # Check detection of unboundedness (x1 unconstrained, -sqrt(x1) -> -inf).
    def test_unbound(self):
        model = admm.Model()
        x1 = admm.Var("x1")
        x3 = admm.Var("x3")
        model.setObjective(admm.sum(-admm.sqrt(x1)))
        model.addConstr(x3 == 68)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertEqual(model.status, 4)

    # min   squared_hinge(x)  = (max(1 - x, 0))^2
    def test_squared_hinge(self):
        model = admm.Model()
        x = admm.Var("x")
        model.setObjective(admm.squared_hinge(x))
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.optimize()

        self.assertLess(math.fabs(model.objval), 1e-3)
        self.assertLess(math.fabs(x.X - 1), 1e-3)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   sum(47 * (2 x1 + x2 + 14)^2)
    # s.t.  81 x1 + 23 >= -94,   -32 x2 + 92 >= -83,   x1 symmetric,   x2 nonneg
    def test_consistent(self):
        model = admm.Model()
        x1 = admm.Var("x1", 2, 2, symmetric=True)
        x2 = admm.Var("x2", 2, 2, nonneg=True)
        model.setObjective(admm.sum(47 * admm.square(2 * x1 + x2 + 14)))
        model.addConstr(81 * x1 + 23 >= -94)
        model.addConstr(-32 * x2 + 92 >= -83)
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertEqual(model.status, 1)
        self.assertLess(math.fabs(model.objval - 23211.906564176228), 1e1)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   37 * ||70 x1 + x3 + 50||_* + 40 * sum(-sqrt(6 x3 + 75))
    # s.t.  -44 x1 + 49 <= -97,   27 x3 + 56 >= -15
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
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.admm_max_iteration, 100000)
        model.optimize()

        self.assertLess(math.fabs(model.objval - 34890) / 34890, 1e-3)
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
        model.optimize()

        self.assertLess(math.fabs(x.X[0, 0] - 146.17992381), 1e-1)
        self.assertLess(math.fabs(x.X[1, 0] + 108.1185738), 1e-1)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   ||x||_2
    # s.t.  x >= 1
    def test_norm_pattern_detect_simple(self):
        """Test norm pattern detection with sqrt(x^T x)."""
        model = admm.Model()
        x = admm.Var("x", 2)
        model.setObjective(admm.sqrt(x.T @ x))
        model.addConstr(x >= 1)
        model.optimize()

        self.assertLess(abs(model.ObjVal - math.sqrt(2)), 1e-4)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    def test_warm_start(self):
        # min   sum((x - c)^2)
        # Optimal solution: x* = c, ObjVal* = 0
        # Warm start x.start = c puts the initial point at the optimum.
        n = 4
        c = np.array([1.0, 2.0, 3.0, 4.0])

        model = admm.Model()
        x = admm.Var("x", n)
        x.start = c
        model.setObjective(admm.sum(admm.square(x - c)))
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertLess(abs(model.ObjVal), 1e-6)

        # Shape mismatch must raise ValueError and leave start unset
        model2 = admm.Model()
        y = admm.Var("y", n)
        with self.assertRaises(ValueError):
            y.start = np.zeros(n + 1)
        self.assertIsNone(y.start)

    def test_warm_start_vector_constrained(self):
        # min   sum((x - c)^2)
        # s.t.  sum(x) = 1,  x >= 0   (projection onto probability simplex)
        # Verifies that a vector warm start produces the same optimal objective
        # as a cold start.
        n = 6
        np.random.seed(7)
        c = np.abs(np.random.randn(n)) + 0.5   # positive entries

        # Cold start (reference)
        m_ref = admm.Model()
        x_ref = admm.Var("x", n)
        m_ref.setObjective(admm.sum(admm.square(x_ref - c)))
        m_ref.addConstr(admm.sum(x_ref) == 1)
        m_ref.addConstr(x_ref >= 0)
        m_ref.optimize()
        self.assertEqual(m_ref.StatusString, "SOLVE_OPT_SUCCESS")

        # Warm start: x.start = uniform feasible point (center of simplex)
        model = admm.Model()
        x = admm.Var("x", n)
        x.start = np.ones(n) / n
        model.setObjective(admm.sum(admm.square(x - c)))
        model.addConstr(admm.sum(x) == 1)
        model.addConstr(x >= 0)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertLess(abs(model.ObjVal - m_ref.ObjVal), 1e-4)


if __name__ == "__main__":

    test_suite = unittest.TestSuite()
    test_suite.addTest(ASTProblemTestCase("test_norm_pattern_detect_simple"))
    unittest.TextTestRunner().run(test_suite)
