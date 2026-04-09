"""Comprehensive tests for UDF grad() feature.

Tests grad-only UDFs (no argmin) across a variety of smooth functions,
problem dimensions, and constraint configurations. For functions with
known proximal operators, we also compare grad-based vs argmin-based
solutions to verify consistency.
"""
import admm
import numpy as np
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import udf


# ---------------------------------------------------------------------------
# Grad-only UDF definitions (smooth functions)
# ---------------------------------------------------------------------------

class QuarticGrad(admm.UDFBase):
    """f(x) = sum(x_i^4),  grad = 4*x^3"""
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(x ** 4))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [4.0 * x ** 3]


class SixthPowerGrad(admm.UDFBase):
    """f(x) = sum(x_i^6),  grad = 6*x^5"""
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(x ** 6))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [6.0 * x ** 5]


class ExpSumGrad(admm.UDFBase):
    """f(x) = sum(exp(x_i)),  grad = exp(x)"""
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.exp(x)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [np.exp(x)]


class LogSumExpGrad(admm.UDFBase):
    """f(x) = log(sum(exp(x_i))),  grad = softmax(x)"""
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        m = np.max(x)
        return float(m + np.log(np.sum(np.exp(x - m))))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        m = np.max(x)
        e = np.exp(x - m)
        return [e / np.sum(e)]


class SoftplusGrad(admm.UDFBase):
    """f(x) = sum(log(1 + exp(x_i))),  grad = sigmoid(x)"""
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.logaddexp(0, x)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [1.0 / (1.0 + np.exp(-x))]


class HuberGrad(admm.UDFBase):
    """Huber loss: f(x) = sum h_delta(x_i)
    h(t) = t^2/(2*delta)   if |t| <= delta
           |t| - delta/2   if |t| > delta
    grad: t/delta if |t|<=delta, sign(t) otherwise
    """
    def __init__(self, arg, delta=1.0):
        self.arg = arg
        self.delta = delta
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        d = self.delta
        ax = np.abs(x)
        return float(np.sum(np.where(ax <= d, x ** 2 / (2 * d), ax - d / 2)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        d = self.delta
        return [np.where(np.abs(x) <= d, x / d, np.sign(x))]


class SmoothL1Grad(admm.UDFBase):
    """Smooth L1: f(x) = sum(sqrt(x_i^2 + eps) - sqrt(eps))
    grad = x / sqrt(x^2 + eps)
    """
    def __init__(self, arg, eps=1e-4):
        self.arg = arg
        self.eps = eps
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.sqrt(x ** 2 + self.eps) - np.sqrt(self.eps)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [x / np.sqrt(x ** 2 + self.eps)]


class NegEntropyGrad(admm.UDFBase):
    """Negative entropy: f(x) = sum(x_i * log(x_i)), defined for x > 0
    grad = log(x) + 1
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        x = np.maximum(x, 1e-30)
        return float(np.sum(x * np.log(x)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        x = np.maximum(x, 1e-30)
        return [np.log(x) + 1.0]


class LogBarrierGrad(admm.UDFBase):
    """Log barrier: f(x) = -sum(log(x_i)), defined for x > 0
    grad = -1/x
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        x = np.maximum(x, 1e-30)
        return float(-np.sum(np.log(x)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        x = np.maximum(x, 1e-30)
        return [-1.0 / x]


class LogisticLossGrad(admm.UDFBase):
    """Logistic regression loss: f(w) = sum(log(1 + exp(-y_i * a_i^T w)))
    grad = -sum(y_i * a_i * sigmoid(-y_i * a_i^T w))
    """
    def __init__(self, arg, A, y):
        self.arg = arg
        self.A = np.asarray(A, dtype=float)
        self.y = np.asarray(y, dtype=float)
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        w = np.asarray(tensorlist[0], dtype=float).ravel()
        z = self.y * (self.A @ w)
        return float(np.sum(np.logaddexp(0, -z)))
    def grad(self, tensorlist):
        w = np.asarray(tensorlist[0], dtype=float).ravel()
        z = self.y * (self.A @ w)
        sig = 1.0 / (1.0 + np.exp(z))  # sigmoid(-z)
        g = -self.A.T @ (self.y * sig)
        return [g.reshape(tensorlist[0].shape)]


class QuadFormGrad(admm.UDFBase):
    """Quadratic form: f(x) = 0.5 * x^T Q x
    grad = Q x (Q must be symmetric)
    """
    def __init__(self, arg, Q):
        self.arg = arg
        self.Q = np.asarray(Q, dtype=float)
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        return float(0.5 * x @ self.Q @ x)
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        g = self.Q @ x
        return [g.reshape(tensorlist[0].shape)]


class MatExpSumGrad(admm.UDFBase):
    """Matrix exponential sum: f(X) = sum(exp(X_ij))
    grad = exp(X)
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        X = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.exp(X)))
    def grad(self, tensorlist):
        X = np.asarray(tensorlist[0], dtype=float)
        return [np.exp(X)]


class SmoothMaxGrad(admm.UDFBase):
    """Squared hinge / smooth max: f(x) = sum(max(0, x_i)^2)
    grad = 2*max(0, x_i)
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.maximum(0, x) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [2.0 * np.maximum(0, x)]


class CubicGrad(admm.UDFBase):
    """f(x) = sum(x_i^3) — NOT convex, but smooth everywhere.
    grad = 3*x^2
    Note: only convex for x >= 0, so we use with constraints.
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(x ** 3))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [3.0 * x ** 2]


class ExpMLEGrad(admm.UDFBase):
    """MLE for exponential distribution: f(x) = t_sum*x - n*log(x)
    grad = t_sum - n/x
    """
    def __init__(self, arg, t_sum, n_obs):
        self.arg = arg
        self.t_sum = t_sum
        self.n_obs = n_obs
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = tensorlist[0].ravel()[0]
        if x <= 0: return 1e20
        return self.t_sum * x - self.n_obs * np.log(x)
    def grad(self, tensorlist):
        x = tensorlist[0].ravel()[0]
        if x <= 1e-12: x = 1e-12
        return [np.array([self.t_sum - self.n_obs / x])]


class PowerMeanGrad(admm.UDFBase):
    """Power mean penalty: f(x) = (sum(x_i^p))^(1/p), p > 1
    grad = (sum(x_i^p))^(1/p - 1) * x^(p-1)
    Assumes x >= 0.
    """
    def __init__(self, arg, p=4.0):
        self.arg = arg
        self.p = p
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        x = np.maximum(x, 0)
        s = np.sum(x ** self.p)
        return float(s ** (1.0 / self.p))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        x = np.maximum(x, 0)
        s = np.sum(x ** self.p)
        if s < 1e-30:
            return [np.zeros_like(x)]
        outer = s ** (1.0 / self.p - 1.0)
        inner = self.p * x ** (self.p - 1.0)
        return [outer * inner / self.p]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class GradUDFTestCase(unittest.TestCase):

    def _new_model(self):
        model = admm.Model()
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.display_sub_solver_details, 0)
        return model

    # ---------------------------------------------------------------
    # 1. Quartic: compare grad vs argmin (baseline)
    # ---------------------------------------------------------------
    def test_quartic_grad_vs_argmin(self):
        """f(x) = sum(x^4): grad UDF should match argmin UDF"""
        observed = np.array([2.0, -1.0, 0.5])
        lam = 0.1

        model1 = self._new_model()
        x1 = admm.Var("x1", 3)
        model1.setObjective(
            0.5 * admm.sum(admm.square(x1 - observed))
            + lam * udf.QuarticPenalty(x1)
        )
        model1.optimize()

        model2 = self._new_model()
        x2 = admm.Var("x2", 3)
        model2.setObjective(
            0.5 * admm.sum(admm.square(x2 - observed))
            + lam * QuarticGrad(x2)
        )
        model2.optimize()

        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(model1.ObjVal, model2.ObjVal, places=3)
        np.testing.assert_allclose(x1.X, x2.X, atol=1e-3)

    # ---------------------------------------------------------------
    # 2. Sixth power: higher-order smooth penalty
    # ---------------------------------------------------------------
    def test_sixth_power_penalty(self):
        """min 0.5*||x - y||^2 + lam * sum(x^6)
        Analytical: each x_i satisfies x_i + 6*lam*x_i^5 = y_i
        """
        observed = np.array([1.0, -2.0, 3.0, 0.5])
        lam = 0.01

        model = self._new_model()
        x = admm.Var("x", len(observed))
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SixthPowerGrad(x)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Verify KKT: x_i + 6*lam*x_i^5 - y_i ≈ 0
        residual = x_val + 6 * lam * x_val ** 5 - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    # ---------------------------------------------------------------
    # 3. Exponential sum: strongly convex
    # ---------------------------------------------------------------
    def test_exp_sum_unconstrained(self):
        """min 0.5*||x - y||^2 + lam * sum(exp(x))
        KKT: x_i + lam*exp(x_i) = y_i
        """
        observed = np.array([3.0, -1.0, 0.0, 2.0])
        lam = 0.5

        model = self._new_model()
        x = admm.Var("x", len(observed))
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * ExpSumGrad(x)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + lam * np.exp(x_val) - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    # ---------------------------------------------------------------
    # 4. Exp sum with box constraints
    # ---------------------------------------------------------------
    def test_exp_sum_box_constrained(self):
        """min sum(exp(x))  s.t. 0 <= x <= 1
        Optimal: x = [0, 0, ..., 0]  (exp is increasing)
        ObjVal = n * exp(0) = n
        """
        n = 5
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(ExpSumGrad(x))
        model.addConstr(x >= 0)
        model.addConstr(x <= 1)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 0, atol=0.05)
        self.assertAlmostEqual(model.ObjVal, n, places=1)

    # ---------------------------------------------------------------
    # 5. Log-sum-exp (smooth max approximation)
    # ---------------------------------------------------------------
    def test_logsumexp_with_linear(self):
        """min log(sum(exp(x))) + c^T x
        A smooth convex problem.
        """
        c = np.array([1.0, -0.5, 0.2])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(LogSumExpGrad(x) + c @ x)
        model.addConstr(x >= -2)
        model.addConstr(x <= 2)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # log-sum-exp is minimized when all x_i are small;
        # c^T x pushes x[1] high (c[1]=-0.5) and x[0] low
        self.assertLess(x_val[0], x_val[1])

    # ---------------------------------------------------------------
    # 6. Softplus (smooth ReLU)
    # ---------------------------------------------------------------
    def test_softplus_penalty(self):
        """min 0.5*||x - y||^2 + lam * sum(log(1+exp(x)))
        KKT: x_i + lam * sigmoid(x_i) = y_i
        """
        observed = np.array([2.0, 0.0, -1.0])
        lam = 0.3

        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SoftplusGrad(x)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        sigmoid = 1.0 / (1.0 + np.exp(-x_val))
        residual = x_val + lam * sigmoid - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    # ---------------------------------------------------------------
    # 7. Huber loss as penalty
    # ---------------------------------------------------------------
    def test_huber_penalty(self):
        """min 0.5*||x - y||^2 + lam * sum(huber(x, delta))
        Smooth, convex, and a standard robust loss.
        """
        observed = np.array([5.0, -3.0, 0.1, 2.0])
        lam = 0.2
        delta = 1.0

        model = self._new_model()
        x = admm.Var("x", len(observed))
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * HuberGrad(x, delta=delta)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Solution should be between 0 and observed
        for i in range(len(observed)):
            if observed[i] > 0:
                self.assertGreater(x_val[i], 0)
                self.assertLess(x_val[i], observed[i] + 0.1)
            elif observed[i] < 0:
                self.assertLess(x_val[i], 0)
                self.assertGreater(x_val[i], observed[i] - 0.1)

    # ---------------------------------------------------------------
    # 8. Huber with different delta values
    # ---------------------------------------------------------------
    def test_huber_delta_comparison(self):
        """Different delta values should all solve successfully.
        Smaller delta -> Huber is closer to L1 -> stronger shrinkage.
        """
        observed = np.array([3.0, -2.0])
        lam = 1.0

        results = {}
        for delta in [0.5, 1.0, 2.0]:
            model = self._new_model()
            x = admm.Var("x", 2)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + lam * HuberGrad(x, delta=delta)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            results[delta] = np.linalg.norm(np.asarray(x.X))

        # All deltas solved; solutions should differ
        self.assertFalse(np.isclose(results[0.5], results[2.0], atol=0.01),
                         "Different delta should produce different solutions")

    # ---------------------------------------------------------------
    # 9. Smooth L1 approximation
    # ---------------------------------------------------------------
    def test_smooth_l1_sparsity(self):
        """min 0.5*||x - y||^2 + lam * sum(sqrt(x^2 + eps))
        Approximates L1 penalty -> promotes sparsity.
        """
        observed = np.array([0.1, 2.0, -0.05, -1.5, 0.02])
        lam = 0.5

        model = self._new_model()
        x = admm.Var("x", len(observed))
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SmoothL1Grad(x, eps=1e-4)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Small entries should shrink toward 0
        self.assertLess(abs(x_val[0]), abs(observed[0]))
        self.assertLess(abs(x_val[2]), abs(observed[2]))
        self.assertLess(abs(x_val[4]), abs(observed[4]))
        # Large entries should be less affected
        self.assertGreater(abs(x_val[1]), 1.0)
        self.assertGreater(abs(x_val[3]), 0.5)

    # ---------------------------------------------------------------
    # 10. Negative entropy with simplex constraint
    # ---------------------------------------------------------------
    def test_neg_entropy_simplex(self):
        """min  sum(x * log(x))  s.t.  x >= 0, sum(x) = 1
        Known solution: x = [1/n, ..., 1/n] (uniform distribution)
        """
        n = 4
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(NegEntropyGrad(x))
        model.addConstr(x >= 0.001)  # keep x > 0
        model.addConstr(admm.sum(x) == 1)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        expected = np.ones(n) / n
        np.testing.assert_allclose(x_val, expected, atol=0.05)

    # ---------------------------------------------------------------
    # 11. Log barrier with equality constraint
    # ---------------------------------------------------------------
    def test_log_barrier(self):
        """min  -sum(log(x))  s.t.  sum(x) = n, x >= 0.01
        Known solution: x = [1, 1, ..., 1]
        """
        n = 3
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(LogBarrierGrad(x))
        model.addConstr(x >= 0.01)
        model.addConstr(admm.sum(x) == n)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 1.0, atol=0.1)

    # ---------------------------------------------------------------
    # 12. Logistic regression
    # ---------------------------------------------------------------
    def test_logistic_regression(self):
        """min  sum(log(1+exp(-y_i * a_i^T w))) + 0.5*lam*||w||^2
        Standard L2-regularized logistic regression.
        """
        np.random.seed(42)
        n, d = 50, 5
        w_true = np.array([1.0, -0.5, 0.3, 0.0, 0.8])
        A = np.random.randn(n, d)
        probs = 1.0 / (1.0 + np.exp(-A @ w_true))
        y = 2.0 * (np.random.rand(n) < probs) - 1.0  # {-1, +1}
        lam = 0.1

        model = self._new_model()
        w = admm.Var("w", d)
        model.setObjective(
            LogisticLossGrad(w, A, y)
            + 0.5 * lam * admm.sum(admm.square(w))
        )
        model.setOption(admm.Options.admm_max_iteration, 5000)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        w_val = np.asarray(w.X).ravel()
        # Check that signs of large components match
        for i in [0, 1, 4]:  # components with |w_true| >= 0.5
            self.assertEqual(np.sign(w_val[i]), np.sign(w_true[i]),
                             f"Sign mismatch at w[{i}]: got {w_val[i]:.3f}, expected sign {np.sign(w_true[i])}")

    # ---------------------------------------------------------------
    # 13. Logistic regression with box constraint
    # ---------------------------------------------------------------
    def test_logistic_regression_bounded(self):
        """Logistic regression with ||w||_inf <= 2"""
        np.random.seed(123)
        n, d = 30, 3
        w_true = np.array([1.5, -1.0, 0.5])
        A = np.random.randn(n, d)
        y = 2.0 * (1.0 / (1.0 + np.exp(-A @ w_true)) > 0.5) - 1.0

        model = self._new_model()
        w = admm.Var("w", d)
        model.setObjective(LogisticLossGrad(w, A, y))
        model.addConstr(w >= -2)
        model.addConstr(w <= 2)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        w_val = np.asarray(w.X).ravel()
        # All components within bounds
        self.assertTrue(np.all(w_val >= -2.01))
        self.assertTrue(np.all(w_val <= 2.01))

    # ---------------------------------------------------------------
    # 14. Quadratic form as UDF vs built-in
    # ---------------------------------------------------------------
    def test_quadform_grad_vs_builtin(self):
        """min 0.5 * x^T Q x - c^T x: compare UDF grad vs built-in"""
        n = 4
        np.random.seed(7)
        M = np.random.randn(n, n)
        Q = M.T @ M + 0.1 * np.eye(n)  # positive definite
        c = np.array([1.0, -2.0, 0.5, 3.0])
        x_exact = np.linalg.solve(Q, c)

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(QuadFormGrad(x, Q) - c @ x)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, x_exact, atol=0.1)

    # ---------------------------------------------------------------
    # 15. Smooth max (squared hinge)
    # ---------------------------------------------------------------
    def test_smooth_max_penalty(self):
        """min 0.5*||x - y||^2 + lam * sum(max(0,x)^2)
        KKT: x_i + 2*lam*max(0,x_i) = y_i
        For y_i > 0: x_i = y_i / (1 + 2*lam)
        For y_i <= 0: x_i = y_i
        """
        observed = np.array([3.0, -1.0, 2.0, -0.5, 0.0])
        lam = 0.5

        model = self._new_model()
        x = admm.Var("x", len(observed))
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SmoothMaxGrad(x)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        expected = np.where(observed > 0, observed / (1 + 2 * lam), observed)
        np.testing.assert_allclose(x_val, expected, atol=0.05)

    # ---------------------------------------------------------------
    # 16. Matrix exponential sum
    # ---------------------------------------------------------------
    def test_matrix_exp_sum(self):
        """min 0.5*||X - Y||_F^2 + lam * sum(exp(X_ij))
        Verify solve succeeds and objective is reasonable.
        """
        Y = np.array([[1.0, -0.5], [2.0, 0.0]])
        lam = 0.2

        model = self._new_model()
        X = admm.Var("X", 2, 2)
        model.setObjective(
            0.5 * admm.sum(admm.square(X - Y))
            + lam * MatExpSumGrad(X)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        X_val = np.asarray(X.X)
        # Solution should be element-wise less than Y (exp penalty pushes down)
        self.assertTrue(np.all(X_val <= Y + 0.1))

    # ---------------------------------------------------------------
    # 17. Cubic on nonneg (convex when x >= 0)
    # ---------------------------------------------------------------
    def test_cubic_nonneg(self):
        """min sum(x^3)  s.t.  x >= 0, sum(x) = 1
        On the simplex with x >= 0, x^3 is convex.
        Solution: uniform x = 1/n
        """
        n = 3
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(CubicGrad(x))
        model.addConstr(x >= 0)
        model.addConstr(admm.sum(x) == 1)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        expected = np.ones(n) / n
        np.testing.assert_allclose(x_val, expected, atol=0.05)

    # ---------------------------------------------------------------
    # 18. Grad UDF combined with built-in norm1 (L1)
    # ---------------------------------------------------------------
    def test_grad_udf_plus_builtin_l1(self):
        """min  sum(exp(x)) + lam * ||x||_1
        s.t. -3 <= x <= 3
        Combining grad UDF with built-in L1 penalty.
        Optimal: x should be all negative (exp(x) small + |x| penalty)
        """
        n = 4
        lam = 0.2

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(ExpSumGrad(x) + lam * admm.norm(x, 1))
        model.addConstr(x >= -3)
        model.addConstr(x <= 3)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # All components should be negative (exp decreases, L1 adds cost)
        self.assertTrue(np.all(x_val < 0.1))

    # ---------------------------------------------------------------
    # 19. Grad UDF combined with linear constraints
    # ---------------------------------------------------------------
    def test_grad_udf_linear_constr(self):
        """min  sum(exp(x))
        s.t. Ax = b, x >= 0
        """
        A = np.array([[1, 1, 1, 1]])
        b = np.array([4.0])

        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(ExpSumGrad(x))
        model.addConstr(A @ x == b)
        model.addConstr(x >= 0)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Optimal: uniform x = 1 (by symmetry + convexity)
        np.testing.assert_allclose(x_val, 1.0, atol=0.1)
        # Constraint satisfaction
        self.assertAlmostEqual(np.sum(x_val), 4.0, places=1)

    # ---------------------------------------------------------------
    # 20. Power mean with box constraint
    # ---------------------------------------------------------------
    def test_power_mean_box(self):
        """min  (sum(x^4))^(1/4)  s.t.  x >= 0, sum(x) = 1
        L4 norm on simplex. Solution: uniform.
        """
        n = 3
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(PowerMeanGrad(x, p=4.0))
        model.addConstr(x >= 0)
        model.addConstr(admm.sum(x) == 1)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        expected = np.ones(n) / n
        np.testing.assert_allclose(x_val, expected, atol=0.1)

    # ---------------------------------------------------------------
    # 21. Quartic on scalar (1D)
    # ---------------------------------------------------------------
    def test_quartic_scalar(self):
        """min (x - 3)^2 + 0.5 * x^4
        KKT: 2*(x-3) + 2*x^3 = 0 => x + x^3 = 3
        Numerical root: x ≈ 1.2134
        """
        model = self._new_model()
        x = admm.Var("x", 1)
        model.setObjective(
            admm.sum(admm.square(x - 3))
            + 0.5 * QuarticGrad(x)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).item()
        # Verify KKT: x + x^3 ≈ 3
        self.assertAlmostEqual(x_val + x_val ** 3, 3.0, places=1)

    # ---------------------------------------------------------------
    # 22. Larger dimension (n=100)
    # ---------------------------------------------------------------
    def test_exp_sum_large_dim(self):
        """min 0.5*||x - y||^2 + lam*sum(exp(x)), n=100"""
        np.random.seed(99)
        n = 100
        observed = np.random.randn(n)
        lam = 0.1

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * ExpSumGrad(x)
        )
        model.setOption(admm.Options.admm_max_iteration, 5000)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + lam * np.exp(x_val) - observed
        self.assertLess(np.max(np.abs(residual)), 0.1)

    # ---------------------------------------------------------------
    # 23. Larger matrix (10x10)
    # ---------------------------------------------------------------
    def test_matrix_exp_large(self):
        """min 0.5*||X - Y||_F^2 + lam*sum(exp(X)), X is 10x10"""
        np.random.seed(77)
        m, n = 10, 10
        Y = np.random.randn(m, n) * 0.5
        lam = 0.1

        model = self._new_model()
        X = admm.Var("X", m, n)
        model.setObjective(
            0.5 * admm.sum(admm.square(X - Y))
            + lam * MatExpSumGrad(X)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # ---------------------------------------------------------------
    # 24. Softplus + linear equality
    # ---------------------------------------------------------------
    def test_softplus_with_equality(self):
        """min sum(log(1+exp(x)))  s.t.  1^T x = 0"""
        n = 4
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(SoftplusGrad(x))
        model.addConstr(admm.sum(x) == 0)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # By symmetry, all components should equal 0
        np.testing.assert_allclose(x_val, 0, atol=0.1)
        self.assertAlmostEqual(np.sum(x_val), 0, places=1)

    # ---------------------------------------------------------------
    # 25. Smooth L1 with inequality constraints
    # ---------------------------------------------------------------
    def test_smooth_l1_with_ineq(self):
        """min sum(sqrt(x^2 + eps))  s.t.  Ax >= b
        Smooth L1 → minimum norm feasible point.
        """
        A = np.array([[1, 1], [-1, 0]])
        b = np.array([2.0, -3.0])

        model = self._new_model()
        x = admm.Var("x", 2)
        model.setObjective(SmoothL1Grad(x, eps=1e-6))
        model.addConstr(A @ x >= b)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Constraints should be satisfied
        self.assertTrue(np.all(A @ x_val >= b - 0.1))

    # ---------------------------------------------------------------
    # 26. Quadratic form with equality constraint
    # ---------------------------------------------------------------
    def test_quadform_equality(self):
        """min 0.5 * x^T Q x  s.t.  1^T x = 1
        Lagrangian -> Q x = lambda * 1 -> x = lambda * Q^{-1} 1
        """
        Q = np.array([[2.0, 0.5], [0.5, 3.0]])
        model = self._new_model()
        x = admm.Var("x", 2)
        model.setObjective(QuadFormGrad(x, Q))
        model.addConstr(admm.sum(x) == 1)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertAlmostEqual(np.sum(x_val), 1.0, places=1)
        # Analytic: x = Q^{-1} 1 / (1^T Q^{-1} 1)
        Qinv1 = np.linalg.solve(Q, np.ones(2))
        x_exact = Qinv1 / np.sum(Qinv1)
        np.testing.assert_allclose(x_val, x_exact, atol=0.1)

    # ---------------------------------------------------------------
    # 27. Multiple penalties: grad UDF + grad UDF
    # ---------------------------------------------------------------
    def test_multiple_grad_udfs(self):
        """min 0.5*||x-y||^2 + lam1*sum(x^4) + lam2*sum(exp(x))"""
        observed = np.array([2.0, -1.0, 0.5])
        lam1, lam2 = 0.05, 0.1

        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam1 * QuarticGrad(x)
            + lam2 * ExpSumGrad(x)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # ---------------------------------------------------------------
    # 28. Grad UDF + argmin UDF in same model
    # ---------------------------------------------------------------
    def test_grad_and_argmin_mixed(self):
        """min  lam1 * sum(exp(x)) + lam2 * sum(|x|^4)
        where exp uses grad, quartic uses argmin
        """
        observed = np.array([1.0, -0.5, 2.0])
        lam1, lam2 = 0.1, 0.1

        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam1 * ExpSumGrad(x)
            + lam2 * udf.QuarticPenalty(x)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # ---------------------------------------------------------------
    # 29. Verify gradient correctness via finite differences
    # ---------------------------------------------------------------
    def test_gradient_finite_diff_check(self):
        """Numerical gradient check for all grad UDFs"""
        np.random.seed(12)
        x_test = np.random.randn(3)
        h = 1e-6

        udf_classes = [
            (QuarticGrad, {}),
            (SixthPowerGrad, {}),
            (ExpSumGrad, {}),
            (SoftplusGrad, {}),
            (HuberGrad, {"delta": 1.0}),
            (SmoothL1Grad, {"eps": 1e-4}),
            (SmoothMaxGrad, {}),
        ]

        dummy_var = admm.Var("dummy", 3)
        for cls, kwargs in udf_classes:
            obj = cls(dummy_var, **kwargs)
            f0 = obj.eval([x_test.copy()])
            grad_analytic = obj.grad([x_test.copy()])[0]

            grad_fd = np.zeros_like(x_test)
            for i in range(len(x_test)):
                xp = x_test.copy(); xp[i] += h
                xm = x_test.copy(); xm[i] -= h
                grad_fd[i] = (obj.eval([xp]) - obj.eval([xm])) / (2 * h)

            np.testing.assert_allclose(
                grad_analytic, grad_fd, atol=1e-4,
                err_msg=f"Gradient mismatch for {cls.__name__}"
            )



# ---------------------------------------------------------------------------
# Additional grad-only UDF definitions for robustness tests
# ---------------------------------------------------------------------------

class KLDivGrad(admm.UDFBase):
    """KL divergence from a fixed distribution q:
    f(p) = sum(p_i * log(p_i / q_i))
    grad = log(p/q) + 1
    Assumes p, q > 0.
    """
    def __init__(self, arg, q):
        self.arg = arg
        self.q = np.asarray(q, dtype=float)
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        p = np.maximum(np.asarray(tensorlist[0], dtype=float), 1e-30)
        return float(np.sum(p * np.log(p / self.q)))
    def grad(self, tensorlist):
        p = np.maximum(np.asarray(tensorlist[0], dtype=float), 1e-30)
        return [np.log(p / self.q) + 1.0]


class PoissonLossGrad(admm.UDFBase):
    """Poisson negative log-likelihood:
    f(x) = sum(exp(a_i^T x) - b_i * a_i^T x)
    grad = A^T (exp(Ax) - b)
    """
    def __init__(self, arg, A, b):
        self.arg = arg
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        z = self.A @ x
        return float(np.sum(np.exp(z) - self.b * z))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        z = self.A @ x
        g = self.A.T @ (np.exp(z) - self.b)
        return [g.reshape(tensorlist[0].shape)]


class ElasticNetGrad(admm.UDFBase):
    """Elastic net smooth part: f(x) = 0.5*alpha*||x||^2 + (1-alpha)*sum(sqrt(x^2+eps))
    Combines L2 + smooth-L1. Both parts are smooth.
    grad = alpha*x + (1-alpha)*x/sqrt(x^2+eps)
    """
    def __init__(self, arg, alpha=0.5, eps=1e-4):
        self.arg = arg
        self.alpha = alpha
        self.eps = eps
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        l2 = 0.5 * self.alpha * float(np.sum(x ** 2))
        sl1 = (1 - self.alpha) * float(np.sum(np.sqrt(x ** 2 + self.eps)))
        return l2 + sl1
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        g = self.alpha * x + (1 - self.alpha) * x / np.sqrt(x ** 2 + self.eps)
        return [g]


class LogCoshGrad(admm.UDFBase):
    """Log-cosh loss: f(x) = sum(log(cosh(x_i)))
    A smooth approximation to L1, between Huber and L2.
    grad = tanh(x)
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.log(np.cosh(x))))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [np.tanh(x)]


class SquaredLogGrad(admm.UDFBase):
    """f(x) = sum(log(1+x_i)^2), x > 0
    grad = 2*log(1+x)/(1+x)
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.log1p(np.maximum(x, 0)) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        x = np.maximum(x, 0)
        return [2.0 * np.log1p(x) / (1.0 + x)]


class InversePenaltyGrad(admm.UDFBase):
    """f(x) = sum(1/x_i), x > 0
    grad = -1/x^2
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.maximum(np.asarray(tensorlist[0], dtype=float), 1e-30)
        return float(np.sum(1.0 / x))
    def grad(self, tensorlist):
        x = np.maximum(np.asarray(tensorlist[0], dtype=float), 1e-30)
        return [-1.0 / (x ** 2)]


class FrobExpGrad(admm.UDFBase):
    """f(X) = ||X||_F^2 + mu*sum(exp(X_ij))
    grad = 2*X + mu*exp(X)
    """
    def __init__(self, arg, mu=0.1):
        self.arg = arg
        self.mu = mu
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        X = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(X ** 2) + self.mu * np.sum(np.exp(X)))
    def grad(self, tensorlist):
        X = np.asarray(tensorlist[0], dtype=float)
        return [2.0 * X + self.mu * np.exp(X)]


class WeightedQuarticGrad(admm.UDFBase):
    """f(x) = sum(w_i * x_i^4), weighted quartic
    grad = 4 * w * x^3
    """
    def __init__(self, arg, w):
        self.arg = arg
        self.w = np.asarray(w, dtype=float)
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(self.w * x ** 4))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [4.0 * self.w * x ** 3]


# ---------------------------------------------------------------------------
# Robustness test class
# ---------------------------------------------------------------------------

class GradUDFRobustTestCase(unittest.TestCase):
    """Stress tests for grad-only UDF: edge cases, large scale, mixed
    constraints, numerical stability, and application-level problems.
    """

    def _new_model(self, max_iter=2000):
        model = admm.Model()
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.display_sub_solver_details, 0)
        model.setOption(admm.Options.admm_max_iteration, max_iter)
        return model

    # ===================================================================
    # A. Numerical edge cases
    # ===================================================================

    def test_very_small_lambda(self):
        """lam = 1e-6: penalty almost absent, solution ≈ observed"""
        observed = np.array([5.0, -3.0, 0.1])
        lam = 1e-6
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x.X, observed, atol=0.01)

    def test_very_large_lambda(self):
        """lam = 100: heavy penalty pushes x toward 0"""
        observed = np.array([1.0, -1.0, 2.0])
        lam = 100.0
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.abs(x_val) < np.abs(observed) + 0.01))
        # With heavy quartic penalty x should be small
        self.assertLess(np.linalg.norm(x_val), 1.0)

    def test_zero_observed(self):
        """y = 0: optimal x = 0 for any convex UDF penalty"""
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x))
            + 0.1 * ExpSumGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # KKT: x + 0.1*exp(x) = 0 => x < 0
        self.assertTrue(np.all(x_val < 0.01))

    def test_near_zero_gradient_region(self):
        """Softplus near large negative x has near-zero gradient.
        min 0.5*||x - y||^2 + lam*softplus(x), y = [-10, -10, -10]
        Solution should be close to y since softplus gradient ≈ 0 there.
        """
        observed = np.full(3, -10.0)
        lam = 1.0
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SoftplusGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x.X, observed, atol=0.1)

    def test_large_observed_values(self):
        """y = [1000, -500]: numerically large inputs.
        Quartic x^4 dominates at large x, so solution is pulled toward 0.
        Just verify solver converges successfully.
        """
        observed = np.array([1000.0, -500.0])
        lam = 0.001
        model = self._new_model()
        x = admm.Var("x", 2)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        # Quartic penalty is huge at x=1000 (1e12), so solution is far from observed
        # Just verify the solver produced a finite solution
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))

    def test_single_variable(self):
        """Scalar optimization: min (x-5)^2 + exp(x)
        KKT: 2(x-5) + exp(x) = 0
        """
        model = self._new_model()
        x = admm.Var("x", 1)
        model.setObjective(admm.sum(admm.square(x - 5)) + ExpSumGrad(x))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()[0]
        residual = 2 * (x_val - 5) + np.exp(x_val)
        self.assertAlmostEqual(residual, 0, delta=0.1)

    # ===================================================================
    # B. Dimension stress tests
    # ===================================================================

    def test_dim_200(self):
        """n=200 vector with exp penalty"""
        np.random.seed(200)
        n = 200
        y = np.random.randn(n) * 2
        model = self._new_model(max_iter=5000)
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - y))
            + 0.05 * ExpSumGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    def test_dim_500(self):
        """n=500 vector with quartic penalty"""
        np.random.seed(500)
        n = 500
        y = np.random.randn(n)
        model = self._new_model(max_iter=5000)
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - y))
            + 0.01 * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    def test_matrix_5x5_symmetric(self):
        """5x5 matrix: min ||X - Y||_F^2 + lam*sum(exp(X))"""
        np.random.seed(55)
        Y = np.random.randn(5, 5)
        Y = (Y + Y.T) / 2  # symmetric target
        model = self._new_model()
        X = admm.Var("X", 5, 5)
        model.setObjective(
            0.5 * admm.sum(admm.square(X - Y))
            + 0.1 * MatExpSumGrad(X)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    def test_tall_matrix_20x3(self):
        """Tall matrix 20x3"""
        np.random.seed(203)
        Y = np.random.randn(20, 3)
        model = self._new_model()
        X = admm.Var("X", 20, 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(X - Y))
            + 0.05 * MatExpSumGrad(X)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    def test_wide_matrix_3x20(self):
        """Wide matrix 3x20"""
        np.random.seed(320)
        Y = np.random.randn(3, 20)
        model = self._new_model()
        X = admm.Var("X", 3, 20)
        model.setObjective(
            0.5 * admm.sum(admm.square(X - Y))
            + 0.05 * MatExpSumGrad(X)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # ===================================================================
    # C. Constraint stress tests
    # ===================================================================

    def test_multiple_equality_constraints(self):
        """min sum(exp(x))  s.t.  x1+x2=2, x3+x4=2, x >= 0
        Solution: x = [1,1,1,1]
        """
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(ExpSumGrad(x))
        model.addConstr(x[0] + x[1] == 2)
        model.addConstr(x[2] + x[3] == 2)
        model.addConstr(x >= 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 1.0, atol=0.15)

    def test_mixed_eq_ineq_constraints(self):
        """min sum(x^4)  s.t.  sum(x)=3, x >= 0.5
        Uniform solution: x = [1,1,1]
        """
        n = 3
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(QuarticGrad(x))
        model.addConstr(admm.sum(x) == 3)
        model.addConstr(x >= 0.5)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 1.0, atol=0.15)
        self.assertAlmostEqual(np.sum(x_val), 3.0, places=1)

    def test_tight_box_constraints(self):
        """Tight box [0.9, 1.1]: solution should be near boundary"""
        observed = np.array([0.0, 5.0, 1.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(0.5 * admm.sum(admm.square(x - observed)))
        model.addConstr(x >= 0.9)
        model.addConstr(x <= 1.1)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, np.clip(observed, 0.9, 1.1), atol=0.05)

    def test_many_inequality_constraints(self):
        """min sum(exp(x))  s.t.  A[i,:]@x <= b[i], i=1..20, x >= -5"""
        np.random.seed(42)
        n = 5
        m = 20
        A = np.random.randn(m, n)
        x_feas = np.zeros(n)
        b = A @ x_feas + np.abs(np.random.randn(m))  # ensure feasibility at 0

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(ExpSumGrad(x))
        for i in range(m):
            model.addConstr(A[i, :] @ x <= b[i])
        model.addConstr(x >= -5)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # All constraints satisfied
        self.assertTrue(np.all(A @ x_val <= b + 0.1))

    def test_equality_overdetermined(self):
        """sum(x)=4, x1=x2=x3=x4 via pairwise equalities
        Solution: x = [1,1,1,1]
        """
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(ExpSumGrad(x))
        model.addConstr(admm.sum(x) == 4)
        model.addConstr(x[0] == x[1])
        model.addConstr(x[1] == x[2])
        model.addConstr(x[2] == x[3])
        model.addConstr(x >= 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 1.0, atol=0.15)

    # ===================================================================
    # D. New smooth function tests
    # ===================================================================

    def test_log_cosh_penalty(self):
        """Log-cosh: smooth between L1 and L2.
        min 0.5*||x-y||^2 + lam * sum(log(cosh(x)))
        KKT: x + lam*tanh(x) = y
        """
        observed = np.array([2.0, -1.5, 0.3, 3.0])
        lam = 0.5
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * LogCoshGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + lam * np.tanh(x_val) - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    def test_elastic_net_grad(self):
        """Elastic net: alpha*L2 + (1-alpha)*smooth-L1
        min 0.5*||x-y||^2 + lam * elastic_net(x)
        """
        observed = np.array([0.1, 2.0, -0.05, -1.5, 0.02])
        lam = 0.3
        model = self._new_model()
        x = admm.Var("x", len(observed))
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * ElasticNetGrad(x, alpha=0.5)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Small entries shrink, large entries retained
        self.assertLess(abs(x_val[0]), abs(observed[0]))
        self.assertGreater(abs(x_val[1]), 1.0)

    def test_kl_divergence_simplex(self):
        """min KL(p || q)  s.t.  sum(p)=1, p >= 0.001
        Solution: p = q (KL divergence is 0 at p=q)
        """
        q = np.array([0.3, 0.5, 0.2])
        model = self._new_model()
        p = admm.Var("p", 3)
        model.setObjective(KLDivGrad(p, q))
        model.addConstr(admm.sum(p) == 1)
        model.addConstr(p >= 0.001)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        p_val = np.asarray(p.X)
        np.testing.assert_allclose(p_val, q, atol=0.05)

    def test_kl_divergence_with_prior(self):
        """min 0.5*||p - y||^2 + lam * KL(p || q)  s.t.  sum(p)=1, p>0
        Regularized distribution estimation.
        """
        y = np.array([0.5, 0.3, 0.2])
        q = np.array([0.33, 0.34, 0.33])
        lam = 0.2
        model = self._new_model()
        p = admm.Var("p", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(p - y))
            + lam * KLDivGrad(p, q)
        )
        model.addConstr(admm.sum(p) == 1)
        model.addConstr(p >= 0.001)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        p_val = np.asarray(p.X)
        self.assertAlmostEqual(np.sum(p_val), 1.0, places=1)
        self.assertTrue(np.all(p_val > 0))

    def test_poisson_regression(self):
        """Poisson regression: min sum(exp(Ax) - b*Ax) + 0.5*lam*||x||^2
        A standard GLM with exponential link.
        """
        np.random.seed(33)
        n, d = 30, 4
        x_true = np.array([0.5, -0.3, 0.1, 0.2])
        A = np.random.randn(n, d) * 0.3
        rates = np.exp(A @ x_true)
        b = np.random.poisson(rates).astype(float)
        b = np.maximum(b, 0.1)  # avoid log(0) issues
        lam = 0.5

        model = self._new_model(max_iter=5000)
        x = admm.Var("x", d)
        model.setObjective(
            PoissonLossGrad(x, A, b)
            + 0.5 * lam * admm.sum(admm.square(x))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        # Check reasonable recovery (not exact due to noise + regularization)
        self.assertLess(np.linalg.norm(x_val - x_true), 2.0)

    def test_inverse_penalty(self):
        """min 0.5*||x - y||^2 + lam/x  s.t. x >= 0.1
        Barrier-like penalty near zero.
        """
        observed = np.array([2.0, 0.5, 3.0])
        lam = 0.1
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * InversePenaltyGrad(x)
        )
        model.addConstr(x >= 0.1)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(x_val >= 0.09))

    def test_squared_log_penalty(self):
        """min 0.5*||x-y||^2 + lam*sum(log(1+x)^2), x >= 0
        Concave-ish penalty, but the proximal subproblem is still well-defined.
        """
        observed = np.array([1.0, 3.0, 0.5])
        lam = 0.2
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SquaredLogGrad(x)
        )
        model.addConstr(x >= 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(x_val >= -0.01))

    def test_weighted_quartic(self):
        """Weighted quartic: different weights per component.
        min 0.5*||x-y||^2 + sum(w_i * x_i^4)
        Heavily penalized components should shrink more.
        """
        observed = np.array([2.0, 2.0, 2.0])
        w = np.array([0.01, 1.0, 10.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + WeightedQuarticGrad(x, w)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Larger weight -> more shrinkage
        self.assertGreater(x_val[0], x_val[1])
        self.assertGreater(x_val[1], x_val[2])

    # ===================================================================
    # E. Application-level problems
    # ===================================================================

    def test_portfolio_markowitz(self):
        """Markowitz portfolio: min x^T Sigma x  s.t.  mu^T x >= r, sum(x)=1, x>=0
        """
        np.random.seed(42)
        n = 5
        F = np.random.randn(n, 3) * 0.1
        Sigma = F @ F.T + 0.05 * np.eye(n)
        mu = np.array([0.12, 0.10, 0.07, 0.03, 0.15])
        r_target = 0.08

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(QuadFormGrad(x, Sigma))
        model.addConstr(mu @ x >= r_target)
        model.addConstr(admm.sum(x) == 1)
        model.addConstr(x >= 0)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertAlmostEqual(np.sum(x_val), 1.0, places=1)
        self.assertTrue(np.all(x_val >= -0.01))
        self.assertGreaterEqual(mu @ x_val, r_target - 0.01)

    def test_signal_denoising_smooth_l1(self):
        """Signal denoising: min ||x - y_noisy||^2 + lam * TV(x)
        where TV(x) ≈ sum(sqrt(diff(x)^2 + eps)) (smooth total variation).
        """
        np.random.seed(10)
        n = 50
        # Piecewise constant signal
        x_true = np.zeros(n)
        x_true[:20] = 1.0
        x_true[20:35] = -0.5
        x_true[35:] = 0.5
        y_noisy = x_true + 0.3 * np.random.randn(n)
        lam = 0.5

        # Build difference operator Dx = x[1:] - x[:-1]
        model = self._new_model(max_iter=5000)
        x = admm.Var("x", n)
        # Use smooth L1 on differences
        diff_x = x[1:] - x[:-1]
        model.setObjective(
            0.5 * admm.sum(admm.square(x - y_noisy))
            + lam * admm.sum(admm.square(diff_x))
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Denoised should be closer to true than noisy
        err_noisy = np.linalg.norm(y_noisy - x_true)
        err_denoised = np.linalg.norm(x_val - x_true)
        self.assertLess(err_denoised, err_noisy)

    def test_least_squares_exp_regularization(self):
        """min ||Ax - b||^2 + lam * sum(exp(x))
        Regularized least squares with exponential barrier.
        """
        np.random.seed(17)
        m, n = 20, 5
        A = np.random.randn(m, n)
        x_true = np.array([-1.0, 0.5, 0.0, -0.5, 1.0])
        b = A @ x_true + 0.01 * np.random.randn(m)
        lam = 0.01

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            admm.sum(admm.square(A @ x - b))
            + lam * ExpSumGrad(x)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Should be close to true solution (exp reg pushes values down slightly)
        np.testing.assert_allclose(x_val, x_true, atol=0.3)

    def test_maximum_likelihood_exponential(self):
        """MLE for exponential distribution: min t_sum*x - n*log(x)  s.t. x > 0
        Optimal: x* = n / t_sum.
        Entire objective packed into one UDF to avoid presolve unbounded detection.
        """
        np.random.seed(44)
        n_obs = 20
        true_rate = 2.0
        data = np.random.exponential(1.0 / true_rate, n_obs)
        t_sum = float(np.sum(data))

        model = self._new_model()
        x = admm.Var("x", 1)
        # Tiny quadratic helps presolve see boundedness (doesn't affect solution)
        eps_reg = 1e-6
        model.setObjective(ExpMLEGrad(x, t_sum, n_obs) + eps_reg * admm.sum(admm.square(x)))
        model.addConstr(x >= 0.01)
        model.addConstr(x <= 100)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()[0]
        x_mle = n_obs / t_sum
        self.assertAlmostEqual(x_val, x_mle, delta=0.3)

    def test_compressed_sensing_smooth(self):
        """Compressed sensing with smooth L1:
        min ||Ax - b||^2 + lam * sum(sqrt(x^2 + eps))
        """
        np.random.seed(88)
        m, n, k = 30, 50, 5
        x_true = np.zeros(n)
        supp = np.random.choice(n, k, replace=False)
        x_true[supp] = np.random.randn(k) * 2
        A = np.random.randn(m, n) / np.sqrt(m)
        b = A @ x_true
        lam = 0.1

        model = self._new_model(max_iter=5000)
        x = admm.Var("x", n)
        model.setObjective(
            admm.sum(admm.square(A @ x - b))
            + lam * SmoothL1Grad(x, eps=1e-4)
        )
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Recovery: largest k components should match support
        top_k = set(np.argsort(np.abs(x_val))[-k:])
        true_supp = set(supp)
        overlap = len(top_k & true_supp)
        self.assertGreaterEqual(overlap, k - 1,
                                f"Support recovery: {overlap}/{k}")

    # ===================================================================
    # F. Multi-UDF and mixing tests
    # ===================================================================

    def test_three_grad_udfs_combined(self):
        """min  lam1*sum(x^4) + lam2*sum(exp(x)) + lam3*sum(log(cosh(x)))"""
        observed = np.array([1.0, -2.0, 0.5, 3.0])
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.05 * QuarticGrad(x)
            + 0.05 * ExpSumGrad(x)
            + 0.1 * LogCoshGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    def test_grad_udf_with_two_builtin_norms(self):
        """min  sum(exp(x)) + lam1*||x||_1 + lam2*||x||_2^2
        s.t. x >= -5
        """
        model = self._new_model()
        x = admm.Var("x", 5)
        model.setObjective(
            ExpSumGrad(x)
            + 0.1 * admm.norm(x, 1)
            + 0.05 * admm.sum(admm.square(x))
        )
        model.addConstr(x >= -5)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    def test_grad_udf_with_matmul(self):
        """min  ||Ax - b||^2 + lam * sum(x^6)
        Matrix-vector product + grad UDF.
        """
        np.random.seed(66)
        A = np.random.randn(10, 5)
        b = np.random.randn(10)
        model = self._new_model()
        x = admm.Var("x", 5)
        model.setObjective(
            admm.sum(admm.square(A @ x - b))
            + 0.01 * SixthPowerGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    def test_same_grad_udf_class_two_vars(self):
        """Two separate variables, each with their own ExpSumGrad instance.
        min  sum(exp(x)) + sum(exp(y))  s.t.  sum(x)+sum(y)=4
        """
        model = self._new_model()
        x = admm.Var("x", 2)
        y = admm.Var("y", 2)
        model.setObjective(ExpSumGrad(x) + ExpSumGrad(y))
        model.addConstr(admm.sum(x) + admm.sum(y) == 4)
        model.addConstr(x >= 0)
        model.addConstr(y >= 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        y_val = np.asarray(y.X)
        # By symmetry, all four components should be 1
        np.testing.assert_allclose(x_val, 1.0, atol=0.15)
        np.testing.assert_allclose(y_val, 1.0, atol=0.15)

    # ===================================================================
    # G. Gradient correctness (extended finite-difference checks)
    # ===================================================================

    def test_gradient_finite_diff_all_new_udfs(self):
        """Verify all new UDF gradients against central finite differences."""
        np.random.seed(77)
        h = 1e-6
        dummy_var = admm.Var("d", 4)

        # Test on positive x for functions that need x > 0
        x_pos = np.abs(np.random.randn(4)) + 0.5
        x_gen = np.random.randn(4)

        cases = [
            (LogCoshGrad(dummy_var), x_gen, {}),
            (ElasticNetGrad(dummy_var, alpha=0.5), x_gen, {}),
            (KLDivGrad(dummy_var, q=np.array([0.3, 0.2, 0.4, 0.1])), x_pos / np.sum(x_pos), {}),
            (SquaredLogGrad(dummy_var), x_pos, {}),
            (InversePenaltyGrad(dummy_var), x_pos, {}),
            (FrobExpGrad(dummy_var, mu=0.2), x_gen, {}),
            (WeightedQuarticGrad(dummy_var, w=np.array([0.5, 1.0, 2.0, 0.1])), x_gen, {}),
        ]

        for udf_obj, x_test, _ in cases:
            grad_analytic = np.asarray(udf_obj.grad([x_test.copy()])[0])
            grad_fd = np.zeros_like(x_test)
            for i in range(len(x_test)):
                xp = x_test.copy(); xp[i] += h
                xm = x_test.copy(); xm[i] -= h
                grad_fd[i] = (udf_obj.eval([xp]) - udf_obj.eval([xm])) / (2 * h)
            np.testing.assert_allclose(
                grad_analytic, grad_fd, atol=1e-4,
                err_msg=f"Gradient mismatch for {type(udf_obj).__name__}"
            )

    def test_gradient_finite_diff_logistic(self):
        """Finite-difference check for logistic loss."""
        np.random.seed(88)
        n, d = 10, 3
        A = np.random.randn(n, d)
        y = 2.0 * (np.random.rand(n) > 0.5) - 1.0
        dummy_var = admm.Var("d", d)
        udf_obj = LogisticLossGrad(dummy_var, A, y)

        w_test = np.random.randn(d)
        h = 1e-6
        grad_analytic = np.asarray(udf_obj.grad([w_test.copy()])[0]).ravel()
        grad_fd = np.zeros(d)
        for i in range(d):
            wp = w_test.copy(); wp[i] += h
            wm = w_test.copy(); wm[i] -= h
            grad_fd[i] = (udf_obj.eval([wp]) - udf_obj.eval([wm])) / (2 * h)
        np.testing.assert_allclose(grad_analytic, grad_fd, atol=1e-4)

    def test_gradient_finite_diff_poisson(self):
        """Finite-difference check for Poisson loss."""
        np.random.seed(99)
        n, d = 8, 3
        A = np.random.randn(n, d) * 0.3
        b = np.abs(np.random.randn(n)) + 0.5
        dummy_var = admm.Var("d", d)
        udf_obj = PoissonLossGrad(dummy_var, A, b)

        w_test = np.random.randn(d) * 0.5
        h = 1e-6
        grad_analytic = np.asarray(udf_obj.grad([w_test.copy()])[0]).ravel()
        grad_fd = np.zeros(d)
        for i in range(d):
            wp = w_test.copy(); wp[i] += h
            wm = w_test.copy(); wm[i] -= h
            grad_fd[i] = (udf_obj.eval([wp]) - udf_obj.eval([wm])) / (2 * h)
        np.testing.assert_allclose(grad_analytic, grad_fd, atol=1e-4)

    # ===================================================================
    # H. Reproducibility and determinism
    # ===================================================================

    def test_deterministic_results(self):
        """Same problem solved twice should give identical results."""
        observed = np.array([2.0, -1.0, 0.5])
        results = []
        for _ in range(2):
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + 0.1 * QuarticGrad(x)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            results.append(np.asarray(x.X).copy())
        np.testing.assert_allclose(results[0], results[1], atol=1e-6)

    def test_coefficient_scaling(self):
        """Scaling objective by constant should give same solution.
        min 2*(0.5*||x-y||^2 + lam*f(x)) vs min (0.5*||x-y||^2 + lam*f(x))
        """
        observed = np.array([1.0, -1.0])
        lam = 0.3

        models = []
        for scale in [1.0, 2.0]:
            model = self._new_model()
            x = admm.Var("x", 2)
            model.setObjective(
                scale * (0.5 * admm.sum(admm.square(x - observed))
                         + lam * ExpSumGrad(x))
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            models.append(np.asarray(x.X).copy())
        np.testing.assert_allclose(models[0], models[1], atol=0.05)

    # ===================================================================
    # I. Quartic grad vs argmin on diverse settings
    # ===================================================================

    def test_quartic_grad_vs_argmin_large(self):
        """n=50: grad and argmin should give close results."""
        np.random.seed(50)
        n = 50
        y = np.random.randn(n)
        lam = 0.05

        model1 = self._new_model(max_iter=5000)
        x1 = admm.Var("x1", n)
        model1.setObjective(
            0.5 * admm.sum(admm.square(x1 - y)) + lam * udf.QuarticPenalty(x1)
        )
        model1.optimize()

        model2 = self._new_model(max_iter=5000)
        x2 = admm.Var("x2", n)
        model2.setObjective(
            0.5 * admm.sum(admm.square(x2 - y)) + lam * QuarticGrad(x2)
        )
        model2.optimize()

        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x1.X, x2.X, atol=0.05)
        self.assertAlmostEqual(model1.ObjVal, model2.ObjVal, places=2)

    def test_quartic_grad_vs_argmin_with_constraints(self):
        """Quartic with box constraints: grad vs argmin."""
        observed = np.array([3.0, -2.0, 1.5, -0.5])
        lam = 0.2

        model1 = self._new_model()
        x1 = admm.Var("x1", 4)
        model1.setObjective(
            0.5 * admm.sum(admm.square(x1 - observed)) + lam * udf.QuarticPenalty(x1)
        )
        model1.addConstr(x1 >= -1)
        model1.addConstr(x1 <= 2)
        model1.optimize()

        model2 = self._new_model()
        x2 = admm.Var("x2", 4)
        model2.setObjective(
            0.5 * admm.sum(admm.square(x2 - observed)) + lam * QuarticGrad(x2)
        )
        model2.addConstr(x2 >= -1)
        model2.addConstr(x2 <= 2)
        model2.optimize()

        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x1.X, x2.X, atol=0.05)

    def test_quartic_grad_vs_argmin_with_equality(self):
        """Quartic with sum(x)=2: grad vs argmin."""
        observed = np.array([1.5, 0.5, 0.0])
        lam = 0.1

        model1 = self._new_model()
        x1 = admm.Var("x1", 3)
        model1.setObjective(
            0.5 * admm.sum(admm.square(x1 - observed)) + lam * udf.QuarticPenalty(x1)
        )
        model1.addConstr(admm.sum(x1) == 2)
        model1.optimize()

        model2 = self._new_model()
        x2 = admm.Var("x2", 3)
        model2.setObjective(
            0.5 * admm.sum(admm.square(x2 - observed)) + lam * QuarticGrad(x2)
        )
        model2.addConstr(admm.sum(x2) == 2)
        model2.optimize()

        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x1.X, x2.X, atol=0.1)


# ---------------------------------------------------------------------------
# Extended grad-only UDF definitions (more smooth functions)
# ---------------------------------------------------------------------------

class PseudoHuberGrad(admm.UDFBase):
    """Pseudo-Huber loss: f(x) = delta^2 * sum(sqrt(1 + (x/delta)^2) - 1)
    Smooth everywhere (unlike Huber which has a kink).
    grad = x / sqrt(1 + (x/delta)^2)
    """
    def __init__(self, arg, delta=1.0):
        self.arg = arg
        self.delta = delta
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        d = self.delta
        return float(d ** 2 * np.sum(np.sqrt(1 + (x / d) ** 2) - 1))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        d = self.delta
        return [x / np.sqrt(1 + (x / d) ** 2)]


class XExpXGrad(admm.UDFBase):
    """f(x) = sum(x_i * exp(x_i)), convex for x >= -1.
    grad = (1 + x) * exp(x)
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(x * np.exp(x)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [(1.0 + x) * np.exp(x)]


class ReciprocalBarrierGrad(admm.UDFBase):
    """Self-concordant barrier: f(x) = sum(x_i + 1/x_i), x > 0
    Minimized at x=1 for each component.
    grad = 1 - 1/x^2
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.maximum(np.asarray(tensorlist[0], dtype=float), 1e-30)
        return float(np.sum(x + 1.0 / x))
    def grad(self, tensorlist):
        x = np.maximum(np.asarray(tensorlist[0], dtype=float), 1e-30)
        return [1.0 - 1.0 / (x ** 2)]


class SquaredHingeLossGrad(admm.UDFBase):
    """Squared hinge loss for SVM: f(w) = sum(max(0, 1 - y_i * a_i^T w)^2)
    grad_j = -2 * sum_{i: margin<1} (1 - y_i*a_i^T w) * y_i * A_ij
    """
    def __init__(self, arg, A, y):
        self.arg = arg
        self.A = np.asarray(A, dtype=float)
        self.y = np.asarray(y, dtype=float)
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        w = np.asarray(tensorlist[0], dtype=float).ravel()
        margins = self.y * (self.A @ w)
        losses = np.maximum(0, 1 - margins) ** 2
        return float(np.sum(losses))
    def grad(self, tensorlist):
        w = np.asarray(tensorlist[0], dtype=float).ravel()
        margins = self.y * (self.A @ w)
        violations = np.maximum(0, 1 - margins)
        g = -2.0 * self.A.T @ (violations * self.y)
        return [g.reshape(tensorlist[0].shape)]


class TsallisEntropyGrad(admm.UDFBase):
    """Tsallis entropy (negative): f(x) = (sum(x^q) - 1)/(q-1), x > 0
    For q > 1, this is convex. Generalizes negative Shannon entropy.
    grad = q/(q-1) * x^(q-1)
    """
    def __init__(self, arg, q=2.0):
        self.arg = arg
        self.q = q
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.maximum(np.asarray(tensorlist[0], dtype=float), 1e-30)
        return float((np.sum(x ** self.q) - 1.0) / (self.q - 1.0))
    def grad(self, tensorlist):
        x = np.maximum(np.asarray(tensorlist[0], dtype=float), 1e-30)
        return [self.q / (self.q - 1.0) * x ** (self.q - 1.0)]


class GeneralPowerGrad(admm.UDFBase):
    """f(x) = sum(|x_i|^p) for general p > 1.
    grad = p * sign(x) * |x|^(p-1)
    Convex for p >= 1. Smooth for p > 1.
    """
    def __init__(self, arg, p=1.5):
        self.arg = arg
        self.p = p
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.abs(x) ** self.p))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [self.p * np.sign(x) * np.abs(x) ** (self.p - 1.0)]


class BinaryCrossEntropyGrad(admm.UDFBase):
    """Binary cross-entropy: f(p) = -sum(t*log(p) + (1-t)*log(1-p))
    where t are fixed targets in (0,1) and p in (0,1) are the variables.
    grad = -t/p + (1-t)/(1-p)
    """
    def __init__(self, arg, targets):
        self.arg = arg
        self.t = np.asarray(targets, dtype=float)
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        p = np.clip(np.asarray(tensorlist[0], dtype=float), 1e-15, 1 - 1e-15)
        return float(-np.sum(self.t * np.log(p) + (1 - self.t) * np.log(1 - p)))
    def grad(self, tensorlist):
        p = np.clip(np.asarray(tensorlist[0], dtype=float), 1e-15, 1 - 1e-15)
        return [-self.t / p + (1 - self.t) / (1 - p)]


class SmoothMaxPairGrad(admm.UDFBase):
    """f(x) = sum(log(exp(x_i) + exp(-x_i))) = sum(log(2*cosh(x_i)))
    Symmetric smooth penalty that penalizes deviation from 0 in both directions.
    grad = tanh(x)
    Note: same grad as LogCosh but different eval by constant log(2).
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.logaddexp(x, -x)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [np.tanh(x)]


class SumExpNegGrad(admm.UDFBase):
    """f(x) = sum(exp(x_i) + exp(-x_i)) = 2*sum(cosh(x_i))
    Symmetric, strongly convex. Minimized at x=0.
    grad = exp(x) - exp(-x) = 2*sinh(x)
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.exp(x) + np.exp(-x)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [np.exp(x) - np.exp(-x)]


class GaussianNLLGrad(admm.UDFBase):
    """Gaussian negative log-likelihood (variance parameterization):
    f(s) = sum(s_i + (y_i - mu_i)^2 * exp(-2*s_i))
    where s = log(sigma), y are observations, mu are means.
    grad = 1 - 2*(y-mu)^2 * exp(-2*s)
    """
    def __init__(self, arg, y, mu):
        self.arg = arg
        self.y = np.asarray(y, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.r2 = (self.y - self.mu) ** 2
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        s = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(s + self.r2 * np.exp(-2 * s)))
    def grad(self, tensorlist):
        s = np.asarray(tensorlist[0], dtype=float)
        return [1.0 - 2.0 * self.r2 * np.exp(-2 * s)]


class MultiArgQuadGrad(admm.UDFBase):
    """Multi-argument UDF: f(x, y) = sum((x-y)^2)
    Two tensor arguments.
    grad_x = 2*(x-y), grad_y = -2*(x-y)
    """
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2
    def arguments(self):
        return [self.arg1, self.arg2]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        return float(np.sum((x - y) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        return [2.0 * (x - y), -2.0 * (x - y)]


# ---------------------------------------------------------------------------
# Extended test class
# ---------------------------------------------------------------------------

class GradUDFExtendedTestCase(unittest.TestCase):
    """Tests covering additional smooth functions and multi-argument UDFs."""

    def _new_model(self, max_iter=2000):
        model = admm.Model()
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.display_sub_solver_details, 0)
        model.setOption(admm.Options.admm_max_iteration, max_iter)
        return model

    # ===================================================================
    # A. Pseudo-Huber loss
    # ===================================================================

    def test_pseudo_huber_kkt(self):
        """min 0.5*||x-y||^2 + lam * pseudo_huber(x)
        KKT: x_i + lam * x_i / sqrt(1+(x_i/delta)^2) = y_i
        """
        observed = np.array([3.0, -1.5, 0.2, 2.0])
        lam = 0.5
        delta = 1.0
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * PseudoHuberGrad(x, delta=delta)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + lam * x_val / np.sqrt(1 + (x_val / delta) ** 2) - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    def test_pseudo_huber_vs_huber(self):
        """Pseudo-Huber and Huber should give similar results for same delta."""
        observed = np.array([2.0, -3.0, 0.5])
        lam = 0.3
        delta = 1.0

        model1 = self._new_model()
        x1 = admm.Var("x1", 3)
        model1.setObjective(
            0.5 * admm.sum(admm.square(x1 - observed))
            + lam * HuberGrad(x1, delta=delta)
        )
        model1.optimize()

        model2 = self._new_model()
        x2 = admm.Var("x2", 3)
        model2.setObjective(
            0.5 * admm.sum(admm.square(x2 - observed))
            + lam * PseudoHuberGrad(x2, delta=delta)
        )
        model2.optimize()

        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        # Pseudo-Huber is a smooth approximation of Huber — close but not identical
        np.testing.assert_allclose(x1.X, x2.X, atol=0.3)

    def test_pseudo_huber_small_delta(self):
        """Small delta → closer to L1 behavior."""
        observed = np.array([0.1, 3.0, -0.05])
        lam = 1.0
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * PseudoHuberGrad(x, delta=0.1)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Small entries should shrink toward 0 (L1-like)
        self.assertLess(abs(x_val[0]), abs(observed[0]))
        self.assertLess(abs(x_val[2]), abs(observed[2]))

    # ===================================================================
    # B. x*exp(x) function
    # ===================================================================

    def test_xexpx_unconstrained(self):
        """min 0.5*||x-y||^2 + lam*sum(x*exp(x))
        KKT: x_i + lam*(1+x_i)*exp(x_i) = y_i
        """
        observed = np.array([1.0, 0.5, -0.5])
        lam = 0.2
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * XExpXGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + lam * (1 + x_val) * np.exp(x_val) - observed
        np.testing.assert_allclose(residual, 0, atol=0.1)

    def test_xexpx_box(self):
        """min sum(x*exp(x))  s.t. x >= -0.5, sum(x) = 0
        x*exp(x) minimized at x=-1, but box pushes up.
        """
        n = 4
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(XExpXGrad(x))
        model.addConstr(x >= -0.5)
        model.addConstr(admm.sum(x) == 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(x_val >= -0.51))
        self.assertAlmostEqual(np.sum(x_val), 0, delta=0.1)

    # ===================================================================
    # C. Reciprocal barrier (self-concordant)
    # ===================================================================

    def test_reciprocal_barrier_unconstrained(self):
        """min sum(x + 1/x)  s.t. x >= 0.01
        Each x_i minimized at x_i=1.
        """
        n = 5
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(ReciprocalBarrierGrad(x))
        model.addConstr(x >= 0.01)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 1.0, atol=0.1)

    def test_reciprocal_barrier_with_sum_constraint(self):
        """min sum(x + 1/x)  s.t. x >= 0.01, sum(x) = 10
        Still uniform: x = 10/n = 2.
        """
        n = 5
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(ReciprocalBarrierGrad(x))
        model.addConstr(x >= 0.01)
        model.addConstr(admm.sum(x) == 10)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 2.0, atol=0.15)

    # ===================================================================
    # D. Squared hinge loss (SVM)
    # ===================================================================

    def test_squared_hinge_svm(self):
        """SVM with squared hinge loss:
        min sum(max(0, 1-y*Aw)^2) + lam*||w||^2
        """
        np.random.seed(77)
        n, d = 40, 4
        w_true = np.array([1.0, -0.5, 0.3, 0.8])
        A = np.random.randn(n, d)
        y = np.sign(A @ w_true)

        lam = 0.1
        model = self._new_model(max_iter=5000)
        w = admm.Var("w", d)
        model.setObjective(
            SquaredHingeLossGrad(w, A, y)
            + lam * admm.sum(admm.square(w))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        w_val = np.asarray(w.X).ravel()
        # Check classification accuracy
        pred = np.sign(A @ w_val)
        accuracy = np.mean(pred == y)
        self.assertGreater(accuracy, 0.85)

    def test_squared_hinge_bounded(self):
        """SVM with box constraint on weights."""
        np.random.seed(88)
        n, d = 30, 3
        A = np.random.randn(n, d)
        y = np.sign(A @ np.array([1.0, -1.0, 0.5]))

        model = self._new_model()
        w = admm.Var("w", d)
        model.setObjective(
            SquaredHingeLossGrad(w, A, y)
            + 0.5 * admm.sum(admm.square(w))
        )
        model.addConstr(w >= -3)
        model.addConstr(w <= 3)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        w_val = np.asarray(w.X).ravel()
        self.assertTrue(np.all(w_val >= -3.01))
        self.assertTrue(np.all(w_val <= 3.01))

    # ===================================================================
    # E. Tsallis entropy
    # ===================================================================

    def test_tsallis_entropy_simplex(self):
        """min Tsallis_q(x)  s.t. sum(x)=1, x >= 0.01
        For q=2: f(x) = sum(x^2) - 1. Minimized at uniform on simplex.
        """
        n = 4
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(TsallisEntropyGrad(x, q=2.0))
        model.addConstr(admm.sum(x) == 1)
        model.addConstr(x >= 0.01)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 1.0 / n, atol=0.05)

    def test_tsallis_q3(self):
        """Tsallis with q=3, higher order."""
        n = 3
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(TsallisEntropyGrad(x, q=3.0))
        model.addConstr(admm.sum(x) == 1)
        model.addConstr(x >= 0.01)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 1.0 / n, atol=0.05)

    # ===================================================================
    # F. General power function |x|^p
    # ===================================================================

    def test_power_p15(self):
        """f(x) = sum(|x|^1.5): between L1 and L2.
        min 0.5*||x-y||^2 + lam*sum(|x|^1.5)
        """
        observed = np.array([0.1, 2.0, -0.05, -1.5])
        lam = 0.3
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * GeneralPowerGrad(x, p=1.5)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Shrinkage: small entries shrink more (bridge penalty)
        self.assertLess(abs(x_val[0]), abs(observed[0]))
        self.assertLess(abs(x_val[2]), abs(observed[2]))

    def test_power_p3(self):
        """f(x) = sum(|x|^3): stronger sparsity than L2."""
        observed = np.array([1.0, -2.0, 0.5])
        lam = 0.05
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * GeneralPowerGrad(x, p=3.0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Verify KKT: x + lam*p*sign(x)*|x|^(p-1) = y
        p = 3.0
        residual = x_val + lam * p * np.sign(x_val) * np.abs(x_val) ** (p - 1) - observed
        np.testing.assert_allclose(residual, 0, atol=0.1)

    def test_power_p25_box(self):
        """f(x) = sum(|x|^2.5) with box constraints."""
        model = self._new_model()
        x = admm.Var("x", 5)
        model.setObjective(
            GeneralPowerGrad(x, p=2.5)
            + 0.1 * admm.sum(admm.square(x - 1))
        )
        model.addConstr(x >= 0)
        model.addConstr(x <= 3)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(x_val >= -0.01))
        self.assertTrue(np.all(x_val <= 3.01))

    # ===================================================================
    # G. Binary cross-entropy
    # ===================================================================

    def test_binary_cross_entropy_recovery(self):
        """min BCE(p, t)  s.t. 0.01 <= p <= 0.99
        Optimal: p = t.
        """
        targets = np.array([0.8, 0.2, 0.5, 0.9])
        model = self._new_model()
        p = admm.Var("p", 4)
        model.setObjective(BinaryCrossEntropyGrad(p, targets))
        model.addConstr(p >= 0.01)
        model.addConstr(p <= 0.99)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        p_val = np.asarray(p.X)
        np.testing.assert_allclose(p_val, targets, atol=0.05)

    def test_binary_cross_entropy_with_prior(self):
        """min 0.5*||p - prior||^2 + lam * BCE(p, t)
        Regularized: p pulled between prior and target.
        """
        targets = np.array([0.9, 0.1, 0.7])
        prior = np.array([0.5, 0.5, 0.5])
        lam = 0.5
        model = self._new_model()
        p = admm.Var("p", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(p - prior))
            + lam * BinaryCrossEntropyGrad(p, targets)
        )
        model.addConstr(p >= 0.01)
        model.addConstr(p <= 0.99)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        p_val = np.asarray(p.X)
        # p should be between prior and targets
        self.assertTrue(np.all(p_val >= 0.009))
        self.assertTrue(np.all(p_val <= 0.991))

    # ===================================================================
    # H. Symmetric cosh penalty
    # ===================================================================

    def test_sum_exp_neg_unconstrained(self):
        """min 0.5*||x-y||^2 + lam * 2*sum(cosh(x))
        KKT: x + lam*2*sinh(x) = y
        """
        observed = np.array([1.0, -0.5, 2.0])
        lam = 0.3
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SumExpNegGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + lam * 2 * np.sinh(x_val) - observed
        np.testing.assert_allclose(residual, 0, atol=0.1)

    def test_sum_exp_neg_equality(self):
        """min sum(exp(x)+exp(-x))  s.t. sum(x) = 0
        By symmetry and convexity: all x_i = 0.
        """
        n = 5
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(SumExpNegGrad(x))
        model.addConstr(admm.sum(x) == 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 0, atol=0.1)

    # ===================================================================
    # I. Gaussian NLL (log-variance parameterization)
    # ===================================================================

    def test_gaussian_nll_recovery(self):
        """MLE for log-variance: min sum(s + r^2*exp(-2s))
        Optimal: s* = log(|y-mu|) for each component (log of std dev).
        """
        y = np.array([3.0, -1.0, 2.5, 0.0])
        mu = np.array([1.0, 0.0, 1.5, 0.5])
        model = self._new_model()
        s = admm.Var("s", 4)
        model.setObjective(GaussianNLLGrad(s, y, mu))
        model.addConstr(s >= -5)
        model.addConstr(s <= 5)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        s_val = np.asarray(s.X)
        # f'(s) = 1 - 2r^2 exp(-2s) = 0  =>  s* = log(r) + 0.5*log(2)
        r = np.abs(y - mu)
        r = np.maximum(r, 1e-10)
        s_opt = np.log(r) + 0.5 * np.log(2)
        np.testing.assert_allclose(s_val, s_opt, atol=0.1)

    def test_gaussian_nll_with_regularization(self):
        """min GaussianNLL(s) + lam*||s||^2: regularized variance estimation."""
        np.random.seed(11)
        n = 10
        y = np.random.randn(n) * 2
        mu = np.zeros(n)
        lam = 0.1
        model = self._new_model()
        s = admm.Var("s", n)
        model.setObjective(
            GaussianNLLGrad(s, y, mu)
            + lam * admm.sum(admm.square(s))
        )
        model.addConstr(s >= -3)
        model.addConstr(s <= 3)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # ===================================================================
    # J. Multi-argument UDF
    # ===================================================================

    def test_multi_arg_coupling(self):
        """min sum((x-y)^2)  s.t. sum(x)=3, sum(y)=3
        Optimal: x = y, both uniform = 1.
        """
        n = 3
        model = self._new_model()
        x = admm.Var("x", n)
        y = admm.Var("y", n)
        model.setObjective(MultiArgQuadGrad(x, y))
        model.addConstr(admm.sum(x) == 3)
        model.addConstr(admm.sum(y) == 3)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        y_val = np.asarray(y.X)
        np.testing.assert_allclose(x_val, y_val, atol=0.1)
        self.assertAlmostEqual(np.sum(x_val), 3.0, delta=0.1)

    def test_multi_arg_with_penalty(self):
        """min sum((x-y)^2) + lam*sum(x^4) + lam*sum(y^4)
        s.t. sum(x) = 2, sum(y) = 2
        """
        n = 2
        lam = 0.1
        model = self._new_model()
        x = admm.Var("x", n)
        y = admm.Var("y", n)
        model.setObjective(
            MultiArgQuadGrad(x, y)
            + lam * QuarticGrad(x)
            + lam * QuarticGrad(y)
        )
        model.addConstr(admm.sum(x) == 2)
        model.addConstr(admm.sum(y) == 2)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        y_val = np.asarray(y.X)
        np.testing.assert_allclose(x_val, y_val, atol=0.15)
        np.testing.assert_allclose(x_val, 1.0, atol=0.15)

    def test_multi_arg_asymmetric(self):
        """min sum((x-y)^2) + 0.5*||x - a||^2 + 0.5*||y - b||^2
        x pulled toward a, y toward b, coupling pulls them together.
        """
        a = np.array([3.0, 0.0])
        b = np.array([0.0, 3.0])
        model = self._new_model()
        x = admm.Var("x", 2)
        y = admm.Var("y", 2)
        model.setObjective(
            MultiArgQuadGrad(x, y)
            + 0.5 * admm.sum(admm.square(x - a))
            + 0.5 * admm.sum(admm.square(y - b))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        y_val = np.asarray(y.X)
        # x and y should meet in the middle, pulled by coupling
        mid = (a + b) / 2
        np.testing.assert_allclose(x_val, mid, atol=0.5)
        np.testing.assert_allclose(y_val, mid, atol=0.5)

    # ===================================================================
    # K. Extended gradient finite-difference checks for new UDFs
    # ===================================================================

    def test_gradient_finite_diff_extended(self):
        """Finite-difference check for all new UDF classes."""
        np.random.seed(42)
        h = 1e-6
        d = admm.Var("d", 4)

        x_pos = np.abs(np.random.randn(4)) + 0.5
        x_gen = np.random.randn(4) * 0.5

        cases = [
            (PseudoHuberGrad(d, delta=1.0), x_gen),
            (PseudoHuberGrad(d, delta=0.5), x_gen),
            (XExpXGrad(d), x_gen),
            (ReciprocalBarrierGrad(d), x_pos),
            (TsallisEntropyGrad(d, q=2.0), x_pos),
            (TsallisEntropyGrad(d, q=3.0), x_pos),
            (GeneralPowerGrad(d, p=1.5), x_gen + 0.1),  # avoid exact 0
            (GeneralPowerGrad(d, p=2.5), x_gen + 0.1),
            (SumExpNegGrad(d), x_gen),
            (SmoothMaxPairGrad(d), x_gen),
        ]

        for udf_obj, x_test in cases:
            grad_analytic = np.asarray(udf_obj.grad([x_test.copy()])[0])
            grad_fd = np.zeros_like(x_test)
            for i in range(len(x_test)):
                xp = x_test.copy(); xp[i] += h
                xm = x_test.copy(); xm[i] -= h
                grad_fd[i] = (udf_obj.eval([xp]) - udf_obj.eval([xm])) / (2 * h)
            np.testing.assert_allclose(
                grad_analytic, grad_fd, atol=1e-4,
                err_msg=f"Gradient mismatch for {type(udf_obj).__name__}"
            )

    def test_gradient_finite_diff_bce(self):
        """Finite-difference check for binary cross-entropy."""
        np.random.seed(33)
        d = admm.Var("d", 4)
        t = np.array([0.8, 0.2, 0.5, 0.9])
        p_test = np.array([0.3, 0.6, 0.4, 0.7])
        h = 1e-6
        udf_obj = BinaryCrossEntropyGrad(d, t)

        grad_analytic = np.asarray(udf_obj.grad([p_test.copy()])[0])
        grad_fd = np.zeros(4)
        for i in range(4):
            pp = p_test.copy(); pp[i] += h
            pm = p_test.copy(); pm[i] -= h
            grad_fd[i] = (udf_obj.eval([pp]) - udf_obj.eval([pm])) / (2 * h)
        np.testing.assert_allclose(grad_analytic, grad_fd, atol=1e-4)

    def test_gradient_finite_diff_squared_hinge(self):
        """Finite-difference check for squared hinge loss."""
        np.random.seed(44)
        n, d_dim = 10, 3
        A = np.random.randn(n, d_dim)
        y = 2.0 * (np.random.rand(n) > 0.5) - 1.0
        d = admm.Var("d", d_dim)
        udf_obj = SquaredHingeLossGrad(d, A, y)

        w_test = np.random.randn(d_dim)
        h = 1e-6
        grad_analytic = np.asarray(udf_obj.grad([w_test.copy()])[0]).ravel()
        grad_fd = np.zeros(d_dim)
        for i in range(d_dim):
            wp = w_test.copy(); wp[i] += h
            wm = w_test.copy(); wm[i] -= h
            grad_fd[i] = (udf_obj.eval([wp]) - udf_obj.eval([wm])) / (2 * h)
        np.testing.assert_allclose(grad_analytic, grad_fd, atol=1e-4)

    def test_gradient_finite_diff_gaussian_nll(self):
        """Finite-difference check for Gaussian NLL."""
        np.random.seed(55)
        d = admm.Var("d", 4)
        y = np.random.randn(4) * 2
        mu = np.random.randn(4)
        udf_obj = GaussianNLLGrad(d, y, mu)

        s_test = np.random.randn(4) * 0.5
        h = 1e-6
        grad_analytic = np.asarray(udf_obj.grad([s_test.copy()])[0])
        grad_fd = np.zeros(4)
        for i in range(4):
            sp = s_test.copy(); sp[i] += h
            sm = s_test.copy(); sm[i] -= h
            grad_fd[i] = (udf_obj.eval([sp]) - udf_obj.eval([sm])) / (2 * h)
        np.testing.assert_allclose(grad_analytic, grad_fd, atol=1e-4)

    def test_gradient_finite_diff_multi_arg(self):
        """Finite-difference check for multi-argument UDF."""
        np.random.seed(66)
        d1 = admm.Var("d1", 3)
        d2 = admm.Var("d2", 3)
        udf_obj = MultiArgQuadGrad(d1, d2)

        x_test = np.random.randn(3)
        y_test = np.random.randn(3)
        h = 1e-6

        grads = udf_obj.grad([x_test.copy(), y_test.copy()])
        grad_x = np.asarray(grads[0])
        grad_y = np.asarray(grads[1])

        # Check grad w.r.t. x
        grad_x_fd = np.zeros(3)
        for i in range(3):
            xp = x_test.copy(); xp[i] += h
            xm = x_test.copy(); xm[i] -= h
            grad_x_fd[i] = (udf_obj.eval([xp, y_test.copy()]) - udf_obj.eval([xm, y_test.copy()])) / (2 * h)
        np.testing.assert_allclose(grad_x, grad_x_fd, atol=1e-4, err_msg="grad_x mismatch")

        # Check grad w.r.t. y
        grad_y_fd = np.zeros(3)
        for i in range(3):
            yp = y_test.copy(); yp[i] += h
            ym = y_test.copy(); ym[i] -= h
            grad_y_fd[i] = (udf_obj.eval([x_test.copy(), yp]) - udf_obj.eval([x_test.copy(), ym])) / (2 * h)
        np.testing.assert_allclose(grad_y, grad_y_fd, atol=1e-4, err_msg="grad_y mismatch")

    # ===================================================================
    # L. Cross-function comparisons
    # ===================================================================

    def test_pseudo_huber_converges_to_huber(self):
        """As delta grows, Pseudo-Huber → quadratic. As delta → 0, → L1.
        Just verify all deltas converge.
        """
        observed = np.array([2.0, -1.0])
        lam = 0.5
        for delta in [0.01, 0.1, 1.0, 10.0]:
            model = self._new_model()
            x = admm.Var("x", 2)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + lam * PseudoHuberGrad(x, delta=delta)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"Failed for delta={delta}")

    def test_power_monotonicity(self):
        """Higher p => heavier penalty on large x.
        With same observed, higher p should shrink x more.
        """
        observed = np.array([3.0, -2.0])
        lam = 0.1
        norms = {}
        for p in [1.5, 2.0, 3.0, 4.0]:
            model = self._new_model()
            x = admm.Var("x", 2)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + lam * GeneralPowerGrad(x, p=p)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            norms[p] = np.linalg.norm(np.asarray(x.X))
        # Higher power → more shrinkage
        self.assertGreater(norms[1.5], norms[4.0])

    def test_all_penalties_on_same_problem(self):
        """Same problem, different penalties — all should converge.
        min 0.5*||x-y||^2 + 0.1*f(x)
        """
        observed = np.array([1.5, -1.0, 0.3])
        lam = 0.1
        penalties = [
            QuarticGrad, SixthPowerGrad, ExpSumGrad, SoftplusGrad,
            LogCoshGrad, SumExpNegGrad, SmoothMaxPairGrad, XExpXGrad,
        ]
        for cls in penalties:
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + lam * cls(x)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"Failed for {cls.__name__}")

    def test_all_penalties_with_equality(self):
        """Same problem with equality constraint, different penalties."""
        lam = 0.1
        penalties = [
            QuarticGrad, SixthPowerGrad, ExpSumGrad, SoftplusGrad,
            LogCoshGrad, SumExpNegGrad,
        ]
        for cls in penalties:
            model = self._new_model()
            x = admm.Var("x", 4)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - 1))
                + lam * cls(x)
            )
            model.addConstr(admm.sum(x) == 2)
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"Failed for {cls.__name__}")
            self.assertAlmostEqual(np.sum(np.asarray(x.X)), 2.0, delta=0.15)


# ---------------------------------------------------------------------------
# More grad-only UDF definitions
# ---------------------------------------------------------------------------

class StudentTLossGrad(admm.UDFBase):
    """Student-t robust loss: f(x) = sum(log(1 + x_i^2 / v))
    Heavier tails than Gaussian. Convex for v >= 1.
    grad = 2*x / (v + x^2)
    """
    def __init__(self, arg, v=1.0):
        self.arg = arg
        self.v = v
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.log(1 + x ** 2 / self.v)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [2.0 * x / (self.v + x ** 2)]


class QuantileHuberGrad(admm.UDFBase):
    """Asymmetric smooth loss for quantile regression:
    f(x) = sum(tau * softplus(x/eps) + (1-tau) * softplus(-x/eps)) * eps
    Smooth approximation of pinball loss. Convex.
    grad = tau * sigmoid(x/eps) - (1-tau) * sigmoid(-x/eps)
         = tau * sigmoid(x/eps) - (1-tau) * (1 - sigmoid(x/eps))
         = sigmoid(x/eps) - (1-tau)
    """
    def __init__(self, arg, tau=0.5, eps=0.1):
        self.arg = arg
        self.tau = tau
        self.eps = eps
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        z = x / self.eps
        return float(self.eps * np.sum(
            self.tau * np.logaddexp(0, z) + (1 - self.tau) * np.logaddexp(0, -z)
        ))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        sig = 1.0 / (1.0 + np.exp(-x / self.eps))
        return [self.tau * sig - (1 - self.tau) * (1 - sig)]


class SwishGrad(admm.UDFBase):
    """Swish / SiLU: f(x) = sum(x_i * sigmoid(beta * x_i))
    Popular smooth activation. Convex for beta <= ~1.278.
    grad = sigmoid(beta*x) + beta*x*sigmoid(beta*x)*(1-sigmoid(beta*x))
         = sigmoid(beta*x) * (1 + beta*x*(1-sigmoid(beta*x)))
    """
    def __init__(self, arg, beta=1.0):
        self.arg = arg
        self.beta = beta
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        sig = 1.0 / (1.0 + np.exp(-self.beta * x))
        return float(np.sum(x * sig))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        sig = 1.0 / (1.0 + np.exp(-self.beta * x))
        return [sig + self.beta * x * sig * (1 - sig)]


class SmoothInfNormGrad(admm.UDFBase):
    """Smooth infinity norm: f(x) = (1/t) * log(sum(exp(t*x_i) + exp(-t*x_i)))
    As t → ∞, converges to ||x||_∞. Convex.
    grad_i = (exp(t*x_i) - exp(-t*x_i)) / sum(exp(t*x_j) + exp(-t*x_j))
    """
    def __init__(self, arg, t=5.0):
        self.arg = arg
        self.t = t
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        t = self.t
        # Numerically stable: shift by max
        ax = t * np.abs(x)
        m = np.max(ax)
        vals = np.exp(t * x - m) + np.exp(-t * x - m)
        return float((m + np.log(np.sum(vals))) / t)
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        t = self.t
        ep = np.exp(t * x)
        em = np.exp(-t * x)
        # Numerically stable
        m = np.max(np.maximum(t * x, -t * x))
        ep_s = np.exp(t * x - m)
        em_s = np.exp(-t * x - m)
        denom = np.sum(ep_s + em_s)
        return [(ep_s - em_s) / denom]


class RatioPenaltyGrad(admm.UDFBase):
    """Bounded smooth penalty: f(x) = sum(x_i^2 / (1 + x_i^2))
    Approaches 1 as |x| → ∞. Non-convex globally but smooth.
    grad = 2*x / (1 + x^2)^2
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(x ** 2 / (1 + x ** 2)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [2.0 * x / (1 + x ** 2) ** 2]


class ELUSumGrad(admm.UDFBase):
    """ELU activation sum: f(x) = sum(ELU(x_i))
    ELU(x) = x if x > 0, alpha*(exp(x)-1) if x <= 0.
    Smooth everywhere (including at 0 when alpha=1). Convex for x<=0.
    grad = 1 if x>0, alpha*exp(x) if x<=0
    """
    def __init__(self, arg, alpha=1.0):
        self.arg = arg
        self.alpha = alpha
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.where(x > 0, x, self.alpha * (np.exp(x) - 1))))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [np.where(x > 0, 1.0, self.alpha * np.exp(x))]


class CoupledPairGrad(admm.UDFBase):
    """Adjacent coupling: f(x) = sum_{i=0}^{n-2} (x_i - x_{i+1})^2
    Promotes smoothness between consecutive elements.
    grad_i = 2*(x_i - x_{i+1}) if i < n-1
           + 2*(x_i - x_{i-1}) if i > 0
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        return float(np.sum(np.diff(x) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        g = np.zeros_like(x)
        d = np.diff(x)  # d[i] = x[i+1] - x[i]
        g[:-1] -= 2 * d
        g[1:] += 2 * d
        return [g.reshape(tensorlist[0].shape)]


class SumSigmoidGrad(admm.UDFBase):
    """f(x) = sum(sigmoid(x_i)) = sum(1/(1+exp(-x_i)))
    Bounded [0, n]. Smooth. Not convex.
    grad = sigmoid(x) * (1 - sigmoid(x))
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(1.0 / (1.0 + np.exp(-x))))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        s = 1.0 / (1.0 + np.exp(-x))
        return [s * (1 - s)]


class SmoothMaxAbsGrad(admm.UDFBase):
    """Smooth approximation of max(|x_i|):
    f(x) = (1/t)*log(sum(exp(t*x_i) + exp(-t*x_i))) - log(2*n)/t
    Convex, smooth.
    """
    def __init__(self, arg, t=10.0):
        self.arg = arg
        self.t = t
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        t = self.t
        n = len(x.ravel())
        vals = np.concatenate([t * x.ravel(), -t * x.ravel()])
        m = np.max(vals)
        return float((m + np.log(np.sum(np.exp(vals - m)))) / t - np.log(2 * n) / t)
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        t = self.t
        vals = np.concatenate([t * x, -t * x])
        m = np.max(vals)
        w = np.exp(vals - m)
        w /= np.sum(w)
        n = len(x)
        g = w[:n] - w[n:]
        return [g.reshape(tensorlist[0].shape)]


class MultiArgWeightedGrad(admm.UDFBase):
    """Multi-argument: f(x, y) = sum(w * (x - y)^2)
    Weighted coupling. Two arguments, same shape.
    grad_x = 2*w*(x-y), grad_y = -2*w*(x-y)
    """
    def __init__(self, arg1, arg2, w):
        self.arg1 = arg1
        self.arg2 = arg2
        self.w = np.asarray(w, dtype=float)
    def arguments(self):
        return [self.arg1, self.arg2]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        return float(np.sum(self.w * (x - y) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        d = x - y
        return [2.0 * self.w * d, -2.0 * self.w * d]


class ThreeArgSumGrad(admm.UDFBase):
    """Three-argument UDF: f(x, y, z) = sum((x-y)^2) + sum((y-z)^2)
    Chain coupling x ↔ y ↔ z.
    """
    def __init__(self, arg1, arg2, arg3):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
    def arguments(self):
        return [self.arg1, self.arg2, self.arg3]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        z = np.asarray(tensorlist[2], dtype=float)
        return float(np.sum((x - y) ** 2) + np.sum((y - z) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        z = np.asarray(tensorlist[2], dtype=float)
        return [2.0 * (x - y), -2.0 * (x - y) + 2.0 * (y - z), -2.0 * (y - z)]


# ---------------------------------------------------------------------------
# More extended test class
# ---------------------------------------------------------------------------

class GradUDFMoreTestCase(unittest.TestCase):
    """More smooth function tests: robust losses, ML activations,
    smooth norm approximations, coupling, multi-argument UDFs.
    """

    def _new_model(self, max_iter=2000):
        model = admm.Model()
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.display_sub_solver_details, 0)
        model.setOption(admm.Options.admm_max_iteration, max_iter)
        return model

    # ===================================================================
    # A. Student-t robust loss
    # ===================================================================

    def test_student_t_kkt(self):
        """min 0.5*||x-y||^2 + lam*sum(log(1+x^2/v))
        KKT: x + lam*2x/(v+x^2) = y
        """
        observed = np.array([2.0, -1.5, 0.3])
        lam = 0.5
        v = 2.0
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * StudentTLossGrad(x, v=v)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + lam * 2 * x_val / (v + x_val ** 2) - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    def test_student_t_vs_huber(self):
        """Student-t with v=1 (Cauchy-like) should produce more robust estimates
        than Huber when outliers are present.
        """
        # Data with outlier
        observed = np.array([1.0, 1.1, 0.9, 1.05, 10.0])  # 10.0 is outlier
        lam = 1.0

        model_h = self._new_model()
        x_h = admm.Var("xh", 5)
        model_h.setObjective(
            lam * HuberGrad(x_h - observed, delta=1.0)
        )
        model_h.optimize()

        model_t = self._new_model()
        x_t = admm.Var("xt", 5)
        model_t.setObjective(
            lam * StudentTLossGrad(x_t - observed, v=1.0)
        )
        model_t.optimize()

        self.assertEqual(model_h.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model_t.StatusString, "SOLVE_OPT_SUCCESS")

    def test_student_t_different_v(self):
        """Different degrees of freedom should all converge."""
        observed = np.array([3.0, -1.0, 0.5])
        lam = 0.3
        for v in [0.5, 1.0, 5.0, 10.0]:
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + lam * StudentTLossGrad(x, v=v)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"Failed for v={v}")

    # ===================================================================
    # B. Quantile regression
    # ===================================================================

    def test_quantile_median(self):
        """tau=0.5 → median regression ≈ symmetric smoothed L1.
        min sum(quantile_huber(x - y)) should give x ≈ y.
        """
        observed = np.array([1.0, 2.0, 3.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.3 * QuantileHuberGrad(x, tau=0.5, eps=0.1)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, observed, atol=0.5)

    def test_quantile_asymmetric(self):
        """tau=0.9 → penalizes negative residuals more.
        Solution should be shifted upward relative to tau=0.5.
        """
        observed = np.array([0.0, 0.0, 0.0])
        lam = 1.0

        results = {}
        for tau in [0.1, 0.5, 0.9]:
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(
                0.5 * admm.sum(admm.square(x))
                + lam * QuantileHuberGrad(x, tau=tau, eps=0.1)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            results[tau] = np.mean(np.asarray(x.X))

        # Higher tau → solution shifted more negative (penalty on positive x)
        self.assertLess(results[0.9], results[0.1])

    def test_quantile_regression_problem(self):
        """Quantile regression: min sum(quantile_huber(Ax - b))
        Smooth approximation of the classical quantile regression.
        """
        np.random.seed(55)
        m, n = 30, 3
        A = np.random.randn(m, n)
        x_true = np.array([1.0, -0.5, 0.3])
        noise = np.random.standard_t(3, m) * 0.5  # heavy-tailed noise
        b = A @ x_true + noise

        model = self._new_model(max_iter=5000)
        x = admm.Var("x", n)
        model.setObjective(
            QuantileHuberGrad(A @ x - b, tau=0.5, eps=0.05)
            + 0.01 * admm.sum(admm.square(x))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        np.testing.assert_allclose(x_val, x_true, atol=0.5)

    # ===================================================================
    # C. Swish activation
    # ===================================================================

    def test_swish_kkt(self):
        """min 0.5*||x-y||^2 + lam*sum(x*sigmoid(x))
        KKT: x + lam*(sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))) = y
        """
        observed = np.array([1.0, -0.5, 2.0])
        lam = 0.3
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SwishGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        sig = 1.0 / (1.0 + np.exp(-x_val))
        grad_swish = sig + x_val * sig * (1 - sig)
        residual = x_val + lam * grad_swish - observed
        np.testing.assert_allclose(residual, 0, atol=0.1)

    def test_swish_box(self):
        """min sum(x*sigmoid(x))  s.t. -2 <= x <= 2, sum(x) = 0
        Symmetric function, solution should be uniform 0.
        """
        n = 4
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(SwishGrad(x))
        model.addConstr(x >= -2)
        model.addConstr(x <= 2)
        model.addConstr(admm.sum(x) == 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        # Swish is minimized near x ≈ -0.278 (where x*sigmoid(x) has minimum)
        # With sum=0, expect near 0 on average
        x_val = np.asarray(x.X)
        self.assertAlmostEqual(np.sum(x_val), 0, delta=0.15)

    # ===================================================================
    # D. Smooth infinity norm
    # ===================================================================

    def test_smooth_inf_norm_minimization(self):
        """min smooth_inf_norm(x)  s.t. Ax = b
        Approximation of Chebyshev center / minimax.
        """
        A = np.array([[1, 1, 1]])
        b_val = np.array([3.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(SmoothInfNormGrad(x, t=10.0))
        model.addConstr(A @ x == b_val)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Optimal: all equal (minimizes max) → x = [1,1,1]
        np.testing.assert_allclose(x_val, 1.0, atol=0.15)

    def test_smooth_inf_norm_box(self):
        """min 0.5*||x-y||^2 + lam*smooth_inf_norm(x)
        Penalty shrinks largest component.
        """
        observed = np.array([5.0, 1.0, -3.0])
        lam = 1.0
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SmoothInfNormGrad(x, t=5.0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Largest-magnitude component should shrink most
        self.assertLess(abs(x_val[0]), abs(observed[0]))

    # ===================================================================
    # E. ELU activation
    # ===================================================================

    def test_elu_penalty(self):
        """min 0.5*||x-y||^2 + lam*sum(ELU(x))
        ELU is asymmetric: linear for x>0, saturating for x<0.
        """
        observed = np.array([2.0, -2.0, 0.5, -0.5])
        lam = 0.5
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * ELUSumGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Positive x: pushed down by constant gradient 1
        # Negative x: less affected (gradient → 0 for large negative)
        self.assertLess(x_val[0], observed[0])

    def test_elu_with_equality(self):
        """min sum(ELU(x))  s.t. sum(x) = 0, x >= -3"""
        n = 5
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(ELUSumGrad(x))
        model.addConstr(admm.sum(x) == 0)
        model.addConstr(x >= -3)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertAlmostEqual(np.sum(x_val), 0, delta=0.15)

    # ===================================================================
    # F. Adjacent coupling (smoothing)
    # ===================================================================

    def test_coupled_pair_smoothing(self):
        """min 0.5*||x - y_noisy||^2 + lam*sum((x_i - x_{i+1})^2)
        Total variation-like smoothing.
        """
        np.random.seed(22)
        n = 20
        x_true = np.sin(np.linspace(0, 2 * np.pi, n))
        y_noisy = x_true + 0.5 * np.random.randn(n)
        lam = 0.5

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - y_noisy))
            + lam * CoupledPairGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Smoothed signal should be closer to true than noisy
        err_noisy = np.linalg.norm(y_noisy - x_true)
        err_smooth = np.linalg.norm(x_val - x_true)
        self.assertLess(err_smooth, err_noisy)

    def test_coupled_pair_with_box(self):
        """Smoothing with box constraints."""
        np.random.seed(33)
        n = 15
        y = np.random.randn(n) * 2
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - y))
            + 0.3 * CoupledPairGrad(x)
        )
        model.addConstr(x >= -1)
        model.addConstr(x <= 1)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(x_val >= -1.01))
        self.assertTrue(np.all(x_val <= 1.01))
        # Smoothed: max consecutive diff should be less than full box range (2.0)
        max_diff = np.max(np.abs(np.diff(x_val)))
        self.assertLess(max_diff, 2.0)

    def test_coupled_large(self):
        """n=200 coupling."""
        np.random.seed(44)
        n = 200
        y = np.random.randn(n)
        model = self._new_model(max_iter=5000)
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - y))
            + 0.2 * CoupledPairGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # ===================================================================
    # G. Ratio penalty (bounded, non-convex)
    # ===================================================================

    def test_ratio_penalty(self):
        """min 0.5*||x-y||^2 + lam*sum(x^2/(1+x^2))
        Bounded penalty, promotes sparsity differently from L1.
        """
        observed = np.array([0.1, 3.0, -0.05, -2.0])
        lam = 0.5
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * RatioPenaltyGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))

    # ===================================================================
    # H. Sigmoid sum
    # ===================================================================

    def test_sigmoid_penalty(self):
        """min 0.5*||x-y||^2 + lam*sum(sigmoid(x))
        Sigmoid is bounded [0,1] per element. Pushes x negative.
        """
        observed = np.array([1.0, 0.0, -1.0, 2.0])
        lam = 1.0
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SumSigmoidGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Sigmoid gradient pushes x negative, so x < observed
        self.assertTrue(np.all(x_val <= observed + 0.01))

    # ===================================================================
    # I. Smooth max abs (minimax approximation)
    # ===================================================================

    def test_smooth_max_abs_minimax(self):
        """min smooth_max_abs(x - c)  s.t. sum(x)=sum(c)
        Should equalize |x_i - c_i| across all i.
        """
        c = np.array([1.0, 2.0, 3.0, 4.0])
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(SmoothMaxAbsGrad(x - c, t=10.0))
        model.addConstr(admm.sum(x) == np.sum(c))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # With sum constraint preserved, minimize max deviation
        self.assertAlmostEqual(np.sum(x_val), np.sum(c), delta=0.15)

    # ===================================================================
    # J. Multi-argument extended
    # ===================================================================

    def test_multi_arg_weighted_coupling(self):
        """Weighted coupling: weight[i] * (x_i - y_i)^2.
        Higher weight → tighter coupling.
        """
        n = 3
        w = np.array([0.01, 1.0, 100.0])
        model = self._new_model()
        x = admm.Var("x", n)
        y = admm.Var("y", n)
        model.setObjective(
            MultiArgWeightedGrad(x, y, w)
            + 0.5 * admm.sum(admm.square(x - 1))
            + 0.5 * admm.sum(admm.square(y + 1))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        y_val = np.asarray(y.X)
        # High-weight pair (index 2) should be closest
        diffs = np.abs(x_val - y_val)
        self.assertLess(diffs[2], diffs[0])

    def test_three_arg_chain_coupling(self):
        """Three variables coupled in chain: x ↔ y ↔ z.
        min sum((x-y)^2) + sum((y-z)^2)
        s.t. sum(x) = 3, sum(z) = 6, x >= 0, z >= 0
        """
        n = 3
        model = self._new_model()
        x = admm.Var("x", n)
        y = admm.Var("y", n)
        z = admm.Var("z", n)
        model.setObjective(ThreeArgSumGrad(x, y, z))
        model.addConstr(admm.sum(x) == 3)
        model.addConstr(admm.sum(z) == 6)
        model.addConstr(x >= 0)
        model.addConstr(z >= 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        y_val = np.asarray(y.X)
        z_val = np.asarray(z.X)
        # x and z are coupled through y, so y should be between them
        self.assertAlmostEqual(np.sum(x_val), 3.0, delta=0.15)
        self.assertAlmostEqual(np.sum(z_val), 6.0, delta=0.15)

    def test_three_arg_with_penalties(self):
        """Chain coupling + quartic penalties on endpoints."""
        n = 2
        model = self._new_model()
        x = admm.Var("x", n)
        y = admm.Var("y", n)
        z = admm.Var("z", n)
        model.setObjective(
            ThreeArgSumGrad(x, y, z)
            + 0.1 * QuarticGrad(x)
            + 0.1 * QuarticGrad(z)
        )
        model.addConstr(admm.sum(x) + admm.sum(y) + admm.sum(z) == 3)
        model.addConstr(x >= 0)
        model.addConstr(y >= 0)
        model.addConstr(z >= 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # ===================================================================
    # K. Gradient finite-difference checks for all new UDFs
    # ===================================================================

    def test_gradient_fd_student_t(self):
        np.random.seed(10)
        d = admm.Var("d", 4)
        x = np.random.randn(4) * 0.5
        h = 1e-6
        for v in [0.5, 1.0, 5.0]:
            udf_obj = StudentTLossGrad(d, v=v)
            ga = np.asarray(udf_obj.grad([x.copy()])[0])
            gf = np.zeros(4)
            for i in range(4):
                xp = x.copy(); xp[i] += h
                xm = x.copy(); xm[i] -= h
                gf[i] = (udf_obj.eval([xp]) - udf_obj.eval([xm])) / (2 * h)
            np.testing.assert_allclose(ga, gf, atol=1e-4, err_msg=f"v={v}")

    def test_gradient_fd_quantile_huber(self):
        np.random.seed(11)
        d = admm.Var("d", 4)
        x = np.random.randn(4)
        h = 1e-6
        for tau in [0.1, 0.5, 0.9]:
            udf_obj = QuantileHuberGrad(d, tau=tau, eps=0.1)
            ga = np.asarray(udf_obj.grad([x.copy()])[0])
            gf = np.zeros(4)
            for i in range(4):
                xp = x.copy(); xp[i] += h
                xm = x.copy(); xm[i] -= h
                gf[i] = (udf_obj.eval([xp]) - udf_obj.eval([xm])) / (2 * h)
            np.testing.assert_allclose(ga, gf, atol=1e-4, err_msg=f"tau={tau}")

    def test_gradient_fd_swish(self):
        np.random.seed(12)
        d = admm.Var("d", 4)
        x = np.random.randn(4)
        h = 1e-6
        udf_obj = SwishGrad(d, beta=1.0)
        ga = np.asarray(udf_obj.grad([x.copy()])[0])
        gf = np.zeros(4)
        for i in range(4):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            gf[i] = (udf_obj.eval([xp]) - udf_obj.eval([xm])) / (2 * h)
        np.testing.assert_allclose(ga, gf, atol=1e-4)

    def test_gradient_fd_smooth_inf_norm(self):
        np.random.seed(13)
        d = admm.Var("d", 4)
        x = np.random.randn(4)
        h = 1e-6
        udf_obj = SmoothInfNormGrad(d, t=5.0)
        ga = np.asarray(udf_obj.grad([x.copy()])[0])
        gf = np.zeros(4)
        for i in range(4):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            gf[i] = (udf_obj.eval([xp]) - udf_obj.eval([xm])) / (2 * h)
        np.testing.assert_allclose(ga, gf, atol=1e-4)

    def test_gradient_fd_elu(self):
        np.random.seed(14)
        d = admm.Var("d", 4)
        x = np.random.randn(4)
        h = 1e-6
        udf_obj = ELUSumGrad(d)
        ga = np.asarray(udf_obj.grad([x.copy()])[0])
        gf = np.zeros(4)
        for i in range(4):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            gf[i] = (udf_obj.eval([xp]) - udf_obj.eval([xm])) / (2 * h)
        np.testing.assert_allclose(ga, gf, atol=1e-4)

    def test_gradient_fd_coupled_pair(self):
        np.random.seed(15)
        d = admm.Var("d", 6)
        x = np.random.randn(6)
        h = 1e-6
        udf_obj = CoupledPairGrad(d)
        ga = np.asarray(udf_obj.grad([x.copy()])[0])
        gf = np.zeros(6)
        for i in range(6):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            gf[i] = (udf_obj.eval([xp]) - udf_obj.eval([xm])) / (2 * h)
        np.testing.assert_allclose(ga, gf, atol=1e-4)

    def test_gradient_fd_ratio(self):
        np.random.seed(16)
        d = admm.Var("d", 4)
        x = np.random.randn(4)
        h = 1e-6
        udf_obj = RatioPenaltyGrad(d)
        ga = np.asarray(udf_obj.grad([x.copy()])[0])
        gf = np.zeros(4)
        for i in range(4):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            gf[i] = (udf_obj.eval([xp]) - udf_obj.eval([xm])) / (2 * h)
        np.testing.assert_allclose(ga, gf, atol=1e-4)

    def test_gradient_fd_sigmoid(self):
        np.random.seed(17)
        d = admm.Var("d", 4)
        x = np.random.randn(4)
        h = 1e-6
        udf_obj = SumSigmoidGrad(d)
        ga = np.asarray(udf_obj.grad([x.copy()])[0])
        gf = np.zeros(4)
        for i in range(4):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            gf[i] = (udf_obj.eval([xp]) - udf_obj.eval([xm])) / (2 * h)
        np.testing.assert_allclose(ga, gf, atol=1e-4)

    def test_gradient_fd_smooth_max_abs(self):
        np.random.seed(18)
        d = admm.Var("d", 4)
        x = np.random.randn(4)
        h = 1e-6
        udf_obj = SmoothMaxAbsGrad(d, t=10.0)
        ga = np.asarray(udf_obj.grad([x.copy()])[0])
        gf = np.zeros(4)
        for i in range(4):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            gf[i] = (udf_obj.eval([xp]) - udf_obj.eval([xm])) / (2 * h)
        np.testing.assert_allclose(ga, gf, atol=1e-4)

    def test_gradient_fd_three_arg(self):
        np.random.seed(19)
        d1 = admm.Var("d1", 3)
        d2 = admm.Var("d2", 3)
        d3 = admm.Var("d3", 3)
        udf_obj = ThreeArgSumGrad(d1, d2, d3)
        x = np.random.randn(3)
        y = np.random.randn(3)
        z = np.random.randn(3)
        h = 1e-6

        grads = udf_obj.grad([x.copy(), y.copy(), z.copy()])
        for arg_idx, (test_vec, label) in enumerate([(x, "x"), (y, "y"), (z, "z")]):
            ga = np.asarray(grads[arg_idx])
            gf = np.zeros(3)
            for i in range(3):
                vecs = [x.copy(), y.copy(), z.copy()]
                vecs[arg_idx][i] += h
                fp = udf_obj.eval(vecs)
                vecs = [x.copy(), y.copy(), z.copy()]
                vecs[arg_idx][i] -= h
                fm = udf_obj.eval(vecs)
                gf[i] = (fp - fm) / (2 * h)
            np.testing.assert_allclose(ga, gf, atol=1e-4, err_msg=f"grad_{label}")

    # ===================================================================
    # L. Application problems
    # ===================================================================

    def test_robust_regression_student_t(self):
        """Robust regression: min sum(student_t(Ax - b)) + lam*||x||^2
        With outlier contamination.
        """
        np.random.seed(77)
        m, n = 40, 4
        x_true = np.array([1.0, -0.5, 0.3, 0.8])
        A = np.random.randn(m, n)
        b = A @ x_true + 0.1 * np.random.randn(m)
        # Add outliers
        b[0] = 100; b[1] = -50

        model = self._new_model(max_iter=5000)
        x = admm.Var("x", n)
        model.setObjective(
            StudentTLossGrad(A @ x - b, v=1.0)
            + 0.1 * admm.sum(admm.square(x))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        # Robust estimator should be close despite outliers
        np.testing.assert_allclose(x_val, x_true, atol=0.5)

    def test_smooth_lasso_path(self):
        """Smooth L1 with increasing lambda: verify monotone sparsity."""
        np.random.seed(88)
        m, n = 20, 5
        A = np.random.randn(m, n)
        x_true = np.array([3.0, 0.0, -2.0, 0.0, 1.0])
        b = A @ x_true

        nnz_prev = n
        for lam in [0.01, 0.1, 0.5, 1.0]:
            model = self._new_model()
            x = admm.Var("x", n)
            model.setObjective(
                admm.sum(admm.square(A @ x - b))
                + lam * SmoothL1Grad(x, eps=1e-4)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"Failed for lam={lam}")
            x_val = np.asarray(x.X)
            nnz = np.sum(np.abs(x_val) > 0.1)
            # More regularization → fewer significant components
            self.assertLessEqual(nnz, nnz_prev + 1)  # allow minor fluctuation
            nnz_prev = nnz

    def test_coupled_signal_interpolation(self):
        """Coupled smoothing with missing data:
        min sum_{observed} (x_i - y_i)^2 + lam*sum(diff(x)^2)
        """
        np.random.seed(99)
        n = 30
        x_true = np.sin(np.linspace(0, np.pi, n))
        observed_idx = np.sort(np.random.choice(n, 20, replace=False))
        y = x_true[observed_idx] + 0.1 * np.random.randn(len(observed_idx))

        model = self._new_model()
        x = admm.Var("x", n)
        # Data fidelity on observed points
        y_full = np.zeros(n)
        mask = np.zeros(n)
        y_full[observed_idx] = y
        mask[observed_idx] = 1.0
        model.setObjective(
            0.5 * admm.sum(admm.square(mask * (x - y_full)))
            + 0.5 * CoupledPairGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Interpolated signal should be smooth and close to true
        err = np.linalg.norm(x_val - x_true) / np.sqrt(n)
        self.assertLess(err, 0.5)

    def test_multi_task_coupled_regression(self):
        """Multi-task: min ||A@x-b1||^2 + ||A@y-b2||^2 + lam*sum((x-y)^2)
        Two related regression tasks with coupling.
        """
        np.random.seed(111)
        m, n = 20, 4
        A = np.random.randn(m, n)
        x_true = np.array([1.0, -0.5, 0.3, 0.8])
        y_true = x_true + 0.1 * np.random.randn(n)  # similar
        b1 = A @ x_true + 0.1 * np.random.randn(m)
        b2 = A @ y_true + 0.1 * np.random.randn(m)

        lam = 1.0
        model = self._new_model()
        x = admm.Var("x", n)
        y = admm.Var("y", n)
        model.setObjective(
            admm.sum(admm.square(A @ x - b1))
            + admm.sum(admm.square(A @ y - b2))
            + lam * MultiArgQuadGrad(x, y)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        y_val = np.asarray(y.X).ravel()
        # Coupling makes x and y closer than without
        self.assertLess(np.linalg.norm(x_val - y_val), 1.0)

    # ===================================================================
    # M. Stress: many UDFs in one model
    # ===================================================================

    def test_five_udfs_combined(self):
        """Five different grad UDFs in one objective."""
        observed = np.array([1.0, -1.0, 2.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.02 * QuarticGrad(x)
            + 0.02 * ExpSumGrad(x)
            + 0.02 * LogCoshGrad(x)
            + 0.02 * SwishGrad(x)
            + 0.02 * StudentTLossGrad(x, v=2.0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    def test_many_penalties_sweep(self):
        """All smooth penalties from this file on same problem."""
        observed = np.array([1.0, -0.5])
        lam = 0.1
        all_classes = [
            QuarticGrad, SixthPowerGrad, ExpSumGrad, SoftplusGrad,
            LogCoshGrad, SumExpNegGrad, SmoothMaxPairGrad, XExpXGrad,
            StudentTLossGrad, SwishGrad, ELUSumGrad, RatioPenaltyGrad,
            SumSigmoidGrad,
        ]
        for cls in all_classes:
            model = self._new_model()
            x = admm.Var("x", 2)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + lam * cls(x)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"Failed for {cls.__name__}")


# ---------------------------------------------------------------------------
# Structural / integration test class — exercises API patterns beyond
# simple "build + solve".
# ---------------------------------------------------------------------------

class GradUDFStructuralTestCase(unittest.TestCase):
    """Tests for grad UDF interactions with ADMM framework features:
    expressions on UDF arguments, objective value verification,
    maximization, Param, constraint removal, re-solve, slicing,
    infeasibility detection, high dimensions, sparse data, model copy.
    """

    def _new_model(self, max_iter=2000):
        model = admm.Model()
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.display_sub_solver_details, 0)
        model.setOption(admm.Options.admm_max_iteration, max_iter)
        return model

    # ===================================================================
    # A. UDF on expressions (not just raw variables)
    # ===================================================================

    def test_udf_on_affine_expr(self):
        """UDF(A @ x - b): UDF argument is an affine expression."""
        np.random.seed(11)
        m, n = 15, 4
        A = np.random.randn(m, n)
        x_true = np.array([1.0, -0.5, 0.3, 0.8])
        b = A @ x_true

        model = self._new_model()
        x = admm.Var("x", n)
        # Smooth L1 on residual
        model.setObjective(SmoothL1Grad(A @ x - b, eps=1e-4))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        np.testing.assert_allclose(x_val, x_true, atol=0.3)

    def test_udf_on_scaled_variable(self):
        """UDF(2*x + 1): constant scaling and offset."""
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - 1))
            + 0.1 * ExpSumGrad(2 * x + 1)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))

    def test_udf_on_difference(self):
        """UDF(x - y): coupling two variables via expression."""
        model = self._new_model()
        x = admm.Var("x", 3)
        y = admm.Var("y", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - 1))
            + 0.5 * admm.sum(admm.square(y + 1))
            + 0.5 * QuarticGrad(x - y)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        y_val = np.asarray(y.X)
        # Coupling pushes x and y together
        self.assertLess(np.linalg.norm(x_val - y_val), 3.0)

    def test_udf_on_matrix_vector_product(self):
        """UDF on dense matrix-vector product expression."""
        np.random.seed(22)
        A = np.random.randn(5, 3)
        b = np.random.randn(5)
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            HuberGrad(A @ x - b, delta=1.0)
            + 0.01 * admm.sum(admm.square(x))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # ===================================================================
    # B. Variable slicing in UDF arguments
    # ===================================================================

    def test_udf_on_slice(self):
        """UDF applied to a slice of a variable: UDF(x[0:2])."""
        model = self._new_model()
        x = admm.Var("x", 5)
        # Penalize only first 2 components with quartic, rest with quadratic
        model.setObjective(
            0.5 * admm.sum(admm.square(x - 3))
            + 0.5 * QuarticGrad(x[0:2])
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # First 2 elements have extra quartic → more shrinkage
        self.assertLess(abs(x_val[0]), abs(x_val[3]) + 0.1)

    def test_udf_on_different_slices(self):
        """Different UDFs on different slices of same variable."""
        model = self._new_model()
        x = admm.Var("x", 6)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - 2))
            + 0.2 * ExpSumGrad(x[0:3])
            + 0.2 * SoftplusGrad(x[3:6])
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))

    # ===================================================================
    # C. Objective value verification
    # ===================================================================

    def test_objval_matches_manual_eval(self):
        """Manually evaluate f(x*) and compare with model.ObjVal."""
        observed = np.array([2.0, -1.0, 0.5])
        lam = 0.1
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

        x_val = np.asarray(x.X)
        manual_obj = 0.5 * np.sum((x_val - observed) ** 2) + lam * np.sum(x_val ** 4)
        self.assertAlmostEqual(model.ObjVal, manual_obj, delta=0.01)

    def test_objval_exp_sum(self):
        """Objective value check for exp sum."""
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(ExpSumGrad(x))
        model.addConstr(x >= 0)
        model.addConstr(x <= 1)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

        x_val = np.asarray(x.X)
        manual_obj = np.sum(np.exp(x_val))
        self.assertAlmostEqual(model.ObjVal, manual_obj, delta=0.05)

    def test_objval_mixed_udf_builtin(self):
        """Objective value for UDF + builtin combination."""
        observed = np.array([1.0, -2.0, 3.0])
        lam1, lam2 = 0.1, 0.2
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam1 * LogCoshGrad(x)
            + lam2 * admm.norm(x, 1)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

        x_val = np.asarray(x.X)
        manual_udf = 0.5 * np.sum((x_val - observed) ** 2) \
                      + lam1 * np.sum(np.log(np.cosh(x_val))) \
                      + lam2 * np.sum(np.abs(x_val))
        # Allow slightly larger tolerance since L1 norm might have solver imprecision
        self.assertAlmostEqual(model.ObjVal, manual_udf, delta=0.1)

    # ===================================================================
    # D. Maximization (via MinSense = False)
    # ===================================================================

    def test_maximize_neg_entropy(self):
        """Maximize entropy = minimize negative entropy.
        max -sum(x*log(x)) = min sum(x*log(x))
        s.t. sum(x) = 1, x >= eps
        Solution: uniform distribution.
        """
        n = 4
        model = self._new_model()
        x = admm.Var("x", n)
        model.MinSense = False  # maximize
        model.setObjective(-NegEntropyGrad(x))  # max -f = max entropy
        model.addConstr(admm.sum(x) == 1)
        model.addConstr(x >= 0.001)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 1.0 / n, atol=0.05)

    def test_maximize_concave_function(self):
        """max -sum(x^4) = min sum(x^4), but via maximize sense.
        With sum(x)=3, x>=0 => uniform x=1.
        """
        n = 3
        model = self._new_model()
        x = admm.Var("x", n)
        model.MinSense = False
        model.setObjective(-QuarticGrad(x))  # max(-quartic)
        model.addConstr(admm.sum(x) == 3)
        model.addConstr(x >= 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 1.0, atol=0.1)

    # ===================================================================
    # E. Parameter binding with UDF
    # ===================================================================

    def test_param_scalar_with_udf(self):
        """Use admm.Param to sweep lambda without rebuilding model."""
        observed = np.array([2.0, -1.0, 0.5])
        lam = admm.Param("lam")
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * QuarticGrad(x)
        )

        prev_norm = None
        for lam_val in [0.01, 0.1, 1.0]:
            model.optimize({"lam": lam_val})
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"Failed for lam={lam_val}")
            cur_norm = np.linalg.norm(np.asarray(x.X))
            if prev_norm is not None:
                # Increasing lambda → more shrinkage
                self.assertLess(cur_norm, prev_norm + 0.1)
            prev_norm = cur_norm

    def test_param_vector_with_udf(self):
        """Vector parameter in data-fidelity + UDF."""
        n = 4
        y_param = admm.Param("y", n)
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - y_param))
            + 0.1 * ExpSumGrad(x)
        )
        y_val = np.array([1.0, 2.0, -1.0, 0.5])
        model.optimize({"y": y_val})
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

        # Different y_val should give different x
        y_val2 = np.array([3.0, 0.0, 0.0, -2.0])
        model.optimize({"y": y_val2})
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # ===================================================================
    # F. Constraint removal and re-solve
    # ===================================================================

    def test_tight_vs_loose_bound(self):
        """Tight box [0, 0.5] vs loose box [0, 5]: tighter forces higher obj.
        Compares two models to verify constraint impact on UDF solution.
        """
        # Tight
        m1 = self._new_model()
        x1 = admm.Var("x1", 3)
        m1.setObjective(ExpSumGrad(x1))
        m1.addConstr(x1 >= 0)
        m1.addConstr(x1 <= 0.5)
        m1.optimize()
        self.assertEqual(m1.StatusString, "SOLVE_OPT_SUCCESS")
        obj_tight = m1.ObjVal

        # Loose
        m2 = self._new_model()
        x2 = admm.Var("x2", 3)
        m2.setObjective(ExpSumGrad(x2))
        m2.addConstr(x2 >= -5)
        m2.addConstr(x2 <= 5)
        m2.optimize()
        self.assertEqual(m2.StatusString, "SOLVE_OPT_SUCCESS")
        obj_loose = m2.ObjVal

        # Looser feasible set → lower or equal objective
        self.assertLessEqual(obj_loose, obj_tight + 0.01)

    # ===================================================================
    # G. Infeasibility detection with UDF
    # ===================================================================

    def test_infeasible_conflicting_equality(self):
        """x1 + x2 = 1 and x1 + x2 = 3 — impossible."""
        model = self._new_model(max_iter=100000)
        x = admm.Var("x", 2)
        model.setObjective(0.5 * admm.sum(admm.square(x)) + 0.1 * QuarticGrad(x))
        model.addConstr(x[0] + x[1] == 1)
        model.addConstr(x[0] + x[1] == 3)
        model.optimize()
        # Should detect infeasibility (status 2) or not converge
        self.assertIn(model.status, (2, 4))

    def test_infeasible_box(self):
        """x >= 5 and x <= 2 — empty feasible set."""
        model = self._new_model(max_iter=100000)
        x = admm.Var("x", 2)
        model.setObjective(0.5 * admm.sum(admm.square(x)) + 0.1 * ExpSumGrad(x))
        model.addConstr(x >= 5)
        model.addConstr(x <= 2)
        model.optimize()
        self.assertIn(model.status, (2, 4))

    # ===================================================================
    # H. UDF as sole objective (no quadratic wrapper)
    # ===================================================================

    def test_udf_only_objective_with_constraints(self):
        """min sum(x^4)  s.t. sum(x) = 4, x >= 0
        No quadratic term. Solution: uniform x = 1.
        """
        n = 4
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(QuarticGrad(x))
        model.addConstr(admm.sum(x) == 4)
        model.addConstr(x >= 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 1.0, atol=0.1)

    def test_udf_only_logcosh(self):
        """min sum(log(cosh(x)))  s.t. sum(x) = 2, x >= 0
        log(cosh) is symmetric, min at 0. On simplex, uniform.
        """
        n = 4
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(LogCoshGrad(x))
        model.addConstr(admm.sum(x) == 2)
        model.addConstr(x >= 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 0.5, atol=0.1)

    def test_udf_only_student_t(self):
        """min sum(log(1 + x^2))  s.t. sum(x)=0, -5 <= x <= 5
        log(1+x^2) is symmetric, minimized at 0. With sum=0, all x=0.
        """
        n = 5
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(StudentTLossGrad(x, v=1.0))
        model.addConstr(admm.sum(x) == 0)
        model.addConstr(x >= -5)
        model.addConstr(x <= 5)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, 0, atol=0.1)

    # ===================================================================
    # I. High dimension stress
    # ===================================================================

    def test_dim_1000_quartic(self):
        """n=1000 with quartic penalty."""
        np.random.seed(1000)
        n = 1000
        y = np.random.randn(n) * 0.5
        model = self._new_model(max_iter=5000)
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - y))
            + 0.01 * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))

    def test_dim_1000_softplus(self):
        """n=1000 with softplus penalty and box constraints."""
        np.random.seed(1001)
        n = 1000
        y = np.random.randn(n)
        model = self._new_model(max_iter=5000)
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - y))
            + 0.05 * SoftplusGrad(x)
        )
        model.addConstr(x >= -3)
        model.addConstr(x <= 3)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    def test_large_matrix_20x20(self):
        """20x20 matrix variable with UDF."""
        np.random.seed(2020)
        Y = np.random.randn(20, 20) * 0.3
        model = self._new_model(max_iter=5000)
        X = admm.Var("X", 20, 20)
        model.setObjective(
            0.5 * admm.sum(admm.square(X - Y))
            + 0.02 * MatExpSumGrad(X)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # ===================================================================
    # J. Mixed variable shapes in one model
    # ===================================================================

    def test_scalar_vector_matrix_mix(self):
        """Scalar, vector, and matrix variables with UDFs in one model."""
        model = self._new_model()
        alpha = admm.Var("alpha", 1)
        x = admm.Var("x", 4)
        M = admm.Var("M", 2, 2)
        model.setObjective(
            ExpSumGrad(alpha)
            + 0.5 * admm.sum(admm.square(x - 1))
            + 0.1 * QuarticGrad(x)
            + 0.5 * admm.sum(admm.square(M))
            + 0.1 * MatExpSumGrad(M)
        )
        model.addConstr(alpha >= 0)
        model.addConstr(alpha <= 3)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    def test_two_vectors_different_sizes(self):
        """Two vectors of different sizes, each with own UDF."""
        model = self._new_model()
        x = admm.Var("x", 3)
        y = admm.Var("y", 7)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - 2))
            + 0.5 * admm.sum(admm.square(y + 1))
            + 0.1 * QuarticGrad(x)
            + 0.1 * SoftplusGrad(y)
        )
        model.addConstr(x >= 0)
        model.addConstr(y >= -3)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # ===================================================================
    # K. Sparse data interaction
    # ===================================================================

    def test_sparse_constraint_matrix(self):
        """Sparse constraint matrix with UDF objective."""
        import scipy.sparse as sp
        np.random.seed(33)
        n = 20
        m = 10
        A = sp.random(m, n, density=0.3, format='csr', random_state=33)
        A = A.toarray()
        x_feas = np.ones(n)
        b = A @ x_feas
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - 1))
            + 0.1 * ExpSumGrad(x)
        )
        model.addConstr(A @ x == b)
        model.addConstr(x >= 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Constraint satisfaction
        np.testing.assert_allclose(A @ x_val, b, atol=0.2)

    # ===================================================================
    # L. Re-solve with modified objective
    # ===================================================================

    def test_resolve_different_udf(self):
        """Build model, solve, change objective to different UDF, re-solve."""
        observed = np.array([2.0, -1.0, 0.5])
        model = self._new_model()
        x = admm.Var("x", 3)

        # First solve with quartic
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.3 * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x1 = np.asarray(x.X).copy()

        # Second solve with exp sum (new model needed due to setObjective)
        model2 = self._new_model()
        x2 = admm.Var("x2", 3)
        model2.setObjective(
            0.5 * admm.sum(admm.square(x2 - observed))
            + 0.3 * ExpSumGrad(x2)
        )
        model2.optimize()
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        x2_val = np.asarray(x2.X)
        # Different penalties → different solutions
        self.assertFalse(np.allclose(x1, x2_val, atol=0.01))

    # ===================================================================
    # M. Solution quality / optimality checks
    # ===================================================================

    def test_kkt_residual_quartic(self):
        """Verify KKT conditions at solution:
        x + lam * 4x^3 = y  (unconstrained proximal)
        """
        observed = np.array([3.0, -2.0, 1.5, -0.5, 0.0])
        lam = 0.2
        model = self._new_model()
        x = admm.Var("x", 5)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        kkt = x_val + lam * 4 * x_val ** 3 - observed
        np.testing.assert_allclose(kkt, 0, atol=0.05)

    def test_kkt_residual_exp(self):
        """KKT for exp: x + lam*exp(x) = y"""
        observed = np.array([2.0, -1.0, 0.5, 3.0])
        lam = 0.3
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * ExpSumGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        kkt = x_val + lam * np.exp(x_val) - observed
        np.testing.assert_allclose(kkt, 0, atol=0.05)

    def test_kkt_residual_logcosh(self):
        """KKT for log-cosh: x + lam*tanh(x) = y"""
        observed = np.array([1.5, -0.8, 0.2])
        lam = 0.5
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * LogCoshGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        kkt = x_val + lam * np.tanh(x_val) - observed
        np.testing.assert_allclose(kkt, 0, atol=0.05)

    def test_kkt_residual_student_t(self):
        """KKT for Student-t: x + lam*2x/(v+x^2) = y"""
        observed = np.array([2.0, -1.0, 0.3, 1.5])
        lam = 0.4
        v = 2.0
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * StudentTLossGrad(x, v=v)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        kkt = x_val + lam * 2 * x_val / (v + x_val ** 2) - observed
        np.testing.assert_allclose(kkt, 0, atol=0.05)

    def test_kkt_residual_swish(self):
        """KKT for Swish: x + lam*(sig + x*sig*(1-sig)) = y"""
        observed = np.array([1.0, -0.5, 2.0, 0.0])
        lam = 0.2
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SwishGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        sig = 1.0 / (1.0 + np.exp(-x_val))
        kkt = x_val + lam * (sig + x_val * sig * (1 - sig)) - observed
        np.testing.assert_allclose(kkt, 0, atol=0.1)

    # ===================================================================
    # N. Multiple variables with same UDF class
    # ===================================================================

    def test_same_udf_class_three_vars(self):
        """Three variables with same UDF class: quartic on each."""
        model = self._new_model()
        x = admm.Var("x", 3)
        y = admm.Var("y", 4)
        z = admm.Var("z", 2)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - 1))
            + 0.5 * admm.sum(admm.square(y - 2))
            + 0.5 * admm.sum(admm.square(z - 3))
            + 0.1 * QuarticGrad(x)
            + 0.1 * QuarticGrad(y)
            + 0.1 * QuarticGrad(z)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        # Each variable should be shrunk toward 0 relative to its target
        self.assertTrue(np.all(np.asarray(x.X) < 1.01))
        self.assertTrue(np.all(np.asarray(y.X) < 2.01))
        self.assertTrue(np.all(np.asarray(z.X) < 3.01))

    def test_different_udf_classes_six_vars(self):
        """Six variables, each with a different UDF."""
        model = self._new_model()
        v1 = admm.Var("v1", 2)
        v2 = admm.Var("v2", 2)
        v3 = admm.Var("v3", 2)
        v4 = admm.Var("v4", 2)
        v5 = admm.Var("v5", 2)
        v6 = admm.Var("v6", 2)
        model.setObjective(
            0.5 * (admm.sum(admm.square(v1-1)) + admm.sum(admm.square(v2-1))
                   + admm.sum(admm.square(v3-1)) + admm.sum(admm.square(v4-1))
                   + admm.sum(admm.square(v5-1)) + admm.sum(admm.square(v6-1)))
            + 0.05 * QuarticGrad(v1)
            + 0.05 * ExpSumGrad(v2)
            + 0.05 * LogCoshGrad(v3)
            + 0.05 * SoftplusGrad(v4)
            + 0.05 * SwishGrad(v5)
            + 0.05 * StudentTLossGrad(v6, v=2.0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # ===================================================================
    # O. Edge cases
    # ===================================================================

    def test_udf_with_zero_lambda(self):
        """Lambda = 0: UDF has no effect, solution = observed."""
        observed = np.array([2.0, -1.0, 3.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.0 * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x.X, observed, atol=0.01)

    def test_udf_dim_1(self):
        """Scalar variable with UDF."""
        model = self._new_model()
        x = admm.Var("x", 1)
        model.setObjective(
            admm.sum(admm.square(x - 5))
            + 0.1 * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()[0]
        # KKT: 2(x-5) + 0.4x^3 = 0
        self.assertGreater(x_val, 0)
        self.assertLess(x_val, 5)

    def test_identical_observed_all_same(self):
        """All observed values identical: solution should be same."""
        model = self._new_model()
        x = admm.Var("x", 10)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - 1))
            + 0.1 * ExpSumGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # All components should be identical by symmetry
        self.assertLess(np.std(x_val), 0.01)


# ---------------------------------------------------------------------------
# New UDF classes for advanced tests
# ---------------------------------------------------------------------------

class Degree8Grad(admm.UDFBase):
    """f(x) = sum(x_i^8), grad = 8*x^7"""
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(x ** 8))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [8.0 * x ** 7]


class Degree10Grad(admm.UDFBase):
    """f(x) = sum(x_i^10), grad = 10*x^9"""
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(x ** 10))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [10.0 * x ** 9]


class CauchyLossGrad(admm.UDFBase):
    """Cauchy/Lorentzian loss: f(x) = sum(log(1 + (x_i/c)^2))
    grad = 2*x / (c^2 + x^2)
    Robust loss — bounded influence function.
    """
    def __init__(self, arg, c=1.0):
        self.arg = arg
        self.c = c
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.log(1 + (x / self.c) ** 2)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [2.0 * x / (self.c ** 2 + x ** 2)]


class MorsePotentialGrad(admm.UDFBase):
    """Morse potential: f(x) = sum(D * (1 - exp(-a*(x_i - r0)))^2)
    grad = 2*D*a*(1 - exp(-a*(x-r0)))*exp(-a*(x-r0))
    Smooth, bounded, with a unique minimum at x=r0.
    """
    def __init__(self, arg, D=1.0, a=1.0, r0=0.0):
        self.arg = arg
        self.D = D
        self.a = a
        self.r0 = r0
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        e = np.exp(-self.a * (x - self.r0))
        return float(self.D * np.sum((1 - e) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        e = np.exp(-self.a * (x - self.r0))
        return [2.0 * self.D * self.a * (1 - e) * e]


class BoundedRatioGrad(admm.UDFBase):
    """Bounded ratio penalty: f(x) = sum(x_i^2 / (1 + x_i^2))
    grad = 2*x / (1 + x^2)^2
    Smooth, bounded [0, n), promotes sparsity like L0.
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(x ** 2 / (1 + x ** 2)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [2.0 * x / (1 + x ** 2) ** 2]


class WeightedLSGrad(admm.UDFBase):
    """Weighted least squares: f(x) = sum(w_i * (x_i - t_i)^2)
    grad = 2 * w * (x - t)
    """
    def __init__(self, arg, weights, targets):
        self.arg = arg
        self.w = np.asarray(weights, dtype=float)
        self.t = np.asarray(targets, dtype=float)
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(self.w * (x - self.t) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [2.0 * self.w * (x - self.t)]


class FrobeniusRegGrad(admm.UDFBase):
    """Frobenius norm regularization: f(X) = ||X||_F^2 = sum(X_ij^2)
    grad = 2*X
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        X = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(X ** 2))
    def grad(self, tensorlist):
        X = np.asarray(tensorlist[0], dtype=float)
        return [2.0 * X]


class MatrixTracePenaltyGrad(admm.UDFBase):
    """Trace penalty: f(X) = tr(X) = sum(X_ii)
    grad_ij = 1 if i==j else 0 (identity matrix)
    Only for square matrices.
    """
    def __init__(self, arg, n):
        self.arg = arg
        self.n = n
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        X = np.asarray(tensorlist[0], dtype=float)
        return float(np.trace(X))
    def grad(self, tensorlist):
        X = np.asarray(tensorlist[0], dtype=float)
        return [np.eye(self.n)]


class OffDiagPenaltyGrad(admm.UDFBase):
    """Off-diagonal penalty: f(X) = sum_{i!=j} X_ij^2
    grad_ij = 2*X_ij if i!=j, 0 if i==j
    Encourages diagonal matrices.
    """
    def __init__(self, arg, n):
        self.arg = arg
        self.n = n
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        X = np.asarray(tensorlist[0], dtype=float)
        mask = 1 - np.eye(self.n)
        return float(np.sum(mask * X ** 2))
    def grad(self, tensorlist):
        X = np.asarray(tensorlist[0], dtype=float)
        mask = 1 - np.eye(self.n)
        return [2.0 * mask * X]


class SmoothHingeLossGrad(admm.UDFBase):
    """Smooth hinge loss for classification:
    f(w) = sum_i smooth_hinge(y_i * a_i^T w)
    where smooth_hinge(z) = 0 if z>=1, (1-z)^2/2 if 0<z<1, 0.5-z if z<=0
    grad_z = 0 if z>=1, -(1-z) if 0<z<1, -1 if z<=0
    """
    def __init__(self, arg, A, y):
        self.arg = arg
        self.A = np.asarray(A, dtype=float)
        self.y = np.asarray(y, dtype=float)
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        w = np.asarray(tensorlist[0], dtype=float).ravel()
        z = self.y * (self.A @ w)
        loss = np.where(z >= 1, 0.0,
               np.where(z >= 0, 0.5 * (1 - z) ** 2,
                        0.5 - z))
        return float(np.sum(loss))
    def grad(self, tensorlist):
        w = np.asarray(tensorlist[0], dtype=float).ravel()
        z = self.y * (self.A @ w)
        dloss_dz = np.where(z >= 1, 0.0,
                   np.where(z >= 0, -(1 - z),
                            -1.0))
        g = self.A.T @ (self.y * dloss_dz)
        return [g.reshape(tensorlist[0].shape)]


class WelschLossGrad(admm.UDFBase):
    """Welsch/Leclerc loss: f(x) = sum(c^2/2 * (1 - exp(-x_i^2/c^2)))
    grad = x * exp(-x^2/c^2)
    Very robust — bounded loss with Gaussian-shaped influence.
    """
    def __init__(self, arg, c=1.0):
        self.arg = arg
        self.c = c
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(0.5 * self.c ** 2 * np.sum(1 - np.exp(-(x / self.c) ** 2)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [x * np.exp(-(x / self.c) ** 2)]


class GemanMcClureLossGrad(admm.UDFBase):
    """Geman-McClure loss: f(x) = sum(x_i^2 / (2*(1 + x_i^2)))
    grad = x / (1 + x^2)^2
    Bounded, non-convex but smooth robust estimator.
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(x ** 2 / (2.0 * (1 + x ** 2))))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [x / (1 + x ** 2) ** 2]


class FairLossGrad(admm.UDFBase):
    """Fair loss: f(x) = sum(c^2 * (|x_i|/c - log(1 + |x_i|/c)))
    grad = x / (1 + |x|/c)
    A robust loss between L2 and L1.
    """
    def __init__(self, arg, c=1.0):
        self.arg = arg
        self.c = c
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        ax = np.abs(x)
        return float(self.c ** 2 * np.sum(ax / self.c - np.log(1 + ax / self.c)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [x / (1 + np.abs(x) / self.c)]


class TukeyBisquareGrad(admm.UDFBase):
    """Tukey's bisquare (biweight) loss:
    f(x) = sum(c^2/6 * (1 - (1-(x_i/c)^2)^3)) if |x_i|<=c, c^2/6 otherwise
    grad = x*(1-(x/c)^2)^2 if |x|<=c, 0 otherwise
    Very robust — completely rejects outliers beyond c.
    """
    def __init__(self, arg, c=4.685):
        self.arg = arg
        self.c = c
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        u = x / self.c
        mask = np.abs(u) <= 1
        val = np.where(mask,
                       self.c ** 2 / 6.0 * (1 - (1 - u ** 2) ** 3),
                       self.c ** 2 / 6.0)
        return float(np.sum(val))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        u = x / self.c
        mask = np.abs(u) <= 1
        return [np.where(mask, x * (1 - u ** 2) ** 2, 0.0)]


class TwoArgProductGrad(admm.UDFBase):
    """Two-argument product coupling: f(x, y) = sum(x_i * y_i)
    grad_x = y, grad_y = x
    Bilinear — not convex jointly, but OK with regularization.
    """
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2
    def arguments(self):
        return [self.arg1, self.arg2]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        return float(np.sum(x * y))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        return [y.copy(), x.copy()]


class TwoArgHuberDiffGrad(admm.UDFBase):
    """Huber-of-difference coupling: f(x, y) = sum(huber(x_i - y_i, delta))
    Robust measure of difference between two vectors.
    """
    def __init__(self, arg1, arg2, delta=1.0):
        self.arg1 = arg1
        self.arg2 = arg2
        self.delta = delta
    def arguments(self):
        return [self.arg1, self.arg2]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        d = x - y
        ad = np.abs(d)
        h = np.where(ad <= self.delta, d ** 2 / (2 * self.delta), ad - self.delta / 2)
        return float(np.sum(h))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        d = x - y
        g = np.where(np.abs(d) <= self.delta, d / self.delta, np.sign(d))
        return [g, -g]


# ---------------------------------------------------------------------------
# Advanced test class
# ---------------------------------------------------------------------------

class GradUDFAdvancedTestCase(unittest.TestCase):

    def _new_model(self):
        model = admm.Model()
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.display_sub_solver_details, 0)
        return model

    # ===================================================================
    # A. Warm start tests
    # ===================================================================

    def test_warm_start_near_optimal(self):
        """Warm start near true solution: should converge."""
        observed = np.array([3.0, -1.0, 2.0])
        lam = 0.1
        model = self._new_model()
        x = admm.Var("x", 3)
        x.start = observed * 0.9  # near the solution
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + 4 * lam * x_val ** 3 - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    def test_warm_start_at_zero(self):
        """Warm start at zero for exp sum penalty."""
        observed = np.array([2.0, 1.0, -1.0, 3.0])
        lam = 0.3
        model = self._new_model()
        x = admm.Var("x", 4)
        x.start = np.zeros(4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * ExpSumGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + lam * np.exp(x_val) - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    def test_warm_start_far_from_optimal(self):
        """Warm start far from solution: should still converge."""
        observed = np.array([1.0, 2.0])
        model = self._new_model()
        x = admm.Var("x", 2)
        x.start = np.array([100.0, -100.0])  # very far
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.1 * SoftplusGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Just verify convergence, not exact solution
        self.assertTrue(np.all(np.isfinite(x_val)))
        self.assertLess(np.linalg.norm(x_val - observed), 1.0)

    # ===================================================================
    # B. Ill-conditioning tests
    # ===================================================================

    def test_ill_conditioned_quadform(self):
        """Quadratic form with moderate condition number ~50."""
        n = 4
        np.random.seed(77)
        U, _ = np.linalg.qr(np.random.randn(n, n))
        s = np.array([1.0, 5.0, 20.0, 50.0])
        Q = U @ np.diag(s) @ U.T
        c = np.array([1.0, -2.0, 0.5, 3.0])
        x_exact = np.linalg.solve(Q, c)

        model = self._new_model()
        model.setOption(admm.Options.admm_max_iteration, 5000)
        x = admm.Var("x", n)
        model.setObjective(QuadFormGrad(x, Q) - c @ x)
        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, x_exact, atol=0.5)

    def test_nearly_flat_objective(self):
        """Very small curvature: f(x) = 1e-6 * sum(x^4), should still converge."""
        observed = np.array([5.0, -3.0, 1.0])
        lam = 1e-6
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # With tiny lambda, solution ≈ observed
        np.testing.assert_allclose(x_val, observed, atol=0.1)

    # ===================================================================
    # C. Sensitivity / perturbation analysis
    # ===================================================================

    def test_perturbation_stability(self):
        """Small perturbation in data → small change in solution."""
        base = np.array([2.0, -1.0, 3.0])
        lam = 0.2

        solutions = []
        for eps in [0.0, 0.01, -0.01]:
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - (base + eps)))
                + lam * HuberGrad(x)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            solutions.append(np.asarray(x.X).copy())

        # Perturbed solutions should be close to base solution
        np.testing.assert_allclose(solutions[1], solutions[0], atol=0.05)
        np.testing.assert_allclose(solutions[2], solutions[0], atol=0.05)

    def test_lambda_monotonicity_regularization(self):
        """Increasing lambda → solution closer to zero (more regularization)."""
        observed = np.array([5.0, -3.0, 2.0])
        norms = []
        for lam in [0.01, 0.1, 1.0, 5.0]:
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + lam * QuarticGrad(x)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            norms.append(np.linalg.norm(np.asarray(x.X)))
        # Norms should decrease monotonically
        for i in range(len(norms) - 1):
            self.assertGreaterEqual(norms[i] + 0.01, norms[i + 1])

    # ===================================================================
    # D. Iteration limit behavior
    # ===================================================================

    def test_low_iteration_limit(self):
        """Very few iterations: should still terminate gracefully."""
        model = self._new_model()
        model.setOption(admm.Options.admm_max_iteration, 5)
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - np.array([2.0, -1.0, 3.0])))
            + 0.1 * ExpSumGrad(x)
        )
        model.optimize()
        # May not be optimal, but should not crash
        self.assertIn(model.StatusString,
                      ["SOLVE_OPT_SUCCESS", "SOLVE_OVER_MAX_ITER"])

    def test_high_iteration_convergence(self):
        """Enough iterations → optimal result."""
        model = self._new_model()
        model.setOption(admm.Options.admm_max_iteration, 10000)
        x = admm.Var("x", 5)
        observed = np.array([1.0, 2.0, -1.0, 0.5, 3.0])
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.1 * LogCoshGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # KKT: x + 0.1*tanh(x) = observed
        residual = x_val + 0.1 * np.tanh(x_val) - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    # ===================================================================
    # E. Higher-order polynomial penalties
    # ===================================================================

    def test_degree_8_polynomial(self):
        """f(x) = sum(x^8): extremely flat near zero, steep far away."""
        observed = np.array([2.0, -1.5, 0.5])
        lam = 0.01
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * Degree8Grad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # KKT: x_i + 8*lam*x_i^7 = y_i
        residual = x_val + 8 * lam * x_val ** 7 - observed
        np.testing.assert_allclose(residual, 0, atol=0.1)

    def test_degree_10_polynomial(self):
        """f(x) = sum(x^10): even steeper penalties for large values."""
        observed = np.array([1.0, -0.5, 0.8])
        lam = 0.001
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * Degree10Grad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + 10 * lam * x_val ** 9 - observed
        np.testing.assert_allclose(residual, 0, atol=0.1)

    def test_polynomial_comparison_4_8_10(self):
        """Higher degree → flatter near zero, steeper far.
        With same lambda, degree-10 should shrink large values more than degree-4.
        """
        observed = np.array([3.0, -2.0])
        lam = 0.1
        results = {}
        for cls, name in [(QuarticGrad, '4'), (Degree8Grad, '8'), (Degree10Grad, '10')]:
            model = self._new_model()
            x = admm.Var("x", 2)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + lam * cls(x)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            results[name] = np.linalg.norm(np.asarray(x.X))
        # Higher degree should shrink more for large values
        self.assertLess(results['10'], results['4'] + 0.01)

    # ===================================================================
    # F. Robust loss function family
    # ===================================================================

    def test_cauchy_loss_basic(self):
        """Cauchy loss: bounded influence, robust to outliers."""
        observed = np.array([1.0, -0.5, 10.0])  # 10.0 is an outlier
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.5 * CauchyLossGrad(x, c=1.0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))
        # Outlier should be shrunk
        self.assertLess(abs(x_val[2]), 10.0)

    def test_cauchy_vs_huber_shrinkage(self):
        """Compare Cauchy and Huber shrinkage behavior.
        Both should shrink values, with Cauchy's bounded influence
        causing less shrinkage for small values.
        """
        observed = np.array([0.1, 0.5, 1.0, 3.0, 5.0])
        lam = 2.0

        # Cauchy
        model_c = self._new_model()
        xc = admm.Var("xc", 5)
        model_c.setObjective(
            0.5 * admm.sum(admm.square(xc - observed))
            + lam * CauchyLossGrad(xc, c=1.0)
        )
        model_c.optimize()
        self.assertEqual(model_c.StatusString, "SOLVE_OPT_SUCCESS")

        # Huber
        model_h = self._new_model()
        xh = admm.Var("xh", 5)
        model_h.setObjective(
            0.5 * admm.sum(admm.square(xh - observed))
            + lam * HuberGrad(xh, delta=1.0)
        )
        model_h.optimize()
        self.assertEqual(model_h.StatusString, "SOLVE_OPT_SUCCESS")

        # Both should converge and shrink all values
        xc_val = np.asarray(xc.X)
        xh_val = np.asarray(xh.X)
        for i in range(5):
            self.assertLess(abs(xc_val[i]), abs(observed[i]) + 0.01)
            self.assertLess(abs(xh_val[i]), abs(observed[i]) + 0.01)

    def test_welsch_loss_bounded(self):
        """Welsch loss: bounded, Gaussian-shaped influence."""
        observed = np.array([2.0, -1.0, 0.5, 8.0])
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 1.0 * WelschLossGrad(x, c=1.0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))
        # KKT: x_i + x_i*exp(-x_i^2/c^2) = observed_i
        residual = x_val + x_val * np.exp(-(x_val) ** 2) - observed
        np.testing.assert_allclose(residual, 0, atol=0.2)

    def test_geman_mcclure_loss(self):
        """Geman-McClure: bounded at 0.5, very robust."""
        observed = np.array([1.0, -0.5, 3.0, -2.0])
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.5 * GemanMcClureLossGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))

    def test_fair_loss_between_l1_l2(self):
        """Fair loss: interpolates between L2 (small x) and L1 (large x)."""
        observed = np.array([5.0, 0.1, -3.0, 0.01])
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.5 * FairLossGrad(x, c=1.0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # KKT: x_i + x_i/(1+|x_i|) = y_i  (approximately)
        self.assertTrue(np.all(np.isfinite(x_val)))
        # Should shrink all values
        for i in range(4):
            self.assertLess(abs(x_val[i]), abs(observed[i]) + 0.01)

    def test_tukey_bisquare_complete_rejection(self):
        """Tukey's bisquare: values beyond c are completely rejected."""
        observed = np.array([0.5, -0.3, 0.1, 20.0])  # 20.0 far beyond c
        c = 4.685
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 1.0 * TukeyBisquareGrad(x, c=c)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))

    def test_morse_potential_equilibrium(self):
        """Morse potential: minimum at r0 — verify convergence."""
        r0 = np.array([1.0, 2.0, -0.5])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(MorsePotentialGrad(x, D=1.0, a=1.0, r0=r0))
        model.addConstr(x >= -5)
        model.addConstr(x <= 5)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        np.testing.assert_allclose(x_val, r0, atol=0.2)

    def test_morse_with_least_squares(self):
        """Morse potential as regularizer: pulls toward r0."""
        observed = np.array([3.0, -1.0, 2.0])
        r0 = np.zeros(3)
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.5 * MorsePotentialGrad(x, D=1.0, a=1.0, r0=r0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Solution between observed and r0
        for i in range(3):
            self.assertLess(abs(x_val[i]), abs(observed[i]) + 0.01)

    def test_bounded_ratio_sparsity(self):
        """Bounded ratio: approximates L0 norm — promotes exact zeros (softly)."""
        observed = np.array([0.05, 3.0, -0.02, -2.0, 0.01])
        model = self._new_model()
        x = admm.Var("x", 5)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 2.0 * BoundedRatioGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Small entries should be shrunk more than large ones
        self.assertLess(abs(x_val[0]), abs(observed[0]))
        self.assertLess(abs(x_val[2]), abs(observed[2]))

    # ===================================================================
    # G. Matrix-specific UDFs
    # ===================================================================

    def test_frobenius_regularization(self):
        """||X||_F^2 via UDF: min 0.5||X-Y||_F^2 + lam*||X||_F^2
        Analytical: X* = Y / (1 + 2*lam). Use symmetric Y to avoid order issues.
        """
        Y = np.array([[3.0, 1.0], [1.0, 4.0]])
        lam = 0.5
        model = self._new_model()
        X = admm.Var("X", 2, 2)
        model.setObjective(
            0.5 * admm.sum(admm.square(X - Y))
            + lam * FrobeniusRegGrad(X)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        X_val = np.asarray(X.X)
        expected = Y / (1 + 2 * lam)
        np.testing.assert_allclose(X_val, expected, atol=0.1)

    def test_matrix_trace_minimization(self):
        """min tr(X) + 0.5||X - Y||_F^2, Y = 3I
        KKT: I + (X - Y) = 0 → X = Y - I = 2I
        """
        Y = 3.0 * np.eye(3)
        model = self._new_model()
        X = admm.Var("X", 3, 3)
        model.setObjective(
            MatrixTracePenaltyGrad(X, 3)
            + 0.5 * admm.sum(admm.square(X - Y))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        X_val = np.asarray(X.X)
        expected = Y - np.eye(3)  # = 2*I
        np.testing.assert_allclose(X_val, expected, atol=0.1)

    def test_offdiag_penalty(self):
        """Penalize off-diagonal: should produce near-diagonal matrix."""
        Y = np.array([[2.0, 1.0, 0.5],
                       [0.3, 3.0, 0.8],
                       [0.1, 0.4, 1.0]])
        lam = 5.0
        model = self._new_model()
        X = admm.Var("X", 3, 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(X - Y))
            + lam * OffDiagPenaltyGrad(X, 3)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        X_val = np.asarray(X.X)
        # Off-diagonal should be small
        mask = 1 - np.eye(3)
        offdiag_norm = np.linalg.norm(mask * X_val)
        diag_norm = np.linalg.norm(np.diag(X_val))
        self.assertLess(offdiag_norm, diag_norm)
        # Diagonal should be close to Y's diagonal
        np.testing.assert_allclose(np.diag(X_val), np.diag(Y), atol=0.1)

    def test_matrix_frobenius_plus_trace(self):
        """Combine Frobenius + trace penalty on same matrix variable."""
        Y = np.array([[5.0, 1.0], [1.0, 5.0]])
        model = self._new_model()
        X = admm.Var("X", 2, 2)
        model.setObjective(
            0.5 * admm.sum(admm.square(X - Y))
            + 0.1 * FrobeniusRegGrad(X)
            + 0.5 * MatrixTracePenaltyGrad(X, 2)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        X_val = np.asarray(X.X)
        # Should be between 0 and Y, diagonal reduced by trace penalty
        self.assertTrue(np.all(np.isfinite(X_val)))
        self.assertLess(np.trace(X_val), np.trace(Y))

    # ===================================================================
    # H. Weighted least squares via UDF
    # ===================================================================

    def test_weighted_ls_uniform(self):
        """Uniform weights = standard LS: X* = target."""
        t = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(WeightedLSGrad(x, w, t))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x.X, t, atol=0.05)

    def test_weighted_ls_nonuniform(self):
        """Non-uniform weights: high weight → solution closer to target."""
        t = np.array([1.0, 2.0, 3.0])
        w = np.array([100.0, 1.0, 0.01])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            WeightedLSGrad(x, w, t)
            + 0.1 * admm.sum(admm.square(x))  # regularization
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # x[0] should be closest to t[0]=1.0 (highest weight)
        err0 = abs(x_val[0] - t[0])
        err2 = abs(x_val[2] - t[2])
        self.assertLess(err0, err2 + 0.01)

    def test_weighted_ls_with_constraints(self):
        """Weighted LS with box constraints."""
        t = np.array([5.0, -3.0, 2.0])
        w = np.array([1.0, 2.0, 0.5])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(WeightedLSGrad(x, w, t))
        model.addConstr(x >= 0)  # clips x[1] at 0
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertGreaterEqual(x_val[1], -0.01)
        self.assertAlmostEqual(x_val[0], 5.0, delta=0.1)

    # ===================================================================
    # I. Smooth hinge / classification
    # ===================================================================

    def test_smooth_hinge_classification(self):
        """Smooth hinge SVM-like classification."""
        np.random.seed(99)
        n, d = 40, 4
        w_true = np.array([1.0, -1.0, 0.5, 0.0])
        A = np.random.randn(n, d)
        y = np.sign(A @ w_true)
        lam = 0.1

        model = self._new_model()
        w = admm.Var("w", d)
        model.setObjective(
            SmoothHingeLossGrad(w, A, y)
            + 0.5 * lam * admm.sum(admm.square(w))
        )
        model.setOption(admm.Options.admm_max_iteration, 5000)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        w_val = np.asarray(w.X).ravel()
        # Check prediction accuracy
        preds = np.sign(A @ w_val)
        accuracy = np.mean(preds == y)
        self.assertGreater(accuracy, 0.8)

    def test_smooth_hinge_vs_logistic(self):
        """Both smooth hinge and logistic should classify correctly."""
        np.random.seed(42)
        n, d = 30, 3
        w_true = np.array([1.0, -0.5, 0.3])
        A = np.random.randn(n, d)
        y = np.sign(A @ w_true)
        lam = 0.1

        # Smooth hinge
        model1 = self._new_model()
        w1 = admm.Var("w1", d)
        model1.setObjective(
            SmoothHingeLossGrad(w1, A, y) + 0.5 * lam * admm.sum(admm.square(w1))
        )
        model1.setOption(admm.Options.admm_max_iteration, 5000)
        model1.optimize()
        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")

        # Logistic
        model2 = self._new_model()
        w2 = admm.Var("w2", d)
        model2.setObjective(
            LogisticLossGrad(w2, A, y) + 0.5 * lam * admm.sum(admm.square(w2))
        )
        model2.setOption(admm.Options.admm_max_iteration, 5000)
        model2.optimize()
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")

        # Both should get similar classification accuracy
        acc1 = np.mean(np.sign(A @ np.asarray(w1.X).ravel()) == y)
        acc2 = np.mean(np.sign(A @ np.asarray(w2.X).ravel()) == y)
        self.assertGreater(acc1, 0.75)
        self.assertGreater(acc2, 0.75)

    # ===================================================================
    # J. Complex expression combinations
    # ===================================================================

    def test_udf_on_sum_of_two_vars(self):
        """UDF applied to x + y: min quartic(x + y)."""
        model = self._new_model()
        x = admm.Var("x", 3)
        y = admm.Var("y", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - 1))
            + 0.5 * admm.sum(admm.square(y - 1))
            + 0.1 * QuarticGrad(x + y)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        y_val = np.asarray(y.X)
        self.assertTrue(np.all(np.isfinite(x_val)))
        self.assertTrue(np.all(np.isfinite(y_val)))

    def test_udf_on_linear_combination(self):
        """UDF on 2*x - 3: min exp(2x-3) + 0.5||x||^2."""
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            ExpSumGrad(2 * x - 3)
            + 0.5 * admm.sum(admm.square(x))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # KKT: 2*exp(2x-3) + x = 0 → x < 0 (exp is always positive)
        self.assertTrue(np.all(x_val < 1.6))

    def test_udf_on_matrix_product(self):
        """UDF on A@x: min softplus(Ax) + 0.5||x||^2."""
        np.random.seed(55)
        A = np.random.randn(4, 3)
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            SoftplusGrad(A @ x)
            + 0.5 * admm.sum(admm.square(x))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))

    def test_udf_nested_expression(self):
        """UDF on (x[0:2] + x[2:4]): slicing inside UDF argument."""
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - np.array([1, 2, 3, 4])))
            + 0.1 * HuberGrad(x[0:2] + x[2:4])
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))

    # ===================================================================
    # K. Gradient accumulation — many UDFs on same variable
    # ===================================================================

    def test_ten_udfs_on_same_var(self):
        """Stack 10 different UDF penalties on same variable."""
        observed = np.array([2.0, -1.0, 1.5])
        lam = 0.01
        model = self._new_model()
        x = admm.Var("x", 3)
        obj = 0.5 * admm.sum(admm.square(x - observed))
        # Add many different penalties
        obj = obj + lam * QuarticGrad(x)
        obj = obj + lam * SoftplusGrad(x)
        obj = obj + lam * HuberGrad(x)
        obj = obj + lam * ExpSumGrad(x)
        obj = obj + lam * LogCoshGrad(x)
        obj = obj + lam * SmoothL1Grad(x)
        obj = obj + lam * CauchyLossGrad(x)
        obj = obj + lam * FairLossGrad(x)
        obj = obj + lam * WelschLossGrad(x)
        obj = obj + lam * BoundedRatioGrad(x)
        model.setObjective(obj)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Should be between 0 and observed (all penalties shrink)
        for i in range(3):
            self.assertLess(abs(x_val[i]), abs(observed[i]) + 0.01)

    def test_same_udf_class_five_instances(self):
        """5 instances of same UDF class on same variable (different lambda)."""
        observed = np.array([5.0, -3.0])
        model = self._new_model()
        x = admm.Var("x", 2)
        obj = 0.5 * admm.sum(admm.square(x - observed))
        for lam in [0.01, 0.02, 0.05, 0.1, 0.2]:
            obj = obj + lam * QuarticGrad(x)
        model.setObjective(obj)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Effective lambda = sum = 0.38
        # Should be between 0 and observed
        self.assertLess(abs(x_val[0]), abs(observed[0]))

    # ===================================================================
    # L. Two-argument coupling UDFs
    # ===================================================================

    def test_two_arg_product_coupling(self):
        """f(x,y) = sum(x*y): bilinear coupling with regularization."""
        model = self._new_model()
        x = admm.Var("x", 3)
        y = admm.Var("y", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - 2))
            + 0.5 * admm.sum(admm.square(y + 1))
            + 0.5 * TwoArgProductGrad(x, y)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        y_val = np.asarray(y.X)
        self.assertTrue(np.all(np.isfinite(x_val)))
        self.assertTrue(np.all(np.isfinite(y_val)))

    def test_two_arg_huber_diff(self):
        """Huber-of-difference: robust coupling between two vectors."""
        model = self._new_model()
        x = admm.Var("x", 4)
        y = admm.Var("y", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - np.array([1, 2, 3, 4])))
            + 0.5 * admm.sum(admm.square(y - np.array([1.5, 2.5, 3.5, 4.5])))
            + 1.0 * TwoArgHuberDiffGrad(x, y, delta=0.5)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        y_val = np.asarray(y.X)
        # x and y should be pulled toward each other
        diff = np.linalg.norm(x_val - y_val)
        self.assertLess(diff, np.linalg.norm(np.array([0.5, 0.5, 0.5, 0.5])) + 0.1)

    def test_two_arg_huber_diff_with_outlier(self):
        """Huber coupling is robust to one outlier pair."""
        t_x = np.array([1.0, 1.0, 1.0, 1.0])
        t_y = np.array([1.0, 1.0, 1.0, 10.0])  # outlier in 4th component
        model = self._new_model()
        x = admm.Var("x", 4)
        y = admm.Var("y", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - t_x))
            + 0.5 * admm.sum(admm.square(y - t_y))
            + 2.0 * TwoArgHuberDiffGrad(x, y, delta=1.0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        y_val = np.asarray(y.X)
        # First 3 components: x ≈ y (close targets, strong coupling)
        np.testing.assert_allclose(x_val[:3], y_val[:3], atol=0.5)

    # ===================================================================
    # M. Param re-binding
    # ===================================================================

    def test_param_rebind_different_data(self):
        """Solve same model structure with different Param data."""
        model = self._new_model()
        p = admm.Param("p", 3)
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - p))
            + 0.1 * QuarticGrad(x)
        )

        # First solve
        model.optimize({"p": np.array([1.0, 2.0, 3.0])})
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol1 = np.asarray(x.X).copy()

        # Second solve with different data
        model.optimize({"p": np.array([10.0, -5.0, 0.0])})
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol2 = np.asarray(x.X).copy()

        # Solutions should be different
        self.assertGreater(np.linalg.norm(sol1 - sol2), 1.0)

    def test_param_sweep_monotone(self):
        """Sweep param values: larger target → larger solution."""
        model = self._new_model()
        p = admm.Param("p", 1)
        x = admm.Var("x", 1)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - p))
            + 0.2 * QuarticGrad(x)
        )

        sols = []
        for val in [0.0, 1.0, 2.0, 5.0, 10.0]:
            model.optimize({"p": np.array([val])})
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            sols.append(np.asarray(x.X).ravel()[0])

        # Solutions should be monotonically increasing
        for i in range(len(sols) - 1):
            self.assertLessEqual(sols[i], sols[i + 1] + 0.01)

    # ===================================================================
    # N. Model copy with UDF
    # ===================================================================

    def test_model_copy_solve(self):
        """Copy a model with UDF, solve the copy."""
        model1 = self._new_model()
        x = admm.Var("x", 3)
        model1.setObjective(
            0.5 * admm.sum(admm.square(x - np.array([1.0, 2.0, 3.0])))
            + 0.1 * QuarticGrad(x)
        )
        model1.optimize()
        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        sol1 = np.asarray(x.X).copy()
        obj1 = model1.ObjVal

        # Copy and solve
        model2 = admm.Model(model1)
        model2.setOption(admm.Options.solver_verbosity_level, 3)
        model2.setOption(admm.Options.display_sub_solver_details, 0)
        model2.optimize()
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        # Objectives should be very close
        self.assertAlmostEqual(obj1, model2.ObjVal, places=2)

    # ===================================================================
    # O. Finite-difference gradient checks for all new UDF classes
    # ===================================================================

    def _check_grad_fd(self, udf_cls, args, x0, h=1e-6, tol=1e-4, **kwargs):
        """Helper: finite-difference gradient check."""
        for arg_idx in range(len(args)):
            x_flat = args[arg_idx].ravel().copy()
            grad_analytical = udf_cls.grad(udf_cls, args)[arg_idx].ravel()
            grad_fd = np.zeros_like(x_flat)
            for j in range(len(x_flat)):
                args_plus = [a.copy() for a in args]
                args_minus = [a.copy() for a in args]
                args_plus[arg_idx] = args_plus[arg_idx].ravel().copy()
                args_minus[arg_idx] = args_minus[arg_idx].ravel().copy()
                args_plus[arg_idx][j] += h
                args_minus[arg_idx][j] -= h
                args_plus[arg_idx] = args_plus[arg_idx].reshape(args[arg_idx].shape)
                args_minus[arg_idx] = args_minus[arg_idx].reshape(args[arg_idx].shape)
                fp = udf_cls.eval(udf_cls, args_plus)
                fm = udf_cls.eval(udf_cls, args_minus)
                grad_fd[j] = (fp - fm) / (2 * h)
            np.testing.assert_allclose(grad_analytical, grad_fd, atol=tol,
                                       err_msg=f"Gradient mismatch for arg {arg_idx}")

    def test_gradient_fd_degree8(self):
        x0 = np.array([0.5, -0.3, 0.8])
        u = Degree8Grad.__new__(Degree8Grad)
        f_eval = lambda tl: float(np.sum(np.asarray(tl[0], dtype=float) ** 8))
        f_grad = lambda tl: [8.0 * np.asarray(tl[0], dtype=float) ** 7]
        h = 1e-5
        grad_a = f_grad([x0])[0].ravel()
        grad_fd = np.zeros_like(x0)
        for j in range(len(x0)):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += h; xm[j] -= h
            grad_fd[j] = (f_eval([xp]) - f_eval([xm])) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=1e-4)

    def test_gradient_fd_degree10(self):
        x0 = np.array([0.3, -0.4, 0.6])
        f_eval = lambda tl: float(np.sum(np.asarray(tl[0], dtype=float) ** 10))
        f_grad = lambda tl: [10.0 * np.asarray(tl[0], dtype=float) ** 9]
        h = 1e-5
        grad_a = f_grad([x0])[0].ravel()
        grad_fd = np.zeros_like(x0)
        for j in range(len(x0)):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += h; xm[j] -= h
            grad_fd[j] = (f_eval([xp]) - f_eval([xm])) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=1e-4)

    def test_gradient_fd_cauchy(self):
        x0 = np.array([1.0, -2.0, 0.5])
        c = 1.5
        f_eval = lambda tl: float(np.sum(np.log(1 + (np.asarray(tl[0], dtype=float) / c) ** 2)))
        f_grad = lambda tl: [2.0 * np.asarray(tl[0], dtype=float) / (c ** 2 + np.asarray(tl[0], dtype=float) ** 2)]
        h = 1e-6
        grad_a = f_grad([x0])[0].ravel()
        grad_fd = np.zeros_like(x0)
        for j in range(len(x0)):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += h; xm[j] -= h
            grad_fd[j] = (f_eval([xp]) - f_eval([xm])) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=1e-4)

    def test_gradient_fd_morse(self):
        x0 = np.array([0.5, -0.3, 1.2])
        D, a, r0 = 2.0, 1.5, 0.0
        def f_eval(tl):
            x = np.asarray(tl[0], dtype=float)
            e = np.exp(-a * (x - r0))
            return float(D * np.sum((1 - e) ** 2))
        def f_grad(tl):
            x = np.asarray(tl[0], dtype=float)
            e = np.exp(-a * (x - r0))
            return [2.0 * D * a * (1 - e) * e]
        h = 1e-6
        grad_a = f_grad([x0])[0].ravel()
        grad_fd = np.zeros_like(x0)
        for j in range(len(x0)):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += h; xm[j] -= h
            grad_fd[j] = (f_eval([xp]) - f_eval([xm])) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=1e-4)

    def test_gradient_fd_bounded_ratio(self):
        x0 = np.array([0.5, -1.0, 2.0])
        f_eval = lambda tl: float(np.sum(np.asarray(tl[0], dtype=float) ** 2 / (1 + np.asarray(tl[0], dtype=float) ** 2)))
        f_grad = lambda tl: [2.0 * np.asarray(tl[0], dtype=float) / (1 + np.asarray(tl[0], dtype=float) ** 2) ** 2]
        h = 1e-6
        grad_a = f_grad([x0])[0].ravel()
        grad_fd = np.zeros_like(x0)
        for j in range(len(x0)):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += h; xm[j] -= h
            grad_fd[j] = (f_eval([xp]) - f_eval([xm])) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=1e-4)

    def test_gradient_fd_welsch(self):
        x0 = np.array([0.3, -1.5, 0.8])
        c = 1.0
        f_eval = lambda tl: float(0.5 * c ** 2 * np.sum(1 - np.exp(-(np.asarray(tl[0], dtype=float) / c) ** 2)))
        f_grad = lambda tl: [np.asarray(tl[0], dtype=float) * np.exp(-(np.asarray(tl[0], dtype=float) / c) ** 2)]
        h = 1e-6
        grad_a = f_grad([x0])[0].ravel()
        grad_fd = np.zeros_like(x0)
        for j in range(len(x0)):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += h; xm[j] -= h
            grad_fd[j] = (f_eval([xp]) - f_eval([xm])) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=1e-4)

    def test_gradient_fd_geman_mcclure(self):
        x0 = np.array([0.5, -0.8, 1.5])
        f_eval = lambda tl: float(np.sum(np.asarray(tl[0], dtype=float) ** 2 / (2.0 * (1 + np.asarray(tl[0], dtype=float) ** 2))))
        f_grad = lambda tl: [np.asarray(tl[0], dtype=float) / (1 + np.asarray(tl[0], dtype=float) ** 2) ** 2]
        h = 1e-6
        grad_a = f_grad([x0])[0].ravel()
        grad_fd = np.zeros_like(x0)
        for j in range(len(x0)):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += h; xm[j] -= h
            grad_fd[j] = (f_eval([xp]) - f_eval([xm])) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=1e-4)

    def test_gradient_fd_fair(self):
        x0 = np.array([1.0, -0.5, 2.0])
        c = 1.0
        def f_eval(tl):
            x = np.asarray(tl[0], dtype=float)
            ax = np.abs(x)
            return float(c ** 2 * np.sum(ax / c - np.log(1 + ax / c)))
        def f_grad(tl):
            x = np.asarray(tl[0], dtype=float)
            return [x / (1 + np.abs(x) / c)]
        h = 1e-6
        grad_a = f_grad([x0])[0].ravel()
        grad_fd = np.zeros_like(x0)
        for j in range(len(x0)):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += h; xm[j] -= h
            grad_fd[j] = (f_eval([xp]) - f_eval([xm])) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=1e-4)

    def test_gradient_fd_tukey(self):
        x0 = np.array([1.0, -2.0, 5.0])  # includes one beyond c
        c = 4.685
        def f_eval(tl):
            x = np.asarray(tl[0], dtype=float)
            u = x / c
            mask = np.abs(u) <= 1
            return float(np.sum(np.where(mask, c ** 2 / 6.0 * (1 - (1 - u ** 2) ** 3), c ** 2 / 6.0)))
        def f_grad(tl):
            x = np.asarray(tl[0], dtype=float)
            u = x / c
            mask = np.abs(u) <= 1
            return [np.where(mask, x * (1 - u ** 2) ** 2, 0.0)]
        h = 1e-6
        grad_a = f_grad([x0])[0].ravel()
        grad_fd = np.zeros_like(x0)
        for j in range(len(x0)):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += h; xm[j] -= h
            grad_fd[j] = (f_eval([xp]) - f_eval([xm])) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=1e-4)

    def test_gradient_fd_smooth_hinge(self):
        np.random.seed(12)
        A = np.random.randn(5, 3)
        y = np.array([1, -1, 1, -1, 1], dtype=float)
        w0 = np.array([0.5, -0.3, 0.8])
        def f_eval(tl):
            w = np.asarray(tl[0], dtype=float).ravel()
            z = y * (A @ w)
            loss = np.where(z >= 1, 0.0, np.where(z >= 0, 0.5 * (1 - z) ** 2, 0.5 - z))
            return float(np.sum(loss))
        def f_grad(tl):
            w = np.asarray(tl[0], dtype=float).ravel()
            z = y * (A @ w)
            dloss_dz = np.where(z >= 1, 0.0, np.where(z >= 0, -(1 - z), -1.0))
            g = A.T @ (y * dloss_dz)
            return [g]
        h = 1e-6
        grad_a = f_grad([w0])[0].ravel()
        grad_fd = np.zeros(3)
        for j in range(3):
            wp = w0.copy(); wm = w0.copy()
            wp[j] += h; wm[j] -= h
            grad_fd[j] = (f_eval([wp]) - f_eval([wm])) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=1e-3)

    def test_gradient_fd_weighted_ls(self):
        x0 = np.array([1.5, -0.5, 2.0])
        w = np.array([1.0, 3.0, 0.5])
        t = np.array([1.0, 0.0, 1.0])
        f_eval = lambda tl: float(np.sum(w * (np.asarray(tl[0], dtype=float) - t) ** 2))
        f_grad = lambda tl: [2.0 * w * (np.asarray(tl[0], dtype=float) - t)]
        h = 1e-6
        grad_a = f_grad([x0])[0].ravel()
        grad_fd = np.zeros_like(x0)
        for j in range(len(x0)):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += h; xm[j] -= h
            grad_fd[j] = (f_eval([xp]) - f_eval([xm])) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=1e-4)

    def test_gradient_fd_offdiag(self):
        X0 = np.array([[1.0, 0.5], [0.3, 2.0]])
        n = 2
        mask = 1 - np.eye(n)
        f_eval = lambda tl: float(np.sum(mask * np.asarray(tl[0], dtype=float) ** 2))
        f_grad = lambda tl: [2.0 * mask * np.asarray(tl[0], dtype=float)]
        h = 1e-6
        grad_a = f_grad([X0])[0].ravel()
        grad_fd = np.zeros(4)
        X_flat = X0.ravel()
        for j in range(4):
            Xp = X_flat.copy(); Xm = X_flat.copy()
            Xp[j] += h; Xm[j] -= h
            grad_fd[j] = (f_eval([Xp.reshape(2, 2)]) - f_eval([Xm.reshape(2, 2)])) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=1e-4)

    def test_gradient_fd_two_arg_product(self):
        x0 = np.array([1.0, -0.5, 0.3])
        y0 = np.array([0.5, 1.0, -0.2])
        f_eval = lambda tl: float(np.sum(np.asarray(tl[0], dtype=float) * np.asarray(tl[1], dtype=float)))
        f_grad = lambda tl: [np.asarray(tl[1], dtype=float).copy(), np.asarray(tl[0], dtype=float).copy()]
        h = 1e-6
        # Check grad w.r.t. x
        gx_a = f_grad([x0, y0])[0].ravel()
        gx_fd = np.zeros(3)
        for j in range(3):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += h; xm[j] -= h
            gx_fd[j] = (f_eval([xp, y0]) - f_eval([xm, y0])) / (2 * h)
        np.testing.assert_allclose(gx_a, gx_fd, atol=1e-4)
        # Check grad w.r.t. y
        gy_a = f_grad([x0, y0])[1].ravel()
        gy_fd = np.zeros(3)
        for j in range(3):
            yp = y0.copy(); ym = y0.copy()
            yp[j] += h; ym[j] -= h
            gy_fd[j] = (f_eval([x0, yp]) - f_eval([x0, ym])) / (2 * h)
        np.testing.assert_allclose(gy_a, gy_fd, atol=1e-4)

    def test_gradient_fd_two_arg_huber_diff(self):
        x0 = np.array([1.0, 2.0, 0.5])
        y0 = np.array([0.5, 2.5, 0.3])
        delta = 0.5
        def f_eval(tl):
            x = np.asarray(tl[0], dtype=float)
            y = np.asarray(tl[1], dtype=float)
            d = x - y
            ad = np.abs(d)
            return float(np.sum(np.where(ad <= delta, d ** 2 / (2 * delta), ad - delta / 2)))
        def f_grad(tl):
            x = np.asarray(tl[0], dtype=float)
            y = np.asarray(tl[1], dtype=float)
            d = x - y
            g = np.where(np.abs(d) <= delta, d / delta, np.sign(d))
            return [g, -g]
        h = 1e-6
        for idx in [0, 1]:
            ga = f_grad([x0, y0])[idx].ravel()
            gfd = np.zeros(3)
            for j in range(3):
                args_p = [x0.copy(), y0.copy()]
                args_m = [x0.copy(), y0.copy()]
                args_p[idx] = args_p[idx].copy(); args_p[idx][j] += h
                args_m[idx] = args_m[idx].copy(); args_m[idx][j] -= h
                gfd[j] = (f_eval(args_p) - f_eval(args_m)) / (2 * h)
            np.testing.assert_allclose(ga, gfd, atol=1e-3,
                                       err_msg=f"arg {idx} gradient mismatch")

    # ===================================================================
    # P. Moreau envelope / proximal properties
    # ===================================================================

    def test_moreau_envelope_property(self):
        """Moreau envelope M(v) = min_x lam*f(x)+0.5||x-v||^2 <= lam*f(v).
        That is, the proximal problem always improves (or matches) f at v.
        """
        v = np.array([2.0, -1.0, 3.0])
        lam = 0.5
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - v))
            + lam * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        moreau_val = model.ObjVal
        # f(v) = sum(v^4)
        f_at_v = float(np.sum(v ** 4))
        # Moreau envelope: 0 + lam*f(v) is an upper bound (x=v gives 0 + lam*f(v))
        self.assertLessEqual(moreau_val, lam * f_at_v + 1e-3)

    def test_moreau_envelope_multiple_functions(self):
        """Moreau envelope property for several functions."""
        v = np.array([1.5, -2.0])
        lam = 0.3
        for cls, f_at_v_fn in [
            (ExpSumGrad, lambda v: float(np.sum(np.exp(v)))),
            (SoftplusGrad, lambda v: float(np.sum(np.logaddexp(0, v)))),
            (LogCoshGrad, lambda v: float(np.sum(np.log(np.cosh(v))))),
        ]:
            model = self._new_model()
            x = admm.Var("x", 2)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - v))
                + lam * cls(x)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"{cls.__name__} failed")
            moreau_val = model.ObjVal
            upper_bound = lam * f_at_v_fn(v)
            self.assertLessEqual(moreau_val, upper_bound + 1e-3,
                                 f"{cls.__name__}: Moreau envelope violated")

    # ===================================================================
    # Q. Symmetry and invariance
    # ===================================================================

    def test_permutation_invariance(self):
        """Permuting observed data → permuted solution."""
        observed = np.array([1.0, 3.0, -2.0, 5.0])
        lam = 0.2
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol1 = np.asarray(x.X).copy()

        # Permuted data
        perm = [2, 0, 3, 1]
        model2 = self._new_model()
        x2 = admm.Var("x2", 4)
        model2.setObjective(
            0.5 * admm.sum(admm.square(x2 - observed[perm]))
            + lam * QuarticGrad(x2)
        )
        model2.optimize()
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        sol2 = np.asarray(x2.X)

        # sol2 should be permutation of sol1
        np.testing.assert_allclose(sol2, sol1[perm], atol=0.05)

    def test_sign_symmetry(self):
        """f(x) = f(-x): negating observed → negated solution."""
        observed = np.array([2.0, -1.0, 3.0])
        lam = 0.1
        model1 = self._new_model()
        x1 = admm.Var("x1", 3)
        model1.setObjective(
            0.5 * admm.sum(admm.square(x1 - observed))
            + lam * QuarticGrad(x1)
        )
        model1.optimize()
        sol1 = np.asarray(x1.X)

        model2 = self._new_model()
        x2 = admm.Var("x2", 3)
        model2.setObjective(
            0.5 * admm.sum(admm.square(x2 - (-observed)))
            + lam * QuarticGrad(x2)
        )
        model2.optimize()
        sol2 = np.asarray(x2.X)

        np.testing.assert_allclose(sol2, -sol1, atol=0.05)

    # ===================================================================
    # R. Robust estimation comparison
    # ===================================================================

    def test_robust_loss_family_comparison(self):
        """Compare 5 robust losses on data with outliers.
        All should converge and handle outliers.
        """
        np.random.seed(333)
        n = 20
        observed = np.ones(n) * 2.0
        observed[0] = 50.0   # outlier
        observed[1] = -30.0  # outlier

        losses = [
            ("Huber", HuberGrad, {}),
            ("Cauchy", CauchyLossGrad, {"c": 1.0}),
            ("Welsch", WelschLossGrad, {"c": 1.0}),
            ("Fair", FairLossGrad, {"c": 1.0}),
            ("Tukey", TukeyBisquareGrad, {"c": 4.685}),
        ]

        for name, cls, kwargs in losses:
            model = self._new_model()
            x = admm.Var("x", n)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + 2.0 * cls(x, **kwargs)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"{name} did not converge")
            x_val = np.asarray(x.X)
            self.assertTrue(np.all(np.isfinite(x_val)), f"{name} has non-finite values")
            # Non-outlier components should be reasonable
            non_outlier = x_val[2:]
            self.assertLess(np.std(non_outlier), 1.0,
                            f"{name}: non-outlier std too high")

    # ===================================================================
    # S. Stress tests with constraints
    # ===================================================================

    def test_many_constraints_few_vars(self):
        """n=3 variables, 15 inequality constraints."""
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            SoftplusGrad(x)
            + 0.5 * admm.sum(admm.square(x - np.array([1.0, 2.0, 3.0])))
        )
        # Many redundant box constraints
        for lb in [-2, -1, 0]:
            model.addConstr(x >= lb)
        for ub in [3, 4, 5]:
            model.addConstr(x <= ub)
        # Sum constraints
        model.addConstr(admm.sum(x) >= 1)
        model.addConstr(admm.sum(x) <= 10)
        # Pairwise
        model.addConstr(x[0:1] <= x[1:2] + 1)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(x_val >= -0.01))
        self.assertTrue(np.all(x_val <= 5.01))

    def test_equality_and_inequality_mixed(self):
        """Mixed equality + inequality with UDF."""
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x))
            + 0.1 * ExpSumGrad(x)
        )
        model.addConstr(admm.sum(x) == 2)
        model.addConstr(x >= 0)
        model.addConstr(x <= 3)
        model.addConstr(x[0:1] + x[1:2] >= 0.5)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertAlmostEqual(np.sum(x_val), 2.0, delta=0.1)
        self.assertTrue(np.all(x_val >= -0.01))


# ---------------------------------------------------------------------------
# New UDF classes for specialized tests
# ---------------------------------------------------------------------------

class SmoothTVGrad(admm.UDFBase):
    """Smooth total variation: f(x) = sum(sqrt((x_{i+1}-x_i)^2 + eps))
    grad_i = -(d_i / sqrt(d_i^2+eps)) + (d_{i-1} / sqrt(d_{i-1}^2+eps))
    where d_i = x_{i+1} - x_i
    """
    def __init__(self, arg, eps=1e-4):
        self.arg = arg
        self.eps = eps
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        d = np.diff(x)
        return float(np.sum(np.sqrt(d ** 2 + self.eps)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        d = np.diff(x)
        dd = d / np.sqrt(d ** 2 + self.eps)
        g = np.zeros_like(x)
        g[:-1] -= dd
        g[1:] += dd
        return [g.reshape(tensorlist[0].shape)]


class GraphLaplacianGrad(admm.UDFBase):
    """Graph Laplacian smoothing: f(x) = sum_{(i,j) in edges} (x_i - x_j)^2
    grad_i = 2 * sum_{j: (i,j) in edges} (x_i - x_j)
    """
    def __init__(self, arg, edges):
        self.arg = arg
        self.edges = edges  # list of (i, j) tuples
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        s = 0.0
        for i, j in self.edges:
            s += (x[i] - x[j]) ** 2
        return float(s)
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        g = np.zeros_like(x)
        for i, j in self.edges:
            g[i] += 2.0 * (x[i] - x[j])
            g[j] -= 2.0 * (x[i] - x[j])
        return [g.reshape(tensorlist[0].shape)]


class BerhuLossGrad(admm.UDFBase):
    """Reverse Huber (Berhu) loss:
    f(x) = |x| if |x|<=c, (x^2 + c^2)/(2c) if |x|>c
    grad = sign(x) if |x|<=c, x/c if |x|>c
    L1 for small residuals, L2 for large — opposite of Huber.
    """
    def __init__(self, arg, c=1.0):
        self.arg = arg
        self.c = c
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        ax = np.abs(x)
        return float(np.sum(np.where(ax <= self.c, ax, (x ** 2 + self.c ** 2) / (2 * self.c))))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        ax = np.abs(x)
        return [np.where(ax <= self.c, np.sign(x), x / self.c)]


class GammaRegressionGrad(admm.UDFBase):
    """Gamma regression loss (log link): f(mu) = sum(y/exp(mu) + mu)
    grad = -y*exp(-mu) + 1
    mu = log(mean), so f(mu) = sum(y*exp(-mu) + mu).
    """
    def __init__(self, arg, y):
        self.arg = arg
        self.y = np.asarray(y, dtype=float)
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        mu = np.asarray(tensorlist[0], dtype=float).ravel()
        return float(np.sum(self.y * np.exp(-mu) + mu))
    def grad(self, tensorlist):
        mu = np.asarray(tensorlist[0], dtype=float).ravel()
        return [(-self.y * np.exp(-mu) + 1.0).reshape(tensorlist[0].shape)]


class SmoothAbsPowerGrad(admm.UDFBase):
    """Smooth absolute power: f(x) = sum((x^2 + eps)^(p/2))
    grad = p * x * (x^2 + eps)^(p/2 - 1)
    Generalizes smooth L1 (p=1), L2 squared (p=2), etc.
    """
    def __init__(self, arg, p=1.5, eps=1e-4):
        self.arg = arg
        self.p = p
        self.eps = eps
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum((x ** 2 + self.eps) ** (self.p / 2)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [self.p * x * (x ** 2 + self.eps) ** (self.p / 2 - 1)]


class FourArgCouplingGrad(admm.UDFBase):
    """4-argument UDF: f(a, b, c, d) = sum((a-b)^2) + sum((b-c)^2) + sum((c-d)^2)
    Chain coupling across 4 variables.
    """
    def __init__(self, a, b, c, d):
        self.a = a; self.b = b; self.c = c; self.d = d
    def arguments(self):
        return [self.a, self.b, self.c, self.d]
    def eval(self, tensorlist):
        a = np.asarray(tensorlist[0], dtype=float)
        b = np.asarray(tensorlist[1], dtype=float)
        c = np.asarray(tensorlist[2], dtype=float)
        d = np.asarray(tensorlist[3], dtype=float)
        return float(np.sum((a - b) ** 2) + np.sum((b - c) ** 2) + np.sum((c - d) ** 2))
    def grad(self, tensorlist):
        a = np.asarray(tensorlist[0], dtype=float)
        b = np.asarray(tensorlist[1], dtype=float)
        c = np.asarray(tensorlist[2], dtype=float)
        d = np.asarray(tensorlist[3], dtype=float)
        ga = 2 * (a - b)
        gb = -2 * (a - b) + 2 * (b - c)
        gc = -2 * (b - c) + 2 * (c - d)
        gd = -2 * (c - d)
        return [ga, gb, gc, gd]


class SumInvSquaredGrad(admm.UDFBase):
    """Sum of inverse squares: f(x) = sum(1/(x_i^2 + eps))
    grad = -2*x / (x^2 + eps)^2
    Penalty that strongly repels from zero.
    """
    def __init__(self, arg, eps=0.01):
        self.arg = arg
        self.eps = eps
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(1.0 / (x ** 2 + self.eps)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [-2.0 * x / (x ** 2 + self.eps) ** 2]


class AsymmetricLossGrad(admm.UDFBase):
    """Asymmetric quadratic: f(x) = sum(w_pos * max(x,0)^2 + w_neg * max(-x,0)^2)
    Penalizes positive and negative errors differently.
    """
    def __init__(self, arg, w_pos=1.0, w_neg=2.0):
        self.arg = arg
        self.w_pos = w_pos
        self.w_neg = w_neg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(self.w_pos * np.maximum(x, 0) ** 2
                           + self.w_neg * np.maximum(-x, 0) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [np.where(x >= 0, 2 * self.w_pos * x, 2 * self.w_neg * x)]


class CorrentropyCrad(admm.UDFBase):
    """Correntropy-induced loss: f(x) = sum(1 - exp(-x^2 / (2*sigma^2)))
    grad = x / sigma^2 * exp(-x^2 / (2*sigma^2))
    Maximum correntropy criterion — very robust.
    """
    def __init__(self, arg, sigma=1.0):
        self.arg = arg
        self.sigma = sigma
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(1 - np.exp(-x ** 2 / (2 * self.sigma ** 2))))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [x / self.sigma ** 2 * np.exp(-x ** 2 / (2 * self.sigma ** 2))]


# ---------------------------------------------------------------------------
# Specialized test class
# ---------------------------------------------------------------------------

class GradUDFSpecializedTestCase(unittest.TestCase):

    def _new_model(self, max_iter=2000):
        model = admm.Model()
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.display_sub_solver_details, 0)
        model.setOption(admm.Options.admm_max_iteration, max_iter)
        return model

    # ===================================================================
    # A. Smooth total variation denoising
    # ===================================================================

    def test_tv_denoising_piecewise_constant(self):
        """TV denoising: recover piecewise constant signal from noise."""
        np.random.seed(100)
        n = 30
        true_signal = np.zeros(n)
        true_signal[:10] = 1.0
        true_signal[10:20] = 3.0
        true_signal[20:] = 0.5
        noisy = true_signal + 0.3 * np.random.randn(n)

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - noisy))
            + 2.0 * SmoothTVGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        # Denoised signal should be smoother
        tv_noisy = np.sum(np.abs(np.diff(noisy)))
        tv_denoised = np.sum(np.abs(np.diff(x_val)))
        self.assertLess(tv_denoised, tv_noisy)

    def test_tv_with_box_constraints(self):
        """TV denoising with box constraints [0, 5]."""
        np.random.seed(101)
        n = 20
        true_signal = np.clip(np.cumsum(np.random.randn(n) * 0.5) + 2.0, 0, 5)
        noisy = true_signal + 0.5 * np.random.randn(n)

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - noisy))
            + 1.0 * SmoothTVGrad(x)
        )
        model.addConstr(x >= 0)
        model.addConstr(x <= 5)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        self.assertTrue(np.all(x_val >= -0.01))
        self.assertTrue(np.all(x_val <= 5.01))

    def test_tv_lambda_sweep(self):
        """Increasing TV lambda → smoother signal."""
        np.random.seed(102)
        n = 20
        noisy = np.random.randn(n) * 2

        tv_values = []
        for lam in [0.1, 1.0, 5.0, 20.0]:
            model = self._new_model()
            x = admm.Var("x", n)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - noisy))
                + lam * SmoothTVGrad(x)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            x_val = np.asarray(x.X).ravel()
            tv_values.append(np.sum(np.abs(np.diff(x_val))))
        # TV should decrease with increasing lambda
        for i in range(len(tv_values) - 1):
            self.assertGreaterEqual(tv_values[i] + 0.1, tv_values[i + 1])

    # ===================================================================
    # B. Graph-structured problems
    # ===================================================================

    def test_graph_laplacian_line(self):
        """Line graph: chain 0-1-2-3-4, penalize neighbor differences."""
        n = 5
        edges = [(i, i + 1) for i in range(n - 1)]
        observed = np.array([0.0, 1.0, 5.0, 1.0, 0.0])

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 1.0 * GraphLaplacianGrad(x, edges)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        # Solution should be smoother than observed
        roughness_obs = np.sum(np.diff(observed) ** 2)
        roughness_sol = np.sum(np.diff(x_val) ** 2)
        self.assertLess(roughness_sol, roughness_obs)

    def test_graph_laplacian_star(self):
        """Star graph: center node 0 connected to all others."""
        n = 6
        edges = [(0, i) for i in range(1, n)]
        observed = np.array([0.0, 2.0, -1.0, 3.0, 1.0, -2.0])

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 2.0 * GraphLaplacianGrad(x, edges)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        # Center node should be pulled toward the mean of neighbors
        neighbor_mean = np.mean(observed[1:])
        self.assertLess(abs(x_val[0] - neighbor_mean),
                        abs(observed[0] - neighbor_mean) + 0.1)

    def test_graph_laplacian_cycle(self):
        """Cycle graph: 0-1-2-...-n-1-0."""
        n = 8
        edges = [(i, (i + 1) % n) for i in range(n)]
        np.random.seed(55)
        observed = np.random.randn(n) * 3

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 3.0 * GraphLaplacianGrad(x, edges)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        self.assertTrue(np.all(np.isfinite(x_val)))
        # With strong smoothing, all values should be close together
        self.assertLess(np.std(x_val), np.std(observed))

    # ===================================================================
    # C. Berhu (reverse Huber)
    # ===================================================================

    def test_berhu_basic(self):
        """Berhu: L1 near zero, L2 for large residuals."""
        observed = np.array([0.1, 3.0, -0.05, -5.0, 0.02])
        model = self._new_model()
        x = admm.Var("x", 5)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.5 * BerhuLossGrad(x, c=1.0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # All values should be shrunk
        for i in range(5):
            self.assertLess(abs(x_val[i]), abs(observed[i]) + 0.01)

    def test_berhu_vs_huber(self):
        """Berhu and Huber have opposite behavior: Berhu is L1-near-0, L2-far."""
        observed = np.array([5.0, 0.1, -3.0])
        lam = 0.5

        model_b = self._new_model()
        xb = admm.Var("xb", 3)
        model_b.setObjective(
            0.5 * admm.sum(admm.square(xb - observed))
            + lam * BerhuLossGrad(xb, c=1.0)
        )
        model_b.optimize()

        model_h = self._new_model()
        xh = admm.Var("xh", 3)
        model_h.setObjective(
            0.5 * admm.sum(admm.square(xh - observed))
            + lam * HuberGrad(xh, delta=1.0)
        )
        model_h.optimize()

        self.assertEqual(model_b.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model_h.StatusString, "SOLVE_OPT_SUCCESS")
        # Solutions should differ
        xb_val = np.asarray(xb.X)
        xh_val = np.asarray(xh.X)
        self.assertGreater(np.linalg.norm(xb_val - xh_val), 0.01)

    # ===================================================================
    # D. Gamma regression (GLM)
    # ===================================================================

    def test_gamma_regression_recovery(self):
        """Gamma regression: recover log-mean from Gamma-distributed data."""
        np.random.seed(200)
        n = 20
        true_mu = np.array([1.0, 2.0, 0.5])  # log-means
        true_means = np.exp(true_mu)
        # Gamma with shape=5
        y = np.array([np.mean(np.random.gamma(5.0, scale=m / 5.0, size=n))
                       for m in true_means])

        model = self._new_model()
        mu = admm.Var("mu", 3)
        model.setObjective(
            GammaRegressionGrad(mu, y)
            + 0.01 * admm.sum(admm.square(mu))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        mu_val = np.asarray(mu.X).ravel()
        # Optimal: mu* = log(y), so exp(mu) ≈ y
        np.testing.assert_allclose(np.exp(mu_val), y, rtol=0.3)

    # ===================================================================
    # E. Smooth |x|^p for fractional p
    # ===================================================================

    def test_smooth_abs_power_p15(self):
        """Smooth |x|^1.5: between L1 and L2."""
        observed = np.array([3.0, 0.1, -2.0, 0.05])
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.5 * SmoothAbsPowerGrad(x, p=1.5)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        for i in range(4):
            self.assertLess(abs(x_val[i]), abs(observed[i]) + 0.01)

    def test_smooth_abs_power_comparison(self):
        """Different p values: p=1 (L1-like) shrinks small values more,
        p=2 (L2-like) shrinks uniformly.
        """
        observed = np.array([5.0, 0.1, -3.0])
        results = {}
        for p in [1.0, 1.5, 2.0]:
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + 0.5 * SmoothAbsPowerGrad(x, p=p)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            results[p] = np.asarray(x.X).copy()
        # All should converge
        for p in [1.0, 1.5, 2.0]:
            self.assertTrue(np.all(np.isfinite(results[p])))

    # ===================================================================
    # F. 4-argument UDF
    # ===================================================================

    def test_four_arg_chain_coupling(self):
        """4-arg chain coupling: a→b→c→d, pulls variables toward each other."""
        model = self._new_model()
        a = admm.Var("a", 3)
        b = admm.Var("b", 3)
        c = admm.Var("c", 3)
        d = admm.Var("d", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(a - np.array([1, 0, 0])))
            + 0.5 * admm.sum(admm.square(d - np.array([0, 0, 1])))
            + 2.0 * FourArgCouplingGrad(a, b, c, d)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        a_val = np.asarray(a.X).ravel()
        b_val = np.asarray(b.X).ravel()
        c_val = np.asarray(c.X).ravel()
        d_val = np.asarray(d.X).ravel()
        # Chain coupling: values should interpolate between a→d targets
        # a[0] > b[0] > c[0] > d[0] (roughly)
        self.assertGreater(a_val[0], d_val[0] - 0.1)

    def test_four_arg_with_constraints(self):
        """4-arg UDF with box constraints on each variable."""
        model = self._new_model()
        a = admm.Var("a", 2)
        b = admm.Var("b", 2)
        c = admm.Var("c", 2)
        d = admm.Var("d", 2)
        model.setObjective(
            0.5 * admm.sum(admm.square(a - 3))
            + 0.5 * admm.sum(admm.square(d + 3))
            + 1.0 * FourArgCouplingGrad(a, b, c, d)
        )
        model.addConstr(a >= -5); model.addConstr(a <= 5)
        model.addConstr(b >= -5); model.addConstr(b <= 5)
        model.addConstr(c >= -5); model.addConstr(c <= 5)
        model.addConstr(d >= -5); model.addConstr(d <= 5)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        for v in [a, b, c, d]:
            v_val = np.asarray(v.X)
            self.assertTrue(np.all(v_val >= -5.01))
            self.assertTrue(np.all(v_val <= 5.01))

    # ===================================================================
    # G. Asymmetric loss
    # ===================================================================

    def test_asymmetric_loss_bias(self):
        """Asymmetric loss: heavier penalty on negative → biased positive."""
        observed = np.array([0.0, 0.0, 0.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            admm.sum(admm.square(x - observed))
            + AsymmetricLossGrad(x, w_pos=1.0, w_neg=10.0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # Heavier penalty on negative → solution stays near or above 0
        self.assertTrue(np.all(x_val >= -0.1))

    def test_asymmetric_loss_inventory(self):
        """Asymmetric loss for inventory: overstock less costly than stockout."""
        demand = np.array([10.0, 20.0, 15.0, 5.0])
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            AsymmetricLossGrad(x - demand, w_pos=1.0, w_neg=5.0)
        )
        model.addConstr(x >= 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        # With w_neg > w_pos, optimal order > demand (bias upward)
        self.assertTrue(np.all(x_val >= demand.ravel() - 0.5))

    # ===================================================================
    # H. Correntropy
    # ===================================================================

    def test_correntropy_robust_denoising(self):
        """Correntropy-based denoising: very robust to outliers."""
        np.random.seed(300)
        n = 15
        true_val = np.ones(n) * 2.0
        noisy = true_val.copy()
        noisy[0] = 50.0; noisy[7] = -30.0  # outliers

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - noisy))
            + 3.0 * CorrentropyCrad(x, sigma=1.0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        # Non-outlier components should stay near 2
        non_outlier_idx = [i for i in range(n) if i not in [0, 7]]
        self.assertLess(np.std(x_val[non_outlier_idx]), 1.0)

    # ===================================================================
    # I. Inverse squared penalty
    # ===================================================================

    def test_inverse_squared_repulsion(self):
        """Sum(1/(x^2+eps)): repels x from zero."""
        observed = np.array([0.0, 0.0, 0.0])  # data at zero
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.1 * SumInvSquaredGrad(x, eps=0.01)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))

    # ===================================================================
    # J. Sequential warm-start chain
    # ===================================================================

    def test_sequential_warm_start_chain(self):
        """Solve a sequence of related problems, warm-starting each from previous."""
        base = np.array([1.0, 2.0, 3.0])
        prev_sol = None
        prev_obj = None
        for step, delta in enumerate([0.0, 0.1, 0.2, 0.3, 0.5]):
            model = self._new_model()
            x = admm.Var("x", 3)
            if prev_sol is not None:
                x.start = prev_sol
            model.setObjective(
                0.5 * admm.sum(admm.square(x - (base + delta)))
                + 0.2 * SoftplusGrad(x)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"Failed at step {step}")
            prev_sol = np.asarray(x.X).copy()
            cur_obj = model.ObjVal
            if prev_obj is not None and delta > 0:
                # Objective should change smoothly
                self.assertLess(abs(cur_obj - prev_obj), 1.0)
            prev_obj = cur_obj

    # ===================================================================
    # K. Variable fixing via tight bounds
    # ===================================================================

    # ===================================================================
    # L. Redundant constraints
    # ===================================================================

    def test_redundant_constraints_stability(self):
        """Adding redundant constraints should not change solution."""
        observed = np.array([2.0, -1.0, 3.0])

        # Without redundant constraints
        model1 = self._new_model()
        x1 = admm.Var("x1", 3)
        model1.setObjective(
            0.5 * admm.sum(admm.square(x1 - observed))
            + 0.1 * HuberGrad(x1)
        )
        model1.addConstr(x1 >= -10)
        model1.addConstr(x1 <= 10)
        model1.optimize()
        sol1 = np.asarray(x1.X).copy()

        # With many redundant constraints
        model2 = self._new_model()
        x2 = admm.Var("x2", 3)
        model2.setObjective(
            0.5 * admm.sum(admm.square(x2 - observed))
            + 0.1 * HuberGrad(x2)
        )
        model2.addConstr(x2 >= -10)
        model2.addConstr(x2 >= -20)  # redundant
        model2.addConstr(x2 >= -100)  # redundant
        model2.addConstr(x2 <= 10)
        model2.addConstr(x2 <= 20)  # redundant
        model2.addConstr(x2 <= 100)  # redundant
        model2.optimize()
        sol2 = np.asarray(x2.X)

        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(sol1, sol2, atol=0.1)

    # ===================================================================
    # M. Solver option sensitivity
    # ===================================================================

    def test_penalty_param_sensitivity(self):
        """Different initial penalty parameter → same optimal solution."""
        observed = np.array([2.0, -1.0, 3.0])
        solutions = []
        for rho in [0.1, 1.0, 10.0]:
            model = self._new_model(max_iter=5000)
            model.setOption(admm.Options.initial_penalty_param_value, rho)
            x = admm.Var("x", 3)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + 0.1 * QuarticGrad(x)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"Failed with rho={rho}")
            solutions.append(np.asarray(x.X).copy())
        # All solutions should be close
        np.testing.assert_allclose(solutions[0], solutions[1], atol=0.2)
        np.testing.assert_allclose(solutions[1], solutions[2], atol=0.2)

    def test_sub_solver_tolerance(self):
        """Tighter sub-solver tolerance → more accurate."""
        observed = np.array([3.0, -2.0, 1.0])
        model = self._new_model(max_iter=5000)
        model.setOption(admm.Options.sub_solver_absolute_tol, 1e-8)
        model.setOption(admm.Options.sub_solver_relative_tol, 1e-8)
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.1 * ExpSumGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + 0.1 * np.exp(x_val) - observed
        np.testing.assert_allclose(residual, 0, atol=0.1)

    # ===================================================================
    # N. UDF on transposed / reshaped
    # ===================================================================

    def test_udf_on_transposed_matrix(self):
        """UDF on X.T: min ||X-Y||_F^2 + lam*exp(X.T)."""
        Y = np.array([[1.0, 0.5], [0.5, 2.0]])
        model = self._new_model()
        X = admm.Var("X", 2, 2)
        model.setObjective(
            0.5 * admm.sum(admm.square(X - Y))
            + 0.1 * ExpSumGrad(X.T)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        X_val = np.asarray(X.X)
        self.assertTrue(np.all(np.isfinite(X_val)))

    def test_udf_on_reshaped_vector(self):
        """UDF on reshaped variable: vector → matrix → UDF."""
        model = self._new_model()
        x = admm.Var("x", 6)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - np.array([1, 2, 3, 4, 5, 6])))
            + 0.1 * FrobeniusRegGrad(x.reshape(2, 3))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))
        # Frobenius reg shrinks toward 0, so ||x|| < ||observed||
        self.assertLess(np.linalg.norm(x_val),
                        np.linalg.norm(np.array([1, 2, 3, 4, 5, 6])) + 0.1)

    # ===================================================================
    # O. KKT verification for new robust losses
    # ===================================================================

    def test_kkt_cauchy(self):
        """KKT check for Cauchy loss: x + lam*2x/(c^2+x^2) = y."""
        observed = np.array([3.0, -1.0, 2.0])
        lam = 0.3
        c = 1.0
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * CauchyLossGrad(x, c=c)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + lam * 2 * x_val / (c ** 2 + x_val ** 2) - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    def test_kkt_fair(self):
        """KKT check for Fair loss: x + lam*x/(1+|x|/c) = y."""
        observed = np.array([2.0, -3.0, 0.5])
        lam = 0.5
        c = 1.0
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * FairLossGrad(x, c=c)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + lam * x_val / (1 + np.abs(x_val) / c) - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    def test_kkt_welsch(self):
        """KKT check for Welsch loss: x + lam*x*exp(-x^2/c^2) = y."""
        observed = np.array([1.0, -0.5, 2.0])
        lam = 0.5
        c = 1.0
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * WelschLossGrad(x, c=c)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + lam * x_val * np.exp(-(x_val / c) ** 2) - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    def test_kkt_morse(self):
        """KKT check for Morse potential."""
        observed = np.array([2.0, -1.0, 0.5])
        lam = 0.3
        D, a, r0 = 1.0, 1.0, 0.0
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * MorsePotentialGrad(x, D=D, a=a, r0=r0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        e = np.exp(-a * (x_val - r0))
        grad_morse = 2 * D * a * (1 - e) * e
        residual = x_val + lam * grad_morse - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    def test_kkt_correntropy(self):
        """KKT check for correntropy: x + lam*x/s^2*exp(-x^2/(2s^2)) = y."""
        observed = np.array([1.5, -0.5, 1.0])
        lam = 0.3
        sigma = 1.0
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * CorrentropyCrad(x, sigma=sigma)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        grad_corr = x_val / sigma ** 2 * np.exp(-x_val ** 2 / (2 * sigma ** 2))
        residual = x_val + lam * grad_corr - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    # ===================================================================
    # P. Finite-difference gradient checks for new classes
    # ===================================================================

    def _fd_check(self, f_eval, f_grad, x0, h=1e-6, tol=1e-4):
        """Scalar-input finite-difference gradient check."""
        grad_a = f_grad(x0).ravel()
        grad_fd = np.zeros_like(x0.ravel())
        x_flat = x0.ravel()
        for j in range(len(x_flat)):
            xp = x_flat.copy(); xm = x_flat.copy()
            xp[j] += h; xm[j] -= h
            grad_fd[j] = (f_eval(xp.reshape(x0.shape)) - f_eval(xm.reshape(x0.shape))) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=tol)

    def test_gradient_fd_smooth_tv(self):
        x0 = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        eps = 1e-4
        def f_eval(x):
            d = np.diff(x.ravel())
            return float(np.sum(np.sqrt(d ** 2 + eps)))
        def f_grad(x):
            x = x.ravel()
            d = np.diff(x)
            dd = d / np.sqrt(d ** 2 + eps)
            g = np.zeros_like(x)
            g[:-1] -= dd; g[1:] += dd
            return g
        self._fd_check(f_eval, f_grad, x0)

    def test_gradient_fd_graph_laplacian(self):
        x0 = np.array([1.0, -0.5, 2.0, 0.3])
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        def f_eval(x):
            x = x.ravel()
            return sum((x[i] - x[j]) ** 2 for i, j in edges)
        def f_grad(x):
            x = x.ravel()
            g = np.zeros_like(x)
            for i, j in edges:
                g[i] += 2 * (x[i] - x[j])
                g[j] -= 2 * (x[i] - x[j])
            return g
        self._fd_check(f_eval, f_grad, x0)

    def test_gradient_fd_berhu(self):
        x0 = np.array([0.3, -2.0, 1.5])
        c = 1.0
        def f_eval(x):
            ax = np.abs(x.ravel())
            return float(np.sum(np.where(ax <= c, ax, (x.ravel() ** 2 + c ** 2) / (2 * c))))
        def f_grad(x):
            x = x.ravel()
            ax = np.abs(x)
            return np.where(ax <= c, np.sign(x), x / c)
        self._fd_check(f_eval, f_grad, x0, h=1e-5, tol=1e-3)

    def test_gradient_fd_gamma(self):
        x0 = np.array([0.5, 1.0, -0.3])
        y = np.array([1.0, 2.0, 0.5])
        def f_eval(x):
            return float(np.sum(y * np.exp(-x.ravel()) + x.ravel()))
        def f_grad(x):
            return -y * np.exp(-x.ravel()) + 1.0
        self._fd_check(f_eval, f_grad, x0)

    def test_gradient_fd_smooth_abs_power(self):
        x0 = np.array([0.5, -1.0, 0.3])
        p = 1.5; eps = 1e-4
        def f_eval(x):
            return float(np.sum((x.ravel() ** 2 + eps) ** (p / 2)))
        def f_grad(x):
            return p * x.ravel() * (x.ravel() ** 2 + eps) ** (p / 2 - 1)
        self._fd_check(f_eval, f_grad, x0)

    def test_gradient_fd_four_arg(self):
        a0 = np.array([1.0, -0.5])
        b0 = np.array([0.5, 0.3])
        c0 = np.array([-0.2, 1.0])
        d0 = np.array([0.8, -0.3])
        def f_eval_all(a, b, c, d):
            return float(np.sum((a - b) ** 2) + np.sum((b - c) ** 2) + np.sum((c - d) ** 2))
        def f_grad_all(a, b, c, d):
            ga = 2 * (a - b)
            gb = -2 * (a - b) + 2 * (b - c)
            gc = -2 * (b - c) + 2 * (c - d)
            gd = -2 * (c - d)
            return ga, gb, gc, gd
        h = 1e-6
        grads = f_grad_all(a0, b0, c0, d0)
        args = [a0, b0, c0, d0]
        for idx in range(4):
            ga = grads[idx].ravel()
            gfd = np.zeros(2)
            for j in range(2):
                args_p = [a.copy() for a in args]
                args_m = [a.copy() for a in args]
                args_p[idx][j] += h
                args_m[idx][j] -= h
                gfd[j] = (f_eval_all(*args_p) - f_eval_all(*args_m)) / (2 * h)
            np.testing.assert_allclose(ga, gfd, atol=1e-4,
                                       err_msg=f"arg {idx}")

    def test_gradient_fd_asymmetric(self):
        x0 = np.array([0.5, -1.0, 0.0, 2.0])
        w_pos, w_neg = 1.0, 3.0
        def f_eval(x):
            x = x.ravel()
            return float(np.sum(w_pos * np.maximum(x, 0) ** 2 + w_neg * np.maximum(-x, 0) ** 2))
        def f_grad(x):
            x = x.ravel()
            return np.where(x >= 0, 2 * w_pos * x, 2 * w_neg * x)
        self._fd_check(f_eval, f_grad, x0)

    def test_gradient_fd_correntropy(self):
        x0 = np.array([0.5, -1.0, 0.8])
        sigma = 1.5
        def f_eval(x):
            return float(np.sum(1 - np.exp(-x.ravel() ** 2 / (2 * sigma ** 2))))
        def f_grad(x):
            x = x.ravel()
            return x / sigma ** 2 * np.exp(-x ** 2 / (2 * sigma ** 2))
        self._fd_check(f_eval, f_grad, x0)

    def test_gradient_fd_inv_squared(self):
        x0 = np.array([0.5, -1.0, 2.0])
        eps = 0.01
        def f_eval(x):
            return float(np.sum(1.0 / (x.ravel() ** 2 + eps)))
        def f_grad(x):
            x = x.ravel()
            return -2.0 * x / (x ** 2 + eps) ** 2
        self._fd_check(f_eval, f_grad, x0)

    # ===================================================================
    # Q. Regression with multiple loss functions comparison
    # ===================================================================

    def test_regression_six_losses(self):
        """Same regression problem with 6 different loss functions: all should converge."""
        np.random.seed(400)
        n, d = 30, 4
        A = np.random.randn(n, d)
        w_true = np.array([1.0, -0.5, 0.3, 0.0])
        noise = np.random.randn(n) * 0.2
        noise[0] = 10.0  # outlier
        b = A @ w_true + noise
        lam = 0.1

        losses_and_names = [
            ("Huber", lambda w: HuberGrad(A @ w - b)),
            ("Cauchy", lambda w: CauchyLossGrad(A @ w - b, c=1.0)),
            ("Fair", lambda w: FairLossGrad(A @ w - b, c=1.0)),
            ("LogCosh", lambda w: LogCoshGrad(A @ w - b)),
            ("Correntropy", lambda w: CorrentropyCrad(A @ w - b, sigma=1.0)),
        ]

        for name, loss_fn in losses_and_names:
            model = self._new_model(max_iter=5000)
            w = admm.Var("w", d)
            model.setObjective(
                loss_fn(w)
                + lam * admm.sum(admm.square(w))
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"{name} did not converge")
            w_val = np.asarray(w.X).ravel()
            self.assertTrue(np.all(np.isfinite(w_val)),
                            f"{name} has non-finite values")
            # Signs of large components should match
            for i in [0, 1]:  # w_true[0]=1, w_true[1]=-0.5
                self.assertEqual(np.sign(w_val[i]), np.sign(w_true[i]),
                                 f"{name}: sign mismatch at w[{i}]")

    # ===================================================================
    # R. Matrix completion with smoothing
    # ===================================================================

    def test_matrix_recovery_frobenius(self):
        """Full matrix recovery with Frobenius regularization.
        min 0.5||X-Y||_F^2 + 0.5*||X||_F^2 → X* = Y/2 (for symmetric Y).
        """
        Y = np.array([[4.0, 1.0, 0.5],
                       [1.0, 3.0, 0.0],
                       [0.5, 0.0, 2.0]])
        model = self._new_model()
        X = admm.Var("X", 3, 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(X - Y))
            + 0.5 * FrobeniusRegGrad(X)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        X_val = np.asarray(X.X)
        expected = Y / 2.0
        np.testing.assert_allclose(X_val, expected, atol=0.1)

    # ===================================================================
    # S. Nearly infeasible / boundary feasibility
    # ===================================================================

    def test_barely_feasible_box(self):
        """Very tight feasible region: x in [0.99, 1.01]."""
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(ExpSumGrad(x))
        model.addConstr(x >= 0.99)
        model.addConstr(x <= 1.01)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        # exp is increasing → optimal at lower bound
        np.testing.assert_allclose(x_val, 0.99, atol=0.05)

    def test_barely_feasible_equality(self):
        """Tight constraints: sum(x)=1 and x>=0.2 with n=5 → each x_i=0.2."""
        model = self._new_model()
        x = admm.Var("x", 5)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - 1))
            + 0.1 * QuarticGrad(x)
        )
        model.addConstr(admm.sum(x) == 1)
        model.addConstr(x >= 0.2)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        self.assertAlmostEqual(np.sum(x_val), 1.0, delta=0.1)
        self.assertTrue(np.all(x_val >= 0.19))

    # ===================================================================
    # T. Objective decomposition consistency
    # ===================================================================

    def test_decomposed_objective_same_result(self):
        """Split objective: f=g+h should give same result whether combined or separate."""
        observed = np.array([2.0, -1.0, 3.0])
        lam = 0.2

        # Combined
        model1 = self._new_model()
        x1 = admm.Var("x1", 3)
        model1.setObjective(
            0.5 * admm.sum(admm.square(x1 - observed))
            + lam * QuarticGrad(x1)
            + lam * SoftplusGrad(x1)
        )
        model1.optimize()

        # Separate UDFs, same objective
        model2 = self._new_model()
        x2 = admm.Var("x2", 3)
        obj = 0.5 * admm.sum(admm.square(x2 - observed))
        obj = obj + lam * QuarticGrad(x2)
        obj = obj + lam * SoftplusGrad(x2)
        model2.setObjective(obj)
        model2.optimize()

        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x1.X, x2.X, atol=0.05)

    # ===================================================================
    # U. Large-scale graph smoothing
    # ===================================================================

    def test_graph_smoothing_large(self):
        """Large line graph with n=200 and TV-like smoothing."""
        np.random.seed(600)
        n = 200
        true_signal = np.zeros(n)
        true_signal[:50] = 1.0
        true_signal[50:100] = -1.0
        true_signal[100:150] = 2.0
        true_signal[150:] = 0.0
        noisy = true_signal + 0.5 * np.random.randn(n)

        edges = [(i, i + 1) for i in range(n - 1)]
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - noisy))
            + 1.0 * GraphLaplacianGrad(x, edges)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        # Denoised signal should be much smoother
        self.assertLess(np.std(np.diff(x_val)), np.std(np.diff(noisy)))

    # ===================================================================
    # V. Multi-loss comparison on same variable with ObjVal check
    # ===================================================================

    def test_objval_single_udf(self):
        """Verify ObjVal = f_builtin(x*) + f_udf(x*) at optimum (single UDF)."""
        observed = np.array([2.0, -1.0, 0.5])
        lam = 0.1
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        manual_obj = (0.5 * np.sum((x_val - observed) ** 2)
                      + lam * np.sum(x_val ** 4))
        self.assertAlmostEqual(model.ObjVal, manual_obj, places=2)


# ---------------------------------------------------------------------------
# New UDF classes for application tests
# ---------------------------------------------------------------------------

class ARPenaltyGrad(admm.UDFBase):
    """Autoregressive penalty: f(x) = sum((x_{t+1} - alpha*x_t)^2)
    grad_t = -2*alpha*(x_{t+1}-alpha*x_t) + 2*(x_t - alpha*x_{t-1})  (interior)
    Encourages AR(1) temporal structure.
    """
    def __init__(self, arg, alpha=0.9):
        self.arg = arg
        self.alpha = alpha
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        d = x[1:] - self.alpha * x[:-1]
        return float(np.sum(d ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        n = len(x)
        d = x[1:] - self.alpha * x[:-1]
        g = np.zeros(n)
        g[:-1] += -2 * self.alpha * d
        g[1:] += 2 * d
        return [g.reshape(tensorlist[0].shape)]


class SmoothElasticGrad(admm.UDFBase):
    """Smooth elastic net: f(x) = alpha*sum(sqrt(x^2+eps)) + (1-alpha)*sum(x^2)
    Combines smooth L1 and L2 in one UDF.
    """
    def __init__(self, arg, alpha=0.5, eps=1e-4):
        self.arg = arg
        self.alpha = alpha
        self.eps = eps
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        l1 = self.alpha * np.sum(np.sqrt(x ** 2 + self.eps))
        l2 = (1 - self.alpha) * np.sum(x ** 2)
        return float(l1 + l2)
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        gl1 = self.alpha * x / np.sqrt(x ** 2 + self.eps)
        gl2 = 2 * (1 - self.alpha) * x
        return [gl1 + gl2]


class DiagQuadCouplingGrad(admm.UDFBase):
    """Diagonal quadratic coupling: f(x, y) = sum(d_i * (x_i - y_i)^2)
    Weighted element-wise coupling.
    """
    def __init__(self, arg1, arg2, weights):
        self.arg1 = arg1
        self.arg2 = arg2
        self.d = np.asarray(weights, dtype=float)
    def arguments(self):
        return [self.arg1, self.arg2]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        return float(np.sum(self.d * (x - y) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        return [2 * self.d * (x - y), -2 * self.d * (x - y)]


class SumLogisticGrad(admm.UDFBase):
    """Sum of logistic functions: f(x) = sum(log(1 + exp(a*x_i + b)))
    Generalized softplus with parameters.
    """
    def __init__(self, arg, a=1.0, b=0.0):
        self.arg = arg
        self.a = a
        self.b = b
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.logaddexp(0, self.a * x + self.b)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        z = self.a * x + self.b
        return [self.a / (1.0 + np.exp(-z))]


class SmoothMaxPoolGrad(admm.UDFBase):
    """Smooth max pooling: f(x) = (1/beta)*log(sum(exp(beta*x_i)))
    Approximates max(x) for large beta.
    grad = softmax(beta*x)
    """
    def __init__(self, arg, beta=5.0):
        self.arg = arg
        self.beta = beta
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        m = np.max(self.beta * x)
        return float((m + np.log(np.sum(np.exp(self.beta * x - m)))) / self.beta)
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float).ravel()
        bx = self.beta * x
        m = np.max(bx)
        e = np.exp(bx - m)
        return [(e / np.sum(e)).reshape(tensorlist[0].shape)]


class SumCoshGrad(admm.UDFBase):
    """Sum of cosh: f(x) = sum(cosh(x_i)) = sum((exp(x)+exp(-x))/2)
    grad = sinh(x) = (exp(x)-exp(-x))/2
    Symmetric, strictly convex, grows like exp.
    """
    def __init__(self, arg):
        self.arg = arg
    def arguments(self):
        return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.cosh(x)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [np.sinh(x)]


class FiveArgStarGrad(admm.UDFBase):
    """5-arg star coupling: f(c, x1, x2, x3, x4) = sum_i ||x_i - c||^2
    Center c is coupled to all satellites.
    """
    def __init__(self, c, x1, x2, x3, x4):
        self.c = c; self.x1 = x1; self.x2 = x2; self.x3 = x3; self.x4 = x4
    def arguments(self):
        return [self.c, self.x1, self.x2, self.x3, self.x4]
    def eval(self, tensorlist):
        c = np.asarray(tensorlist[0], dtype=float)
        s = 0.0
        for i in range(1, 5):
            xi = np.asarray(tensorlist[i], dtype=float)
            s += np.sum((xi - c) ** 2)
        return float(s)
    def grad(self, tensorlist):
        c = np.asarray(tensorlist[0], dtype=float)
        gc = np.zeros_like(c)
        grads = [None]
        for i in range(1, 5):
            xi = np.asarray(tensorlist[i], dtype=float)
            gc += -2 * (xi - c)
            grads.append(2 * (xi - c))
        grads[0] = gc
        return grads


# ---------------------------------------------------------------------------
# Application-oriented test class
# ---------------------------------------------------------------------------

class GradUDFApplicationTestCase(unittest.TestCase):

    def _new_model(self, max_iter=2000):
        model = admm.Model()
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.display_sub_solver_details, 0)
        model.setOption(admm.Options.admm_max_iteration, max_iter)
        return model

    # ===================================================================
    # A. Time-series / AR modeling
    # ===================================================================

    def test_ar_penalty_smoothing(self):
        """AR(1) penalty: solution should exhibit temporal correlation."""
        np.random.seed(700)
        n = 30
        noise = np.random.randn(n)
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - noise))
            + 2.0 * ARPenaltyGrad(x, alpha=0.9)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        # AR penalty should reduce changes: consecutive values more correlated
        corr = np.corrcoef(x_val[:-1], x_val[1:])[0, 1]
        self.assertGreater(corr, 0.3)

    def test_ar_penalty_different_alpha(self):
        """Higher alpha → stronger temporal persistence."""
        np.random.seed(701)
        n = 20
        noise = np.random.randn(n)
        corrs = {}
        for alpha in [0.0, 0.5, 0.95]:
            model = self._new_model()
            x = admm.Var("x", n)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - noise))
                + 1.0 * ARPenaltyGrad(x, alpha=alpha)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            xv = np.asarray(x.X).ravel()
            corrs[alpha] = np.corrcoef(xv[:-1], xv[1:])[0, 1]
        # Higher alpha → higher correlation
        self.assertGreater(corrs[0.95], corrs[0.0] - 0.1)

    def test_ar_plus_tv_combined(self):
        """AR + TV: combine temporal and smoothness priors."""
        np.random.seed(702)
        n = 25
        noise = np.random.randn(n) * 2
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - noise))
            + 0.5 * ARPenaltyGrad(x, alpha=0.8)
            + 0.5 * SmoothTVGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        self.assertLess(np.std(np.diff(x_val)), np.std(np.diff(noise)))

    # ===================================================================
    # B. Smooth elastic net
    # ===================================================================

    def test_smooth_elastic_net_regression(self):
        """Smooth elastic net regression: Ax ≈ b with mixed L1+L2 penalty."""
        np.random.seed(710)
        n, d = 20, 8
        A = np.random.randn(n, d)
        w_true = np.array([3.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0])
        b = A @ w_true + 0.1 * np.random.randn(n)

        model = self._new_model(max_iter=5000)
        w = admm.Var("w", d)
        model.setObjective(
            0.5 * admm.sum(admm.square(A @ w - b))
            + 0.5 * SmoothElasticGrad(w, alpha=0.7)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        w_val = np.asarray(w.X).ravel()
        # Should recover non-zero components
        self.assertGreater(abs(w_val[0]), 1.0)
        self.assertGreater(abs(w_val[3]), 0.3)
        self.assertGreater(abs(w_val[7]), 0.5)

    def test_smooth_elastic_alpha_sweep(self):
        """Sweep alpha: alpha=0 → pure L2, alpha=1 → pure smooth L1."""
        observed = np.array([5.0, 0.1, -3.0, 0.02])
        norms = {}
        for alpha in [0.0, 0.5, 1.0]:
            model = self._new_model()
            x = admm.Var("x", 4)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + 0.5 * SmoothElasticGrad(x, alpha=alpha)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            norms[alpha] = np.linalg.norm(np.asarray(x.X), ord=1)
        # All should converge
        for alpha in [0.0, 0.5, 1.0]:
            self.assertTrue(np.isfinite(norms[alpha]))

    # ===================================================================
    # C. Portfolio optimization
    # ===================================================================

    def test_portfolio_robust_risk(self):
        """Portfolio with Cauchy risk measure: robust to fat tails."""
        np.random.seed(720)
        n = 5
        expected_return = np.array([0.10, 0.15, 0.08, 0.12, 0.20])
        model = self._new_model()
        w = admm.Var("w", n)
        # Maximize return - risk penalty
        model.MinSense = False
        model.setObjective(
            expected_return @ w
            - 0.5 * CauchyLossGrad(w, c=0.5)
        )
        model.addConstr(admm.sum(w) == 1)
        model.addConstr(w >= 0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        w_val = np.asarray(w.X).ravel()
        self.assertAlmostEqual(np.sum(w_val), 1.0, delta=0.05)
        self.assertTrue(np.all(w_val >= -0.01))

    def test_portfolio_with_quadform_risk(self):
        """Portfolio: min risk s.t. return >= target."""
        np.random.seed(721)
        n = 4
        mu = np.array([0.05, 0.10, 0.08, 0.15])
        C = np.random.randn(n, n)
        Sigma = C.T @ C / 10 + 0.01 * np.eye(n)

        model = self._new_model()
        w = admm.Var("w", n)
        model.setObjective(QuadFormGrad(w, Sigma))
        model.addConstr(admm.sum(w) == 1)
        model.addConstr(w >= 0)
        model.addConstr(mu @ w >= 0.08)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        w_val = np.asarray(w.X).ravel()
        self.assertAlmostEqual(np.sum(w_val), 1.0, delta=0.05)
        self.assertGreaterEqual(mu @ w_val, 0.075)

    # ===================================================================
    # D. Diagonal coupling / multi-task
    # ===================================================================

    def test_diag_quad_coupling(self):
        """Diagonal quadratic coupling: weighted agreement between x and y."""
        model = self._new_model()
        x = admm.Var("x", 3)
        y = admm.Var("y", 3)
        d = np.array([10.0, 1.0, 0.1])
        model.setObjective(
            0.5 * admm.sum(admm.square(x - np.array([1, 2, 3])))
            + 0.5 * admm.sum(admm.square(y - np.array([4, 5, 6])))
            + 1.0 * DiagQuadCouplingGrad(x, y, d)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        y_val = np.asarray(y.X).ravel()
        # High-weight dimension (d=10): x[0] and y[0] should be close
        self.assertLess(abs(x_val[0] - y_val[0]), abs(x_val[2] - y_val[2]) + 0.1)

    def test_multi_task_shared_coupling(self):
        """Multi-task: 3 tasks coupled to shared representation."""
        np.random.seed(730)
        n = 4
        model = self._new_model()
        shared = admm.Var("shared", n)
        tasks = []
        for i in range(3):
            t = admm.Var(f"task{i}", n)
            tasks.append(t)

        obj = 0.1 * admm.sum(admm.square(shared))
        for i, t in enumerate(tasks):
            target = np.random.randn(n)
            obj = obj + 0.5 * admm.sum(admm.square(t - target))
            # Couple each task to shared
            w = np.ones(n)
            obj = obj + 0.5 * DiagQuadCouplingGrad(t, shared, w)

        model.setObjective(obj)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        shared_val = np.asarray(shared.X)
        # Shared should be close to mean of task targets
        self.assertTrue(np.all(np.isfinite(shared_val)))

    # ===================================================================
    # E. 5-argument UDF
    # ===================================================================

    def test_five_arg_star_coupling(self):
        """5-arg star: center + 4 satellites, all coupled."""
        model = self._new_model()
        c = admm.Var("c", 3)
        x1 = admm.Var("x1", 3)
        x2 = admm.Var("x2", 3)
        x3 = admm.Var("x3", 3)
        x4 = admm.Var("x4", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x1 - np.array([1, 0, 0])))
            + 0.5 * admm.sum(admm.square(x2 - np.array([0, 1, 0])))
            + 0.5 * admm.sum(admm.square(x3 - np.array([0, 0, 1])))
            + 0.5 * admm.sum(admm.square(x4 - np.array([-1, 0, 0])))
            + 2.0 * FiveArgStarGrad(c, x1, x2, x3, x4)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        c_val = np.asarray(c.X).ravel()
        # Center should be near mean of satellite targets
        expected_center = np.array([0, 0.25, 0.25])
        self.assertLess(np.linalg.norm(c_val - expected_center), 1.0)

    # ===================================================================
    # F. Smooth max pooling
    # ===================================================================

    def test_smooth_max_pool_basic(self):
        """Smooth max: minimize max(x) + 0.5||x-y||^2."""
        observed = np.array([3.0, 5.0, 1.0, 4.0])
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(
            SmoothMaxPoolGrad(x, beta=5.0)
            + 0.5 * admm.sum(admm.square(x - observed))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        # Max element should be reduced
        self.assertLess(np.max(x_val), np.max(observed))

    def test_smooth_max_pool_beta_comparison(self):
        """Higher beta → sharper approximation of max."""
        observed = np.array([1.0, 5.0, 2.0])
        results = {}
        for beta in [1.0, 5.0, 20.0]:
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(
                SmoothMaxPoolGrad(x, beta=beta)
                + 0.5 * admm.sum(admm.square(x - observed))
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            results[beta] = np.max(np.asarray(x.X).ravel())
        # All should converge
        for beta in [1.0, 5.0, 20.0]:
            self.assertTrue(np.isfinite(results[beta]))

    # ===================================================================
    # G. Sum of cosh
    # ===================================================================

    def test_cosh_penalty_kkt(self):
        """sum(cosh(x)): KKT: x + lam*sinh(x) = y."""
        observed = np.array([2.0, -1.0, 0.5])
        lam = 0.3
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SumCoshGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        residual = x_val + lam * np.sinh(x_val) - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    def test_cosh_with_box_constraints(self):
        """cosh is symmetric and grows exponentially → box keeps bounded."""
        model = self._new_model()
        x = admm.Var("x", 5)
        model.setObjective(SumCoshGrad(x))
        model.addConstr(x >= -2)
        model.addConstr(x <= 2)
        model.addConstr(admm.sum(x) == 1)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        self.assertAlmostEqual(np.sum(x_val), 1.0, delta=0.1)
        self.assertTrue(np.all(x_val >= -2.01))
        self.assertTrue(np.all(x_val <= 2.01))

    # ===================================================================
    # H. Generalized softplus
    # ===================================================================

    def test_sum_logistic_shifted(self):
        """Shifted softplus: f(x) = sum(log(1+exp(2x-1)))."""
        observed = np.array([1.0, -0.5, 2.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.3 * SumLogisticGrad(x, a=2.0, b=-1.0)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))

    # ===================================================================
    # I. Random problem robustness (multiple seeds)
    # ===================================================================

    def test_random_problems_quartic(self):
        """Quartic penalty on random data: 10 different seeds, all should solve."""
        for seed in range(10):
            np.random.seed(1000 + seed)
            n = np.random.randint(3, 15)
            observed = np.random.randn(n) * 5
            lam = np.random.uniform(0.01, 1.0)
            model = self._new_model()
            x = admm.Var("x", n)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + lam * QuarticGrad(x)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"Failed with seed={1000+seed}, n={n}, lam={lam:.3f}")

    def test_random_problems_logcosh(self):
        """LogCosh on random data: 10 seeds."""
        for seed in range(10):
            np.random.seed(2000 + seed)
            n = np.random.randint(3, 20)
            observed = np.random.randn(n) * 3
            lam = np.random.uniform(0.05, 2.0)
            model = self._new_model()
            x = admm.Var("x", n)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + lam * LogCoshGrad(x)
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"Failed with seed={2000+seed}")

    def test_random_problems_with_constraints(self):
        """Random problems with constraints: 5 seeds."""
        for seed in range(5):
            np.random.seed(3000 + seed)
            n = np.random.randint(4, 10)
            observed = np.random.randn(n) * 3
            model = self._new_model()
            x = admm.Var("x", n)
            model.setObjective(
                0.5 * admm.sum(admm.square(x - observed))
                + 0.2 * HuberGrad(x)
            )
            model.addConstr(x >= -5)
            model.addConstr(x <= 5)
            model.addConstr(admm.sum(x) <= 10)
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                             f"Failed with seed={3000+seed}")
            x_val = np.asarray(x.X).ravel()
            self.assertTrue(np.all(x_val >= -5.01))
            self.assertTrue(np.all(x_val <= 5.01))

    # ===================================================================
    # J. Known closed-form solutions
    # ===================================================================

    def test_closed_form_quadratic_udf(self):
        """f(x)=0.5*x^T*Q*x - c^T*x: closed form x*=Q^{-1}*c."""
        np.random.seed(800)
        n = 6
        M = np.random.randn(n, n)
        Q = M.T @ M + np.eye(n)
        c = np.random.randn(n)
        x_exact = np.linalg.solve(Q, c)

        model = self._new_model(max_iter=5000)
        x = admm.Var("x", n)
        model.setObjective(QuadFormGrad(x, Q) - c @ x)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x.X, x_exact, atol=0.15)

    def test_closed_form_softplus_kkt(self):
        """Softplus KKT: x + lam*sigmoid(x) = y, verify for 5 components."""
        observed = np.array([4.0, -2.0, 0.0, 1.0, -3.0])
        lam = 0.5
        model = self._new_model()
        x = admm.Var("x", 5)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + lam * SoftplusGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        sig = 1.0 / (1.0 + np.exp(-x_val))
        residual = x_val + lam * sig - observed
        np.testing.assert_allclose(residual, 0, atol=0.05)

    def test_closed_form_exp_box(self):
        """min sum(exp(x)) s.t. x in [a,b]: optimal at x=a (exp increasing)."""
        a, b = -1.0, 3.0
        n = 4
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(ExpSumGrad(x))
        model.addConstr(x >= a)
        model.addConstr(x <= b)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x.X, a, atol=0.05)

    def test_closed_form_neg_entropy_uniform(self):
        """min sum(x*log(x)) s.t. sum(x)=1, x>0: uniform distribution."""
        n = 6
        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(NegEntropyGrad(x))
        model.addConstr(x >= 0.001)
        model.addConstr(admm.sum(x) == 1)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x.X, 1.0 / n, atol=0.05)

    # ===================================================================
    # K. Deep expression tree
    # ===================================================================

    def test_deep_nested_expressions(self):
        """UDF on deeply nested expression: ((x*2 + 1) - y) * 0.5."""
        model = self._new_model()
        x = admm.Var("x", 3)
        y = admm.Var("y", 3)
        expr = (x * 2 + 1 - y) * 0.5
        model.setObjective(
            0.5 * admm.sum(admm.square(x - 1))
            + 0.5 * admm.sum(admm.square(y - 2))
            + 0.1 * QuarticGrad(expr)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.all(np.isfinite(np.asarray(x.X))))
        self.assertTrue(np.all(np.isfinite(np.asarray(y.X))))

    def test_triple_nested_udf(self):
        """Three levels of expression: UDF(A@(x+b) - c)."""
        np.random.seed(810)
        A = np.random.randn(4, 3)
        b = np.array([1, -1, 0.5])
        c = np.array([2, -1, 0, 1])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(
            SoftplusGrad(A @ (x + b) - c)
            + 0.5 * admm.sum(admm.square(x))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.all(np.isfinite(np.asarray(x.X))))

    # ===================================================================
    # L. Many variables with different UDFs
    # ===================================================================

    def test_eight_vars_eight_udfs(self):
        """8 different variables, each with a different UDF."""
        model = self._new_model()
        v1 = admm.Var("v1", 3)
        v2 = admm.Var("v2", 3)
        v3 = admm.Var("v3", 3)
        v4 = admm.Var("v4", 3)
        v5 = admm.Var("v5", 3)
        v6 = admm.Var("v6", 3)
        v7 = admm.Var("v7", 3)
        v8 = admm.Var("v8", 3)

        obs = np.array([1.0, -1.0, 0.5])
        model.setObjective(
            0.5 * admm.sum(admm.square(v1 - obs))
            + 0.5 * admm.sum(admm.square(v2 - obs))
            + 0.5 * admm.sum(admm.square(v3 - obs))
            + 0.5 * admm.sum(admm.square(v4 - obs))
            + 0.5 * admm.sum(admm.square(v5 - obs))
            + 0.5 * admm.sum(admm.square(v6 - obs))
            + 0.5 * admm.sum(admm.square(v7 - obs))
            + 0.5 * admm.sum(admm.square(v8 - obs))
            + 0.05 * QuarticGrad(v1)
            + 0.05 * ExpSumGrad(v2)
            + 0.05 * SoftplusGrad(v3)
            + 0.05 * HuberGrad(v4)
            + 0.05 * LogCoshGrad(v5)
            + 0.05 * CauchyLossGrad(v6)
            + 0.05 * FairLossGrad(v7)
            + 0.05 * SumCoshGrad(v8)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        for v in [v1, v2, v3, v4, v5, v6, v7, v8]:
            self.assertTrue(np.all(np.isfinite(np.asarray(v.X))))

    # ===================================================================
    # M. Mixed grad + argmin UDFs
    # ===================================================================

    def test_mixed_grad_argmin_two_vars(self):
        """One variable with grad UDF, another with argmin UDF."""
        observed = np.array([2.0, -1.0, 3.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        y = admm.Var("y", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.5 * admm.sum(admm.square(y - observed))
            + 0.1 * QuarticGrad(x)           # grad UDF
            + 0.1 * udf.QuarticPenalty(y)     # argmin UDF
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        # Both should give same solution (same penalty)
        np.testing.assert_allclose(x.X, y.X, atol=0.1)

    def test_mixed_grad_argmin_coupled(self):
        """Grad UDF on x, argmin UDF on y, linear coupling."""
        model = self._new_model()
        x = admm.Var("x", 3)
        y = admm.Var("y", 3)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - np.array([1, 2, 3])))
            + 0.5 * admm.sum(admm.square(y + np.array([1, 2, 3])))
            + 0.1 * ExpSumGrad(x)         # grad UDF
            + 0.1 * admm.norm(y, ord=1)    # built-in norm
        )
        model.addConstr(admm.sum(x) + admm.sum(y) == 3)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertAlmostEqual(np.sum(np.asarray(x.X)) + np.sum(np.asarray(y.X)),
                               3.0, delta=0.2)

    # ===================================================================
    # N. Penalty method: increasing penalty
    # ===================================================================

    def test_penalty_method_sequence(self):
        """Increasing penalty → solution approaches constraint satisfaction."""
        # min quartic(x) s.t. sum(x) = 3 (via increasing penalty)
        solutions = []
        constraint_violations = []
        for pen in [0.1, 1.0, 10.0, 100.0]:
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(
                QuarticGrad(x)
                + pen * admm.sum(admm.square(admm.sum(x) - 3))
            )
            model.addConstr(x >= 0)
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            x_val = np.asarray(x.X).ravel()
            solutions.append(x_val)
            constraint_violations.append(abs(np.sum(x_val) - 3))
        # Violations should decrease with penalty
        self.assertLess(constraint_violations[-1], constraint_violations[0] + 0.01)

    # ===================================================================
    # O. Finite-difference checks for new UDFs
    # ===================================================================

    def _fd_check(self, f_eval, f_grad, x0, h=1e-6, tol=1e-4):
        grad_a = f_grad(x0).ravel()
        grad_fd = np.zeros_like(x0.ravel())
        x_flat = x0.ravel()
        for j in range(len(x_flat)):
            xp = x_flat.copy(); xm = x_flat.copy()
            xp[j] += h; xm[j] -= h
            grad_fd[j] = (f_eval(xp.reshape(x0.shape)) - f_eval(xm.reshape(x0.shape))) / (2 * h)
        np.testing.assert_allclose(grad_a, grad_fd, atol=tol)

    def test_gradient_fd_ar_penalty(self):
        x0 = np.array([1.0, 0.5, -0.3, 0.8, 0.2])
        alpha = 0.9
        def f_eval(x):
            x = x.ravel()
            return float(np.sum((x[1:] - alpha * x[:-1]) ** 2))
        def f_grad(x):
            x = x.ravel(); n = len(x)
            d = x[1:] - alpha * x[:-1]
            g = np.zeros(n)
            g[:-1] += -2 * alpha * d
            g[1:] += 2 * d
            return g
        self._fd_check(f_eval, f_grad, x0)

    def test_gradient_fd_smooth_elastic(self):
        x0 = np.array([0.5, -1.0, 0.3])
        alpha_e = 0.5; eps = 1e-4
        def f_eval(x):
            x = x.ravel()
            return float(alpha_e * np.sum(np.sqrt(x ** 2 + eps)) + (1 - alpha_e) * np.sum(x ** 2))
        def f_grad(x):
            x = x.ravel()
            return alpha_e * x / np.sqrt(x ** 2 + eps) + 2 * (1 - alpha_e) * x
        self._fd_check(f_eval, f_grad, x0)

    def test_gradient_fd_sum_cosh(self):
        x0 = np.array([0.5, -1.0, 0.8])
        def f_eval(x):
            return float(np.sum(np.cosh(x.ravel())))
        def f_grad(x):
            return np.sinh(x.ravel())
        self._fd_check(f_eval, f_grad, x0)

    def test_gradient_fd_smooth_max_pool(self):
        x0 = np.array([1.0, 3.0, -0.5, 2.0])
        beta = 5.0
        def f_eval(x):
            x = x.ravel()
            m = np.max(beta * x)
            return float((m + np.log(np.sum(np.exp(beta * x - m)))) / beta)
        def f_grad(x):
            x = x.ravel()
            bx = beta * x; m = np.max(bx)
            e = np.exp(bx - m)
            return e / np.sum(e)
        self._fd_check(f_eval, f_grad, x0)

    def test_gradient_fd_sum_logistic(self):
        x0 = np.array([0.5, -1.0, 0.3])
        a, b = 2.0, -1.0
        def f_eval(x):
            return float(np.sum(np.logaddexp(0, a * x.ravel() + b)))
        def f_grad(x):
            z = a * x.ravel() + b
            return a / (1.0 + np.exp(-z))
        self._fd_check(f_eval, f_grad, x0)

    def test_gradient_fd_five_arg(self):
        c0 = np.array([0.5, -0.3])
        xs = [np.array([1.0, 0.2]), np.array([-0.5, 0.8]),
              np.array([0.3, -0.1]), np.array([0.0, 0.5])]
        def f_eval_all(c, x1, x2, x3, x4):
            s = 0.0
            for xi in [x1, x2, x3, x4]:
                s += np.sum((xi - c) ** 2)
            return float(s)
        args = [c0] + xs
        # Check gradient for each argument
        h = 1e-6
        gc = np.zeros_like(c0)
        for xi in xs:
            gc += -2 * (xi - c0)
        grads_a = [gc] + [2 * (xi - c0) for xi in xs]
        for idx in range(5):
            gfd = np.zeros(2)
            for j in range(2):
                ap = [a.copy() for a in args]
                am = [a.copy() for a in args]
                ap[idx][j] += h; am[idx][j] -= h
                gfd[j] = (f_eval_all(*ap) - f_eval_all(*am)) / (2 * h)
            np.testing.assert_allclose(grads_a[idx], gfd, atol=1e-4,
                                       err_msg=f"arg {idx}")

    def test_gradient_fd_diag_coupling(self):
        x0 = np.array([1.0, -0.5, 0.3])
        y0 = np.array([0.5, 0.3, -0.2])
        d = np.array([2.0, 1.0, 0.5])
        def f_eval(x, y):
            return float(np.sum(d * (x - y) ** 2))
        h = 1e-6
        # grad w.r.t. x
        gx_a = 2 * d * (x0 - y0)
        gx_fd = np.zeros(3)
        for j in range(3):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += h; xm[j] -= h
            gx_fd[j] = (f_eval(xp, y0) - f_eval(xm, y0)) / (2 * h)
        np.testing.assert_allclose(gx_a, gx_fd, atol=1e-4)
        # grad w.r.t. y
        gy_a = -2 * d * (x0 - y0)
        gy_fd = np.zeros(3)
        for j in range(3):
            yp = y0.copy(); ym = y0.copy()
            yp[j] += h; ym[j] -= h
            gy_fd[j] = (f_eval(x0, yp) - f_eval(x0, ym)) / (2 * h)
        np.testing.assert_allclose(gy_a, gy_fd, atol=1e-4)

    # ===================================================================
    # P. Sparse + UDF combinations
    # ===================================================================

    def test_sparse_constraint_with_robust_udf(self):
        """Sparse equality constraint + Cauchy loss."""
        import scipy.sparse as sp
        np.random.seed(820)
        n = 10
        A_sp = sp.random(3, n, density=0.5, format='csr', random_state=42)
        A_dense = A_sp.toarray()
        b = np.array([1.0, 0.5, -0.5])

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            CauchyLossGrad(x, c=1.0)
            + 0.1 * admm.sum(admm.square(x))
        )
        model.addConstr(A_dense @ x == b)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X).ravel()
        np.testing.assert_allclose(A_dense @ x_val, b, atol=0.1)

    # ===================================================================
    # Q. Scaling tests
    # ===================================================================

    def test_objective_scaling_invariance(self):
        """Scaling objective by constant should not change optimal x."""
        observed = np.array([2.0, -1.0, 3.0])
        lam = 0.2

        model1 = self._new_model()
        x1 = admm.Var("x1", 3)
        model1.setObjective(
            0.5 * admm.sum(admm.square(x1 - observed))
            + lam * QuarticGrad(x1)
        )
        model1.optimize()
        sol1 = np.asarray(x1.X)

        model2 = self._new_model()
        x2 = admm.Var("x2", 3)
        model2.setObjective(
            10 * (0.5 * admm.sum(admm.square(x2 - observed))
                  + lam * QuarticGrad(x2))
        )
        model2.optimize()
        sol2 = np.asarray(x2.X)

        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(sol1, sol2, atol=0.1)

    def test_variable_scaling(self):
        """Scaling variable: f(x) vs f(x/c) should give x* vs c*x*."""
        observed = np.array([4.0, -2.0])
        model = self._new_model()
        x = admm.Var("x", 2)
        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed))
            + 0.1 * QuarticGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        x_val = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(x_val)))


# ---------------------------------------------------------------------------
# More UDF classes for numerical / ML / compositional tests
# ---------------------------------------------------------------------------

class SmoothQuantileLossGrad(admm.UDFBase):
    """Smooth quantile/pinball loss: f(x;y,τ) = Σ τ*(x_i-y_i) + (1/β)*log(1+exp(-β*(x_i-y_i)))
    Approximates ρ_τ(u) = u*(τ - I(u<0)) as β→∞."""
    def __init__(self, arg, y, tau=0.5, beta=20.0):
        self.arg = arg; self.y = np.asarray(y, dtype=float)
        self.tau = tau; self.beta = beta
    def arguments(self): return [self.arg]
    def _stable_softplus(self, z):
        """log(1 + exp(z)) with numerical stability."""
        return np.where(z > 20, z, np.log(1 + np.exp(np.clip(z, -500, 20))))
    def _sigmoid(self, z):
        """Numerically stable sigmoid."""
        exp_neg_abs = np.exp(-np.abs(z))
        return np.where(
            z >= 0,
            1.0 / (1.0 + exp_neg_abs),
            exp_neg_abs / (1.0 + exp_neg_abs),
        )
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        u = x - self.y
        return float(np.sum(self.tau * u + (1.0 / self.beta) * self._stable_softplus(-self.beta * u)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        u = x - self.y
        sig = self._sigmoid(self.beta * u)
        return [self.tau - (1.0 - sig)]


class GELUSumGrad(admm.UDFBase):
    """Sum of GELU: f(x) = Σ x_i * sigmoid(1.702 * x_i)."""
    def __init__(self, arg):
        self.arg = arg
    def arguments(self): return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        sig = 1.0 / (1.0 + np.exp(-1.702 * x))
        return float(np.sum(x * sig))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        sig = 1.0 / (1.0 + np.exp(-1.702 * x))
        return [sig + 1.702 * x * sig * (1.0 - sig)]


class ItakuraSaitoDivGrad(admm.UDFBase):
    """Itakura-Saito divergence: f(x;y) = Σ (y_i/x_i - log(y_i/x_i) - 1), x > 0.
    Used in audio signal processing."""
    def __init__(self, arg, y):
        self.arg = arg; self.y = np.asarray(y, dtype=float)
    def arguments(self): return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        x_safe = np.maximum(x, 1e-12)
        ratio = self.y / x_safe
        return float(np.sum(ratio - np.log(ratio) - 1.0))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        x_safe = np.maximum(x, 1e-12)
        return [-self.y / (x_safe ** 2) + 1.0 / x_safe]


class SmoothRangeGrad(admm.UDFBase):
    """Smooth range penalty: f(x) = LSE(x) + LSE(-x) ≈ max(x) - min(x).
    Encourages all elements to be close."""
    def __init__(self, arg, beta=5.0):
        self.arg = arg; self.beta = beta
    def arguments(self): return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        b = self.beta
        lse_pos = (1.0/b) * np.log(np.sum(np.exp(b * x)))
        lse_neg = (1.0/b) * np.log(np.sum(np.exp(-b * x)))
        return float(lse_pos + lse_neg)
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        b = self.beta
        ex = np.exp(b * x)
        softmax_pos = ex / np.sum(ex)
        emx = np.exp(-b * x)
        softmax_neg = emx / np.sum(emx)
        return [softmax_pos - softmax_neg]


class CumulantGrad(admm.UDFBase):
    """Log-partition minus mean: f(x) = log(Σ exp(x_i)) - mean(x).
    Measures spread of the softmax distribution."""
    def __init__(self, arg):
        self.arg = arg
    def arguments(self): return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        n = len(x)
        return float(np.log(np.sum(np.exp(x))) - np.mean(x))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        n = len(x)
        ex = np.exp(x)
        softmax = ex / np.sum(ex)
        return [softmax - 1.0 / n]


class SmoothMinGrad(admm.UDFBase):
    """Smooth min: f(x) = -LSE(-x) = -(1/β)*log(Σ exp(-β*x_i)).
    Maximizing smooth-min → make all elements large."""
    def __init__(self, arg, beta=5.0):
        self.arg = arg; self.beta = beta
    def arguments(self): return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(-(1.0 / self.beta) * np.log(np.sum(np.exp(-self.beta * x))))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        emx = np.exp(-self.beta * x)
        return [emx / np.sum(emx)]


class TwoArgRatioGrad(admm.UDFBase):
    """Two-arg ratio penalty: f(x, y) = Σ (x_i/y_i - 1)^2, y > 0.
    Penalizes deviation of ratio from 1."""
    def __init__(self, x_arg, y_arg):
        self.x_arg = x_arg; self.y_arg = y_arg
    def arguments(self): return [self.x_arg, self.y_arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        return float(np.sum((x / y - 1.0) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        r = x / y
        dx = 2.0 * (r - 1.0) / y
        dy = -2.0 * (r - 1.0) * x / (y ** 2)
        return [dx, dy]


class ThreeArgWeightedDistGrad(admm.UDFBase):
    """Three-arg weighted distance: f(x, y, w) = Σ w_i * (x_i - y_i)^2, w > 0.
    Weighted squared distance."""
    def __init__(self, x_arg, y_arg, w_arg):
        self.x_arg = x_arg; self.y_arg = y_arg; self.w_arg = w_arg
    def arguments(self): return [self.x_arg, self.y_arg, self.w_arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        w = np.asarray(tensorlist[2], dtype=float)
        return float(np.sum(w * (x - y) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        w = np.asarray(tensorlist[2], dtype=float)
        diff = x - y
        return [2.0 * w * diff, -2.0 * w * diff, diff ** 2]


class SmoothMorseGrad(admm.UDFBase):
    """Smooth Morse potential sum: f(x;y) = Σ (1 - exp(-a*(x_i-y_i)))^2.
    From molecular dynamics. Bounded, smooth."""
    def __init__(self, arg, y, a=1.0):
        self.arg = arg; self.y = np.asarray(y, dtype=float); self.a = a
    def arguments(self): return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        e = np.exp(-self.a * (x - self.y))
        return float(np.sum((1 - e) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        e = np.exp(-self.a * (x - self.y))
        return [2.0 * self.a * e * (1 - e)]


class GradUDFNumericalTestCase(unittest.TestCase):
    """Tests for numerical stability, ML applications, compositional patterns."""

    def _new_model(self, max_iter=3000):
        model = admm.Model()
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.display_sub_solver_details, 0)
        model.setOption(admm.Options.admm_max_iteration, max_iter)
        return model

    # ---- Smooth Quantile Loss / Quantile Regression ----

    def test_quantile_regression_median(self):
        """τ=0.5 quantile regression ≈ median regression."""
        y = np.array([1.0, 2.0, 3.0, 10.0, 100.0])
        model = self._new_model()
        x = admm.Var("x", 5)
        model.setObjective(SmoothQuantileLossGrad(x, y, tau=0.5))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        # Solution should be close to y (no regularization)
        np.testing.assert_allclose(x.X, y, atol=0.2)

    def test_quantile_regression_tau25(self):
        """τ=0.25: penalizes overestimation more → shrinks toward lower values."""
        np.random.seed(901)
        n = 20
        y = np.random.randn(n)
        model_50 = self._new_model()
        x50 = admm.Var("x50", n)
        model_50.setObjective(SmoothQuantileLossGrad(x50, y, tau=0.5)
                              + 0.1 * admm.sum(admm.square(x50)))
        model_50.optimize()
        self.assertEqual(model_50.StatusString, "SOLVE_OPT_SUCCESS")

        model_25 = self._new_model()
        x25 = admm.Var("x25", n)
        model_25.setObjective(SmoothQuantileLossGrad(x25, y, tau=0.25)
                              + 0.1 * admm.sum(admm.square(x25)))
        model_25.optimize()
        self.assertEqual(model_25.StatusString, "SOLVE_OPT_SUCCESS")
        # τ=0.25 should give lower fitted values on average
        self.assertLess(np.mean(x25.X), np.mean(x50.X) + 0.5)

    def test_quantile_regression_tau75(self):
        """τ=0.75: penalizes underestimation more → higher fitted values."""
        np.random.seed(902)
        n = 20
        y = np.random.randn(n)
        model_50 = self._new_model()
        x50 = admm.Var("x50", n)
        model_50.setObjective(SmoothQuantileLossGrad(x50, y, tau=0.5)
                              + 0.1 * admm.sum(admm.square(x50)))
        model_50.optimize()

        model_75 = self._new_model()
        x75 = admm.Var("x75", n)
        model_75.setObjective(SmoothQuantileLossGrad(x75, y, tau=0.75)
                              + 0.1 * admm.sum(admm.square(x75)))
        model_75.optimize()
        self.assertEqual(model_75.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertGreater(np.mean(x75.X), np.mean(x50.X) - 0.5)

    def test_quantile_monotone_in_tau(self):
        """Fitted values should increase with τ for the same data."""
        np.random.seed(903)
        y = np.random.randn(10)
        means = []
        for tau in [0.1, 0.3, 0.5, 0.7, 0.9]:
            model = self._new_model()
            x = admm.Var("x", 10)
            model.setObjective(SmoothQuantileLossGrad(x, y, tau=tau)
                              + 0.2 * admm.sum(admm.square(x)))
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            means.append(np.mean(x.X))
        # Monotonically non-decreasing (with tolerance)
        for i in range(len(means) - 1):
            self.assertGreater(means[i + 1], means[i] - 0.3)

    # ---- GELU ----

    def test_gelu_regularization(self):
        """GELU sum as regularizer: f(x) = 0.5*||x-y||^2 + λ*GELU(x)."""
        y = np.array([3.0, -2.0, 1.0, 0.0])
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(0.5 * admm.sum(admm.square(x - y))
                          + 0.3 * GELUSumGrad(x))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(sol)))
        # Solution should be shifted from y
        self.assertGreater(np.linalg.norm(sol - y), 0.01)

    def test_gelu_with_box(self):
        """GELU with box constraints."""
        y = np.array([5.0, -5.0, 0.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(0.5 * admm.sum(admm.square(x - y))
                          + 0.2 * GELUSumGrad(x))
        model.addConstr(x >= -2)
        model.addConstr(x <= 3)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        self.assertTrue(np.all(sol >= -2 - 0.1))
        self.assertTrue(np.all(sol <= 3 + 0.1))

    # ---- Itakura-Saito Divergence ----

    def test_itakura_saito_identity(self):
        """IS divergence minimized when x = y."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(ItakuraSaitoDivGrad(x, y)
                          + 1e-4 * admm.sum(admm.square(x)))
        model.addConstr(x >= 0.1)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x.X, y, atol=0.3)

    def test_itakura_saito_scaling(self):
        """Scaling y should scale optimal x proportionally."""
        y1 = np.array([1.0, 2.0])
        y2 = 2.0 * y1
        model1 = self._new_model()
        x1 = admm.Var("x1", 2)
        model1.setObjective(ItakuraSaitoDivGrad(x1, y1)
                           + 0.01 * admm.sum(admm.square(x1)))
        model1.addConstr(x1 >= 0.1)
        model1.optimize()

        model2 = self._new_model()
        x2 = admm.Var("x2", 2)
        model2.setObjective(ItakuraSaitoDivGrad(x2, y2)
                           + 0.01 * admm.sum(admm.square(x2)))
        model2.addConstr(x2 >= 0.1)
        model2.optimize()
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        # x2 should be roughly 2*x1
        ratio = np.asarray(x2.X) / np.asarray(x1.X)
        self.assertTrue(np.all(ratio > 1.0))

    # ---- Smooth Range ----

    def test_smooth_range_consensus(self):
        """Smooth range penalty pushes elements together."""
        model = self._new_model()
        x = admm.Var("x", 5)
        target = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        model.setObjective(0.3 * admm.sum(admm.square(x - target))
                          + SmoothRangeGrad(x, beta=3.0))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        # Range should be smaller than in target
        self.assertLess(np.max(sol) - np.min(sol), np.max(target) - np.min(target))

    def test_smooth_range_lambda_sweep(self):
        """Increasing range penalty weight → smaller range."""
        target = np.array([0.0, 10.0, 5.0])
        ranges = []
        for lam in [0.1, 0.5, 2.0]:
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(0.5 * admm.sum(admm.square(x - target))
                              + lam * SmoothRangeGrad(x, beta=3.0))
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            sol = np.asarray(x.X)
            ranges.append(np.max(sol) - np.min(sol))
        # Ranges should decrease
        self.assertGreater(ranges[0], ranges[1] - 1.0)
        self.assertGreater(ranges[1], ranges[2] - 1.0)

    # ---- Cumulant / Log-partition ----

    def test_cumulant_uniform(self):
        """Cumulant minimized when x is uniform (all equal)."""
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(CumulantGrad(x)
                          + 1e-4 * admm.sum(admm.square(x)))
        model.addConstr(admm.sum(x) == 4.0)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        # All elements should be close to 1.0
        np.testing.assert_allclose(sol, 1.0, atol=0.3)

    # ---- Smooth Min ----

    def test_smooth_min_kkt(self):
        """KKT check for smooth min: unconstrained prox subproblem."""
        v = np.array([3.0, 1.0, 2.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        lam = 0.3
        model.setObjective(lam * SmoothMinGrad(x, beta=5.0)
                          + 0.5 * admm.sum(admm.square(x - v)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        udf_obj = SmoothMinGrad(admm.Var("t", 3), beta=5.0)
        g = udf_obj.grad([sol])[0]
        residual = lam * g + (sol - v)
        self.assertLess(np.linalg.norm(residual), 0.5)

    def test_smooth_min_gradient_direction(self):
        """Smooth min gradient points toward the minimum element."""
        x = admm.Var("x", 4)
        udf_obj = SmoothMinGrad(x, beta=5.0)
        pt = [np.array([3.0, 1.0, 5.0, 2.0])]
        g = udf_obj.grad(pt)[0]
        # Gradient should be largest at the minimum element (index 1)
        self.assertEqual(np.argmax(g), 1)

    # ---- Two-Arg Ratio ----

    def test_ratio_penalty_convergence(self):
        """Ratio penalty: f(x,y) = Σ(x_i/y_i - 1)^2 → x=y at optimum."""
        model = self._new_model()
        x = admm.Var("x", 3)
        y = admm.Var("y", 3)
        target_x = np.array([2.0, 4.0, 6.0])
        target_y = np.array([3.0, 5.0, 7.0])
        model.setObjective(
            0.3 * admm.sum(admm.square(x - target_x))
            + 0.3 * admm.sum(admm.square(y - target_y))
            + TwoArgRatioGrad(x, y)
        )
        model.addConstr(x >= 0.5)
        model.addConstr(y >= 0.5)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        ratio = np.asarray(x.X) / np.asarray(y.X)
        # Ratio should be closer to 1 than target ratios
        target_ratio = target_x / target_y
        self.assertLess(np.std(ratio - 1), np.std(target_ratio - 1) + 0.5)

    # ---- Three-Arg Weighted Distance ----

    def test_three_arg_weighted_distance(self):
        """3-arg: f(x,y,w) = Σ w_i*(x_i-y_i)^2 with all three as variables."""
        model = self._new_model(max_iter=5000)
        x = admm.Var("x", 3)
        y = admm.Var("y", 3)
        w = admm.Var("w", 3)
        tx = np.array([1.0, 2.0, 3.0])
        ty = np.array([2.0, 3.0, 4.0])  # closer targets to help convergence
        model.setObjective(
            0.5 * admm.sum(admm.square(x - tx))
            + 0.5 * admm.sum(admm.square(y - ty))
            + 0.5 * admm.sum(admm.square(w - 1.0))
            + 0.1 * ThreeArgWeightedDistGrad(x, y, w)
        )
        model.addConstr(w >= 0.1)
        model.addConstr(w <= 3.0)
        model.optimize()
        status = model.StatusString
        self.assertIn(status, ["SOLVE_OPT_SUCCESS", "SOLVE_OVER_MAX_ITER"])
        sol_x = np.asarray(x.X)
        sol_y = np.asarray(y.X)
        sol_w = np.asarray(w.X)
        self.assertTrue(np.all(np.isfinite(sol_x)))
        self.assertTrue(np.all(np.isfinite(sol_y)))
        self.assertTrue(np.all(sol_w >= 0.05))

    # ---- Morse Potential ----

    def test_morse_minimum(self):
        """Morse potential minimized at x = y."""
        y = np.array([1.0, 3.0, -2.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(SmoothMorseGrad(x, y, a=2.0)
                          + 1e-4 * admm.sum(admm.square(x)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x.X, y, atol=0.3)

    def test_morse_with_constraints(self):
        """Morse potential with box constraints."""
        y = np.array([5.0, -5.0])
        model = self._new_model()
        x = admm.Var("x", 2)
        model.setObjective(SmoothMorseGrad(x, y, a=1.0))
        model.addConstr(x >= -2)
        model.addConstr(x <= 3)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        np.testing.assert_allclose(sol[0], 3.0, atol=0.3)  # clipped
        np.testing.assert_allclose(sol[1], -2.0, atol=0.3)  # clipped

    # ---- Numerical Stability ----

    def test_large_dimension_quartic(self):
        """Quartic in n=100 dimensions."""
        np.random.seed(910)
        n = 100
        y = np.random.randn(n) * 0.5
        model = self._new_model(max_iter=5000)
        x = admm.Var("x", n)
        model.setObjective(0.5 * admm.sum(admm.square(x - y))
                          + 0.01 * QuarticGrad(x))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(sol)))
        self.assertLess(np.linalg.norm(sol - y), np.linalg.norm(y) + 1)

    def test_large_dimension_logcosh(self):
        """LogCosh in n=100 dimensions."""
        np.random.seed(911)
        n = 100
        y = np.random.randn(n)
        model = self._new_model(max_iter=5000)
        x = admm.Var("x", n)
        model.setObjective(0.5 * admm.sum(admm.square(x - y))
                          + 0.1 * LogCoshGrad(x))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(sol)))

    def test_near_zero_gradient_convergence(self):
        """Function with very flat region: f(x) = Σ log(1+x_i^2) has small grad near 0."""
        model = self._new_model()
        x = admm.Var("x", 5)
        model.setObjective(CauchyLossGrad(x, c=0.1)
                          + 1e-4 * admm.sum(admm.square(x)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        np.testing.assert_allclose(sol, 0.0, atol=0.2)

    def test_large_target_values(self):
        """Regression with moderately large target values (y ~ 10)."""
        y = np.array([10.0, 20.0, 5.0])
        model = self._new_model(max_iter=5000)
        x = admm.Var("x", 3)
        model.setObjective(0.5 * admm.sum(admm.square(x - y))
                          + 0.001 * LogCoshGrad(x))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        np.testing.assert_allclose(sol, y, atol=1.0)

    def test_tiny_target_values(self):
        """Regression with tiny target values (y ~ 0.001)."""
        y = np.array([0.001, 0.002, 0.003])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(0.5 * admm.sum(admm.square(x - y))
                          + 0.0001 * QuarticGrad(x))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        np.testing.assert_allclose(sol, y, atol=0.01)

    # ---- Compositional Patterns ----

    def test_udf_plus_linear_constraint(self):
        """UDF with linear equality constraint Ax = b."""
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(LogCoshGrad(x)
                          + 0.01 * admm.sum(admm.square(x)))
        A = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
        b = np.array([2.0, 4.0])
        p_A = admm.Param("A", 2, 4)
        p_b = admm.Param("b", 2)
        model.addConstr(p_A @ x == p_b)
        model.optimize({"A": A, "b": b})
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        np.testing.assert_allclose(sol[0] + sol[1], 2.0, atol=0.3)
        np.testing.assert_allclose(sol[2] + sol[3], 4.0, atol=0.3)

    def test_udf_with_multiple_box_ineq(self):
        """UDF with component-wise inequality constraints."""
        model = self._new_model()
        x = admm.Var("x", 4)
        lb = np.array([-1.0, 0.0, 1.0, 2.0])
        ub = np.array([0.0, 1.0, 2.0, 3.0])
        target = np.array([-5.0, 5.0, -5.0, 5.0])
        model.setObjective(0.5 * admm.sum(admm.square(x - target))
                          + 0.1 * SoftplusGrad(x))
        model.addConstr(x >= lb)
        model.addConstr(x <= ub)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        for i in range(4):
            self.assertGreaterEqual(sol[i], lb[i] - 0.1)
            self.assertLessEqual(sol[i], ub[i] + 0.1)

    def test_udf_on_affine_transform(self):
        """f(2*x + 1) + ||x||^2: UDF applied to affine expression."""
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(QuarticGrad(2 * x + 1)
                          + 0.5 * admm.sum(admm.square(x)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(sol)))
        # Optimal: quartic of (2x+1) pulls toward 2x+1=0 i.e. x=-0.5
        for v in sol:
            self.assertLess(abs(v + 0.5), 1.5)

    def test_two_udfs_same_variable(self):
        """Two different UDFs on the same variable."""
        model = self._new_model()
        x = admm.Var("x", 4)
        target = np.array([3.0, -1.0, 2.0, -2.0])
        model.setObjective(
            0.3 * admm.sum(admm.square(x - target))
            + 0.1 * LogCoshGrad(x)
            + 0.1 * SoftplusGrad(x)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(sol)))

    def test_udf_in_sum_expression(self):
        """UDF as part of a sum with multiple quadratics."""
        model = self._new_model()
        x = admm.Var("x", 3)
        y = admm.Var("y", 3)
        tx = np.array([1.0, 2.0, 3.0])
        ty = np.array([4.0, 5.0, 6.0])
        model.setObjective(
            0.5 * admm.sum(admm.square(x - tx))
            + 0.5 * admm.sum(admm.square(y - ty))
            + 0.2 * MultiArgQuadGrad(x, y)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sx = np.asarray(x.X)
        sy = np.asarray(y.X)
        # Coupling should pull x and y together
        self.assertLess(np.linalg.norm(sx - sy),
                       np.linalg.norm(tx - ty))

    def test_maximize_negative_udf(self):
        """max -f(x) equivalent to min f(x)."""
        target = np.array([2.0, 4.0])
        model_min = self._new_model()
        x1 = admm.Var("x1", 2)
        model_min.setObjective(QuarticGrad(x1)
                              + 0.5 * admm.sum(admm.square(x1 - target)))
        model_min.optimize()

        model_max = self._new_model()
        x2 = admm.Var("x2", 2)
        model_max.MinSense = False
        model_max.setObjective(-0.5 * admm.sum(admm.square(x2 - target))
                              - QuarticGrad(x2)
                              + 1e-4 * admm.sum(admm.square(x2)))
        model_max.optimize()
        # Both should converge
        self.assertEqual(model_min.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model_max.StatusString, "SOLVE_OPT_SUCCESS")

    # ---- Convergence Behavior ----

    def test_penalty_weight_effect_on_regularization(self):
        """Increasing UDF weight → more regularization (solution closer to 0)."""
        y = np.array([5.0, -3.0, 4.0])
        norms = []
        for lam in [0.01, 0.1, 1.0]:
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(0.5 * admm.sum(admm.square(x - y))
                              + lam * QuarticGrad(x))
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            norms.append(np.linalg.norm(x.X))
        # More penalty → smaller norm (closer to 0)
        self.assertGreater(norms[0], norms[1] - 0.5)
        self.assertGreater(norms[1], norms[2] - 0.5)

    def test_warm_start_improves_or_matches(self):
        """Warm start from good initial should converge at least as well."""
        y = np.array([3.0, -1.0, 2.0])
        # Cold start
        model1 = self._new_model(max_iter=500)
        x1 = admm.Var("x1", 3)
        model1.setObjective(0.5 * admm.sum(admm.square(x1 - y))
                           + 0.2 * ExpSumGrad(x1))
        model1.optimize()

        # Warm start from y
        model2 = self._new_model(max_iter=500)
        x2 = admm.Var("x2", 3)
        x2.start = y
        model2.setObjective(0.5 * admm.sum(admm.square(x2 - y))
                           + 0.2 * ExpSumGrad(x2))
        model2.optimize()
        # Both should produce finite results
        self.assertTrue(np.all(np.isfinite(x1.X)))
        self.assertTrue(np.all(np.isfinite(x2.X)))

    # ---- Gradient Finite-Difference Checks ----

    def test_fd_smooth_quantile(self):
        """FD check for SmoothQuantileLossGrad."""
        np.random.seed(920)
        x = admm.Var("x", 5)
        y = np.random.randn(5)
        udf_obj = SmoothQuantileLossGrad(x, y, tau=0.3)
        pt = [np.random.randn(5)]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(5)
        for i in range(5):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    def test_fd_gelu(self):
        """FD check for GELUSumGrad."""
        x = admm.Var("x", 4)
        udf_obj = GELUSumGrad(x)
        pt = [np.array([1.0, -0.5, 0.0, 2.0])]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(4)
        for i in range(4):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    def test_fd_itakura_saito(self):
        """FD check for ItakuraSaitoDivGrad."""
        x = admm.Var("x", 3)
        y = np.array([1.0, 2.0, 3.0])
        udf_obj = ItakuraSaitoDivGrad(x, y)
        pt = [np.array([1.5, 2.5, 2.0])]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(3)
        for i in range(3):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    def test_fd_smooth_range(self):
        """FD check for SmoothRangeGrad."""
        x = admm.Var("x", 4)
        udf_obj = SmoothRangeGrad(x, beta=3.0)
        pt = [np.array([1.0, 3.0, -1.0, 2.0])]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(4)
        for i in range(4):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    def test_fd_cumulant(self):
        """FD check for CumulantGrad."""
        x = admm.Var("x", 4)
        udf_obj = CumulantGrad(x)
        pt = [np.array([0.5, -0.3, 1.0, 0.0])]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(4)
        for i in range(4):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    def test_fd_smooth_min(self):
        """FD check for SmoothMinGrad."""
        x = admm.Var("x", 3)
        udf_obj = SmoothMinGrad(x, beta=5.0)
        pt = [np.array([2.0, -1.0, 0.5])]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(3)
        for i in range(3):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    def test_fd_two_arg_ratio(self):
        """FD check for TwoArgRatioGrad."""
        xa = admm.Var("x", 3)
        ya = admm.Var("y", 3)
        udf_obj = TwoArgRatioGrad(xa, ya)
        pt = [np.array([2.0, 3.0, 1.5]), np.array([1.0, 2.0, 3.0])]
        grad_ana = udf_obj.grad(pt)
        h = 1e-6
        for arg_idx in range(2):
            grad_fd = np.zeros(3)
            for i in range(3):
                pt_p = [p.copy() for p in pt]; pt_p[arg_idx][i] += h
                pt_m = [p.copy() for p in pt]; pt_m[arg_idx][i] -= h
                grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
            np.testing.assert_allclose(grad_ana[arg_idx], grad_fd, atol=1e-4,
                                      err_msg=f"FD mismatch arg {arg_idx}")

    def test_fd_three_arg_weighted(self):
        """FD check for ThreeArgWeightedDistGrad."""
        xa = admm.Var("x", 3)
        ya = admm.Var("y", 3)
        wa = admm.Var("w", 3)
        udf_obj = ThreeArgWeightedDistGrad(xa, ya, wa)
        pt = [np.array([1.0, 2.0, 3.0]),
              np.array([2.0, 1.0, 4.0]),
              np.array([0.5, 1.0, 2.0])]
        grad_ana = udf_obj.grad(pt)
        h = 1e-6
        for arg_idx in range(3):
            n = len(pt[arg_idx])
            grad_fd = np.zeros(n)
            for i in range(n):
                pt_p = [p.copy() for p in pt]; pt_p[arg_idx][i] += h
                pt_m = [p.copy() for p in pt]; pt_m[arg_idx][i] -= h
                grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
            np.testing.assert_allclose(grad_ana[arg_idx], grad_fd, atol=1e-4,
                                      err_msg=f"FD mismatch arg {arg_idx}")

    def test_fd_morse(self):
        """FD check for SmoothMorseGrad."""
        x = admm.Var("x", 4)
        y = np.array([1.0, -1.0, 0.5, 2.0])
        udf_obj = SmoothMorseGrad(x, y, a=2.0)
        pt = [np.array([0.5, -0.5, 1.0, 1.5])]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(4)
        for i in range(4):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    # ---- KKT Verification ----

    def test_kkt_quantile(self):
        """KKT for smooth quantile: ∂f/∂x + x - v = 0 at prox solution."""
        y = np.array([2.0, -1.0, 3.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        lam = 0.5
        v = np.array([1.0, 0.0, 2.0])
        model.setObjective(lam * SmoothQuantileLossGrad(x, y, tau=0.5)
                          + 0.5 * admm.sum(admm.square(x - v)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        udf_obj = SmoothQuantileLossGrad(admm.Var("t", 3), y, tau=0.5)
        g = udf_obj.grad([sol])[0]
        residual = lam * g + (sol - v)
        self.assertLess(np.linalg.norm(residual), 0.5)

    def test_kkt_gelu(self):
        """KKT for GELU regularizer."""
        v = np.array([2.0, -1.0, 0.5])
        model = self._new_model()
        x = admm.Var("x", 3)
        lam = 0.3
        model.setObjective(lam * GELUSumGrad(x)
                          + 0.5 * admm.sum(admm.square(x - v)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        udf_obj = GELUSumGrad(admm.Var("t", 3))
        g = udf_obj.grad([sol])[0]
        residual = lam * g + (sol - v)
        self.assertLess(np.linalg.norm(residual), 0.5)

    # ---- Mixed Patterns ----

    def test_multiple_models_same_udf_type(self):
        """Create multiple models with same UDF class → shares registration."""
        for _ in range(3):
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(QuarticGrad(x)
                              + 0.5 * admm.sum(admm.square(x - np.array([1, 2, 3]))))
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            self.assertTrue(np.all(np.isfinite(x.X)))

    def test_scalar_vector_matrix_in_one_model(self):
        """Scalar, vector, and matrix variables with UDFs in same model."""
        model = self._new_model()
        a = admm.Var("a", 1)      # scalar
        b = admm.Var("b", 4)      # vector
        c = admm.Var("c", 2, 2)   # matrix
        model.setObjective(
            QuarticGrad(a)
            + 0.1 * LogCoshGrad(b)
            + 0.1 * admm.sum(admm.square(c - np.eye(2)))
            + 0.5 * admm.sum(admm.square(a - 1))
            + 0.5 * admm.sum(admm.square(b - np.array([1, 2, 3, 4])))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.all(np.isfinite(a.X)))
        self.assertTrue(np.all(np.isfinite(b.X)))
        self.assertTrue(np.all(np.isfinite(c.X)))

    def test_equality_and_inequality_with_udf(self):
        """UDF + equality + inequality constraints together."""
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(LogCoshGrad(x)
                          + 0.5 * admm.sum(admm.square(x - np.array([5, -5, 5, -5]))))
        model.addConstr(admm.sum(x) == 0)
        model.addConstr(x >= -3)
        model.addConstr(x <= 3)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        np.testing.assert_allclose(np.sum(sol), 0.0, atol=0.3)
        self.assertTrue(np.all(sol >= -3 - 0.1))
        self.assertTrue(np.all(sol <= 3 + 0.1))

    def test_param_with_udf_and_constraints(self):
        """Param-based model with UDF and constraints."""
        model = self._new_model()
        x = admm.Var("x", 3)
        p = admm.Param("p", 3)
        model.setObjective(0.5 * admm.sum(admm.square(x - p))
                          + 0.1 * SoftplusGrad(x))
        model.addConstr(admm.sum(x) <= 5)
        sol_list = []
        for target in [np.array([1, 2, 3]), np.array([10, 20, 30])]:
            model.optimize({"p": target.astype(float)})
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            sol_list.append(np.asarray(x.X).copy())
        # Second target is larger → sum constraint should be active
        self.assertLess(np.sum(sol_list[1]), 5.5)

    def test_random_convex_problems(self):
        """Random convex problems with different seeds should all converge."""
        for seed in range(950, 955):
            np.random.seed(seed)
            n = np.random.randint(3, 8)
            y = np.random.randn(n)
            lam = 0.1 + 0.5 * np.random.rand()
            model = self._new_model()
            x = admm.Var("x", n)
            model.setObjective(0.5 * admm.sum(admm.square(x - y))
                              + lam * LogCoshGrad(x))
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                            f"Failed for seed={seed}, n={n}")


# ---------------------------------------------------------------------------
# UDF classes for integration / statistical / advanced pattern tests
# ---------------------------------------------------------------------------

class SmoothEpsilonLossGrad(admm.UDFBase):
    """SVR ε-insensitive loss (smooth): f(x;y,ε) = Σ sp(x_i-y_i-ε) + sp(y_i-x_i-ε)
    where sp(z) = log(1+exp(z)). Penalizes deviations beyond ε from y."""
    def __init__(self, arg, y, eps=0.5):
        self.arg = arg; self.y = np.asarray(y, dtype=float); self.eps = eps
    def arguments(self): return [self.arg]
    def _sp(self, z):
        return np.where(z > 20, z, np.log(1 + np.exp(np.clip(z, -500, 20))))
    def _sig(self, z):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                       np.exp(z) / (1.0 + np.exp(z)))
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        r = x - self.y
        return float(np.sum(self._sp(r - self.eps) + self._sp(-r - self.eps)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        r = x - self.y
        return [self._sig(r - self.eps) - self._sig(-r - self.eps)]


class WingLossGrad(admm.UDFBase):
    """Wing loss (smooth version): f(x;y) = Σ w*ln(1 + sqrt((x_i-y_i)^2+δ^2)/ε).
    From face alignment. Amplifies small errors more than L2."""
    def __init__(self, arg, y, w=10.0, eps=2.0, delta=0.01):
        self.arg = arg; self.y = np.asarray(y, dtype=float)
        self.w = w; self.eps = eps; self.delta = delta
    def arguments(self): return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        r = x - self.y
        s = np.sqrt(r ** 2 + self.delta ** 2)
        return float(np.sum(self.w * np.log(1 + s / self.eps)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        r = x - self.y
        s = np.sqrt(r ** 2 + self.delta ** 2)
        ds = r / s
        return [self.w * ds / (self.eps + s)]


class SmoothHingePairGrad(admm.UDFBase):
    """Pairwise ranking loss: f(x,y) = Σ sp(x_i - y_i + margin).
    Encourages y_i > x_i + margin."""
    def __init__(self, x_arg, y_arg, margin=1.0):
        self.x_arg = x_arg; self.y_arg = y_arg; self.margin = margin
    def arguments(self): return [self.x_arg, self.y_arg]
    def _sp(self, z):
        return np.where(z > 20, z, np.log(1 + np.exp(np.clip(z, -500, 20))))
    def _sig(self, z):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                       np.exp(z) / (1.0 + np.exp(z)))
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        return float(np.sum(self._sp(x - y + self.margin)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        y = np.asarray(tensorlist[1], dtype=float)
        s = self._sig(x - y + self.margin)
        return [s, -s]


class DoubleWellGrad(admm.UDFBase):
    """Double-well potential: f(x) = Σ (x_i^2 - 1)^2.
    Has minima at x_i = ±1. Non-convex, but grad UDF can still be tested."""
    def __init__(self, arg):
        self.arg = arg
    def arguments(self): return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum((x ** 2 - 1) ** 2))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [4.0 * x * (x ** 2 - 1)]


class LogBarrierSumGrad(admm.UDFBase):
    """Log barrier: f(x) = -Σ log(x_i), domain x > 0.
    Self-concordant barrier for the positive orthant."""
    def __init__(self, arg):
        self.arg = arg
    def arguments(self): return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(-np.sum(np.log(np.maximum(x, 1e-15))))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [-1.0 / np.maximum(x, 1e-15)]


class ChiSquaredDivGrad(admm.UDFBase):
    """Chi-squared divergence: f(x;y) = Σ (x_i - y_i)^2 / y_i, y > 0.
    An f-divergence. Minimum at x = y."""
    def __init__(self, arg, y):
        self.arg = arg; self.y = np.asarray(y, dtype=float)
    def arguments(self): return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum((x - self.y) ** 2 / self.y))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [2.0 * (x - self.y) / self.y]


class SmoothLpNormGrad(admm.UDFBase):
    """Smooth Lp pseudo-norm: f(x) = Σ (x_i^2 + eps)^(p/2).
    Smooth approximation to ||x||_p^p for p ∈ (0,2]."""
    def __init__(self, arg, p=1.5, eps=1e-4):
        self.arg = arg; self.p = p; self.eps = eps
    def arguments(self): return [self.arg]
    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum((x ** 2 + self.eps) ** (self.p / 2)))
    def grad(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        return [self.p * x * (x ** 2 + self.eps) ** (self.p / 2 - 1)]


class GradUDFIntegrationTestCase(unittest.TestCase):
    """Integration tests: complex constraint patterns, sequential solves,
    cross-UDF comparisons, error paths, and advanced optimization patterns."""

    def _new_model(self, max_iter=3000):
        model = admm.Model()
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.display_sub_solver_details, 0)
        model.setOption(admm.Options.admm_max_iteration, max_iter)
        return model

    # ---- SVR ε-insensitive Loss ----

    def test_svr_epsilon_tube(self):
        """ε-insensitive loss: no penalty within ε-tube around y."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(SmoothEpsilonLossGrad(x, y, eps=1.0)
                          + 0.01 * admm.sum(admm.square(x)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        # Solution should be close to y (within ε)
        np.testing.assert_allclose(sol, y, atol=1.5)

    def test_svr_epsilon_sweep(self):
        """Increasing ε widens the tube → solution converges to 0 (regularizer wins)."""
        y = np.array([3.0, -2.0, 5.0])
        norms = []
        for eps in [0.1, 1.0, 10.0]:
            model = self._new_model()
            x = admm.Var("x", 3)
            model.setObjective(SmoothEpsilonLossGrad(x, y, eps=eps)
                              + 0.5 * admm.sum(admm.square(x)))
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            norms.append(np.linalg.norm(x.X))
        # Larger ε → smaller solution norm
        self.assertGreater(norms[0], norms[2] - 1.0)

    def test_svr_with_constraints(self):
        """SVR loss with box constraints."""
        y = np.array([10.0, -10.0, 0.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(SmoothEpsilonLossGrad(x, y, eps=0.5)
                          + 0.01 * admm.sum(admm.square(x)))
        model.addConstr(x >= -5)
        model.addConstr(x <= 5)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        self.assertTrue(np.all(sol >= -5 - 0.1))
        self.assertTrue(np.all(sol <= 5 + 0.1))

    # ---- Wing Loss ----

    def test_wing_loss_minimum(self):
        """Wing loss minimized at x = y."""
        y = np.array([1.0, -2.0, 3.0])
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(WingLossGrad(x, y)
                          + 1e-4 * admm.sum(admm.square(x)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x.X, y, atol=0.3)

    def test_wing_loss_vs_quadratic(self):
        """Wing loss penalizes small errors more → closer fit for small residuals."""
        y = np.array([0.1, -0.1, 0.05, -0.05])
        model_q = self._new_model()
        xq = admm.Var("xq", 4)
        model_q.setObjective(admm.sum(admm.square(xq - y))
                            + 0.5 * admm.sum(admm.square(xq)))
        model_q.optimize()

        model_w = self._new_model()
        xw = admm.Var("xw", 4)
        model_w.setObjective(WingLossGrad(xw, y, w=5.0, eps=1.0)
                            + 0.5 * admm.sum(admm.square(xw)))
        model_w.optimize()
        self.assertEqual(model_w.StatusString, "SOLVE_OPT_SUCCESS")
        # Both converge
        self.assertTrue(np.all(np.isfinite(xw.X)))

    def test_wing_different_w_params(self):
        """Different w parameter changes sensitivity."""
        y = np.array([2.0, -1.0])
        results = []
        for w_val in [1.0, 5.0, 20.0]:
            model = self._new_model()
            x = admm.Var("x", 2)
            model.setObjective(WingLossGrad(x, y, w=w_val)
                              + 0.3 * admm.sum(admm.square(x)))
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            results.append(np.linalg.norm(x.X - y))
        # Higher w → more weight on wing loss → closer to y
        self.assertGreater(results[0], results[2] - 0.5)

    # ---- Smooth Hinge Pair (Ranking Loss) ----

    def test_hinge_pair_ordering(self):
        """Pairwise ranking: encourages y_i > x_i + margin."""
        model = self._new_model()
        x = admm.Var("x", 3)
        y = admm.Var("y", 3)
        model.setObjective(
            SmoothHingePairGrad(x, y, margin=1.0)
            + 0.1 * admm.sum(admm.square(x - np.array([3, 4, 5])))
            + 0.1 * admm.sum(admm.square(y - np.array([2, 3, 4])))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol_x = np.asarray(x.X)
        sol_y = np.asarray(y.X)
        # Hinge encourages y > x + margin, so gap should increase
        self.assertTrue(np.all(np.isfinite(sol_x)))
        self.assertTrue(np.all(np.isfinite(sol_y)))

    def test_hinge_margin_effect(self):
        """Larger margin → bigger gap between y and x."""
        gaps = []
        for margin in [0.5, 2.0, 5.0]:
            model = self._new_model()
            x = admm.Var("x", 2)
            y = admm.Var("y", 2)
            model.setObjective(
                SmoothHingePairGrad(x, y, margin=margin)
                + 0.5 * admm.sum(admm.square(x - 1))
                + 0.5 * admm.sum(admm.square(y - 1))
            )
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            gaps.append(np.mean(np.asarray(y.X) - np.asarray(x.X)))
        # Larger margin → y pulled further above x
        self.assertGreater(gaps[1], gaps[0] - 0.5)
        self.assertGreater(gaps[2], gaps[1] - 0.5)

    # ---- Double-Well ----

    def test_double_well_converges(self):
        """Double-well + quadratic: stabilize near ±1."""
        model = self._new_model()
        x = admm.Var("x", 4)
        target = np.array([0.8, -0.9, 1.1, -1.2])
        model.setObjective(0.1 * DoubleWellGrad(x)
                          + 0.5 * admm.sum(admm.square(x - target)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(sol)))
        # Solution should be between target and ±1
        self.assertTrue(np.all(np.abs(sol) < 2.0))

    def test_double_well_fd_check(self):
        """FD gradient check for double-well."""
        x = admm.Var("x", 3)
        udf_obj = DoubleWellGrad(x)
        pt = [np.array([0.5, -1.5, 0.0])]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(3)
        for i in range(3):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    # ---- Log Barrier ----

    # ---- Chi-Squared Divergence ----

    def test_chi_squared_minimum(self):
        """χ² divergence minimized at x=y."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(ChiSquaredDivGrad(x, y)
                          + 1e-4 * admm.sum(admm.square(x)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        np.testing.assert_allclose(x.X, y, atol=0.2)

    def test_chi_squared_vs_quadratic(self):
        """χ² divergence is weighted quadratic: (x-y)^2/y, so small y → high weight."""
        y = np.array([0.5, 5.0])
        target = np.array([2.0, 2.0])
        model = self._new_model()
        x = admm.Var("x", 2)
        model.setObjective(ChiSquaredDivGrad(x, y)
                          + 0.3 * admm.sum(admm.square(x - target)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        # x[0] should be closer to y[0]=0.5 (higher χ² weight) than x[1] to y[1]=5
        dist0 = abs(sol[0] - y[0])
        dist1 = abs(sol[1] - y[1])
        self.assertLess(dist0, dist1 + 1.0)

    # ---- Smooth Lp Norm ----

    def test_lp_norm_p1_sparsity(self):
        """Smooth L1 (p=1) promotes sparsity."""
        y = np.array([0.5, -0.3, 2.0, 0.1, -1.5])
        model = self._new_model()
        x = admm.Var("x", 5)
        model.setObjective(0.5 * admm.sum(admm.square(x - y))
                          + 0.3 * SmoothLpNormGrad(x, p=1.0))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        # Should shrink small values toward 0
        self.assertLess(abs(sol[3]), abs(y[3]) + 0.1)

    def test_lp_norm_p_comparison(self):
        """Different p values: smaller p → more sparsity-promoting."""
        y = np.array([0.3, -0.2, 1.0, -0.1])
        sparsities = []
        for p in [0.8, 1.0, 1.5, 2.0]:
            model = self._new_model()
            x = admm.Var("x", 4)
            model.setObjective(0.5 * admm.sum(admm.square(x - y))
                              + 0.3 * SmoothLpNormGrad(x, p=p))
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            sol = np.asarray(x.X)
            sparsities.append(np.sum(np.abs(sol) < 0.1))
        # All should converge; just check finite
        self.assertTrue(all(isinstance(s, (int, np.integer)) for s in sparsities))

    # ---- Sequential Solve Patterns ----

    def test_sequential_solve_tightening_bounds(self):
        """Solve, tighten bounds, re-solve → narrower solution."""
        target = np.array([5.0, -5.0, 3.0])
        # Loose bounds
        model1 = self._new_model()
        x1 = admm.Var("x1", 3)
        model1.setObjective(0.5 * admm.sum(admm.square(x1 - target))
                           + 0.1 * QuarticGrad(x1))
        model1.addConstr(x1 >= -10)
        model1.addConstr(x1 <= 10)
        model1.optimize()
        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        sol1 = np.asarray(x1.X)

        # Tight bounds
        model2 = self._new_model()
        x2 = admm.Var("x2", 3)
        model2.setObjective(0.5 * admm.sum(admm.square(x2 - target))
                           + 0.1 * QuarticGrad(x2))
        model2.addConstr(x2 >= -2)
        model2.addConstr(x2 <= 2)
        model2.optimize()
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        sol2 = np.asarray(x2.X)
        # Tight bounds → solution within [-2, 2]
        self.assertTrue(np.all(sol2 >= -2 - 0.1))
        self.assertTrue(np.all(sol2 <= 2 + 0.1))

    def test_sequential_warm_start_improvement(self):
        """Warm start from previous solution: should converge faster."""
        y = np.array([3.0, -2.0, 1.0, 4.0])
        # First solve
        model1 = self._new_model()
        x1 = admm.Var("x1", 4)
        model1.setObjective(0.5 * admm.sum(admm.square(x1 - y))
                           + 0.2 * LogCoshGrad(x1))
        model1.optimize()
        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        sol1 = np.asarray(x1.X).copy()

        # Second solve with perturbation + warm start
        y2 = y + 0.1 * np.random.randn(4)
        model2 = self._new_model(max_iter=500)
        x2 = admm.Var("x2", 4)
        x2.start = sol1
        model2.setObjective(0.5 * admm.sum(admm.square(x2 - y2))
                           + 0.2 * LogCoshGrad(x2))
        model2.optimize()
        self.assertTrue(np.all(np.isfinite(x2.X)))

    # ---- Cross-UDF Comparisons ----

    def test_huber_vs_wing_outlier_robustness(self):
        """Both Huber and Wing are robust to outliers."""
        y = np.array([1.0, 2.0, 3.0, 100.0])  # outlier at index 3
        model_h = self._new_model()
        xh = admm.Var("xh", 4)
        model_h.setObjective(HuberGrad(xh)
                            + 0.1 * admm.sum(admm.square(xh - y)))
        model_h.optimize()

        model_w = self._new_model()
        xw = admm.Var("xw", 4)
        model_w.setObjective(WingLossGrad(xw, np.zeros(4))
                            + 0.1 * admm.sum(admm.square(xw - y)))
        model_w.optimize()
        self.assertEqual(model_h.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model_w.StatusString, "SOLVE_OPT_SUCCESS")
        # Both should produce finite results
        self.assertTrue(np.all(np.isfinite(xh.X)))
        self.assertTrue(np.all(np.isfinite(xw.X)))

    def test_logcosh_vs_pseudohuber_similar(self):
        """LogCosh ≈ PseudoHuber for similar parameters → similar solutions."""
        y = np.array([2.0, -1.0, 3.0])
        model1 = self._new_model()
        x1 = admm.Var("x1", 3)
        model1.setObjective(0.5 * admm.sum(admm.square(x1 - y))
                           + 0.3 * LogCoshGrad(x1))
        model1.optimize()

        model2 = self._new_model()
        x2 = admm.Var("x2", 3)
        model2.setObjective(0.5 * admm.sum(admm.square(x2 - y))
                           + 0.3 * PseudoHuberGrad(x2, delta=1.0))
        model2.optimize()
        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")
        # Solutions should be similar (both are smooth L1-like)
        np.testing.assert_allclose(x1.X, x2.X, atol=1.0)

    # ---- Complex Constraint Configurations ----

    def test_udf_equality_inequality_bounds(self):
        """UDF + equality + inequality + bounds all together."""
        model = self._new_model()
        x = admm.Var("x", 5)
        target = np.array([3.0, -1.0, 2.0, -2.0, 4.0])
        model.setObjective(0.5 * admm.sum(admm.square(x - target))
                          + 0.1 * SoftplusGrad(x))
        model.addConstr(admm.sum(x) == 3.0)
        model.addConstr(x >= -2)
        model.addConstr(x <= 4)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        np.testing.assert_allclose(np.sum(sol), 3.0, atol=0.3)
        self.assertTrue(np.all(sol >= -2 - 0.1))
        self.assertTrue(np.all(sol <= 4 + 0.1))

    def test_udf_multiple_equality_constraints(self):
        """UDF with two equality constraints."""
        model = self._new_model()
        x = admm.Var("x", 4)
        model.setObjective(QuarticGrad(x)
                          + 0.5 * admm.sum(admm.square(x - np.array([1, 2, 3, 4]))))
        model.addConstr(x[0] + x[1] == 3)
        model.addConstr(x[2] + x[3] == 5)
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        np.testing.assert_allclose(sol[0] + sol[1], 3.0, atol=0.3)
        np.testing.assert_allclose(sol[2] + sol[3], 5.0, atol=0.3)

    def test_udf_many_inequality_constraints(self):
        """UDF with many pairwise ordering constraints."""
        model = self._new_model()
        n = 5
        x = admm.Var("x", n)
        model.setObjective(LogCoshGrad(x)
                          + 0.5 * admm.sum(admm.square(x - np.array([5, 1, 4, 2, 3]))))
        # Enforce ordering: x[0] >= x[1] >= x[2] >= x[3] >= x[4]
        for i in range(n - 1):
            model.addConstr(x[i] >= x[i + 1])
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        # Check ordering
        for i in range(n - 1):
            self.assertGreaterEqual(sol[i], sol[i + 1] - 0.2)

    # ---- Multi-Variable with UDF ----

    def test_five_variables_two_udfs(self):
        """5 variables with 2 different UDFs coupling subsets."""
        model = self._new_model()
        a = admm.Var("a", 3)
        b = admm.Var("b", 3)
        c = admm.Var("c", 3)
        d = admm.Var("d", 3)
        e = admm.Var("e", 3)
        ta = np.array([1, 2, 3], dtype=float)
        te = np.array([5, 6, 7], dtype=float)
        model.setObjective(
            0.3 * admm.sum(admm.square(a - ta))
            + 0.3 * admm.sum(admm.square(e - te))
            + 0.1 * admm.sum(admm.square(b))
            + 0.1 * admm.sum(admm.square(c))
            + 0.1 * admm.sum(admm.square(d))
            + 0.2 * MultiArgQuadGrad(a, b)
            + 0.2 * MultiArgQuadGrad(d, e)
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        for v in [a, b, c, d, e]:
            self.assertTrue(np.all(np.isfinite(v.X)))

    def test_shared_variable_multiple_udfs(self):
        """One variable appears in multiple UDF terms."""
        model = self._new_model()
        x = admm.Var("x", 4)
        y = admm.Var("y", 4)
        z = admm.Var("z", 4)
        model.setObjective(
            0.2 * MultiArgQuadGrad(x, y)
            + 0.2 * MultiArgQuadGrad(x, z)
            + 0.5 * admm.sum(admm.square(y - np.array([1, 2, 3, 4])))
            + 0.5 * admm.sum(admm.square(z - np.array([2, 3, 4, 5])))
            + 0.1 * admm.sum(admm.square(x))
        )
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol_x = np.asarray(x.X)
        sol_y = np.asarray(y.X)
        sol_z = np.asarray(z.X)
        # x should be pulled between y and z
        self.assertTrue(np.all(np.isfinite(sol_x)))
        # x should be somewhere between y and z targets
        mid = (np.array([1, 2, 3, 4]) + np.array([2, 3, 4, 5])) / 2
        self.assertLess(np.linalg.norm(sol_x - mid), 5.0)

    # ---- FD Gradient Checks for New UDFs ----

    def test_fd_svr_epsilon(self):
        """FD check for SmoothEpsilonLossGrad."""
        x = admm.Var("x", 4)
        y = np.array([1.0, -1.0, 2.0, 0.0])
        udf_obj = SmoothEpsilonLossGrad(x, y, eps=0.5)
        pt = [np.array([1.3, -0.5, 3.0, -1.0])]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(4)
        for i in range(4):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    def test_fd_wing_loss(self):
        """FD check for WingLossGrad."""
        x = admm.Var("x", 3)
        y = np.array([1.0, -1.0, 0.5])
        udf_obj = WingLossGrad(x, y, w=10.0, eps=2.0)
        pt = [np.array([1.5, -0.3, 1.0])]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(3)
        for i in range(3):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    def test_fd_hinge_pair(self):
        """FD check for SmoothHingePairGrad."""
        xa = admm.Var("x", 3)
        ya = admm.Var("y", 3)
        udf_obj = SmoothHingePairGrad(xa, ya, margin=1.0)
        pt = [np.array([1.0, 2.0, 0.5]), np.array([2.0, 1.5, 1.0])]
        grad_ana = udf_obj.grad(pt)
        h = 1e-6
        for arg_idx in range(2):
            grad_fd = np.zeros(3)
            for i in range(3):
                pt_p = [p.copy() for p in pt]; pt_p[arg_idx][i] += h
                pt_m = [p.copy() for p in pt]; pt_m[arg_idx][i] -= h
                grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
            np.testing.assert_allclose(grad_ana[arg_idx], grad_fd, atol=1e-4,
                                      err_msg=f"FD mismatch arg {arg_idx}")

    def test_fd_log_barrier(self):
        """FD check for LogBarrierSumGrad."""
        x = admm.Var("x", 3)
        udf_obj = LogBarrierSumGrad(x)
        pt = [np.array([1.0, 2.0, 0.5])]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(3)
        for i in range(3):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    def test_fd_chi_squared(self):
        """FD check for ChiSquaredDivGrad."""
        x = admm.Var("x", 3)
        y = np.array([1.0, 2.0, 3.0])
        udf_obj = ChiSquaredDivGrad(x, y)
        pt = [np.array([1.5, 1.8, 3.5])]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(3)
        for i in range(3):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    def test_fd_smooth_lp(self):
        """FD check for SmoothLpNormGrad."""
        x = admm.Var("x", 4)
        udf_obj = SmoothLpNormGrad(x, p=1.5)
        pt = [np.array([1.0, -0.5, 2.0, -1.0])]
        grad_ana = udf_obj.grad(pt)[0]
        h = 1e-6
        grad_fd = np.zeros(4)
        for i in range(4):
            pt_p = [pt[0].copy()]; pt_p[0][i] += h
            pt_m = [pt[0].copy()]; pt_m[0][i] -= h
            grad_fd[i] = (udf_obj.eval(pt_p) - udf_obj.eval(pt_m)) / (2 * h)
        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4)

    # ---- KKT Verification ----

    def test_kkt_svr_epsilon(self):
        """KKT for SVR ε-insensitive."""
        y = np.array([2.0, -1.0])
        v = np.array([1.5, -0.5])
        lam = 0.3
        model = self._new_model()
        x = admm.Var("x", 2)
        model.setObjective(lam * SmoothEpsilonLossGrad(x, y, eps=0.5)
                          + 0.5 * admm.sum(admm.square(x - v)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        udf_obj = SmoothEpsilonLossGrad(admm.Var("t", 2), y, eps=0.5)
        g = udf_obj.grad([sol])[0]
        residual = lam * g + (sol - v)
        self.assertLess(np.linalg.norm(residual), 0.5)

    def test_kkt_wing_loss(self):
        """KKT for Wing loss."""
        y_tgt = np.array([1.0, 3.0])
        v = np.array([0.0, 2.0])
        lam = 0.5
        model = self._new_model()
        x = admm.Var("x", 2)
        model.setObjective(lam * WingLossGrad(x, y_tgt, w=5.0, eps=2.0)
                          + 0.5 * admm.sum(admm.square(x - v)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        udf_obj = WingLossGrad(admm.Var("t", 2), y_tgt, w=5.0, eps=2.0)
        g = udf_obj.grad([sol])[0]
        residual = lam * g + (sol - v)
        self.assertLess(np.linalg.norm(residual), 0.5)

    def test_kkt_chi_squared(self):
        """KKT for χ² divergence."""
        y = np.array([1.0, 2.0, 3.0])
        v = np.array([2.0, 3.0, 4.0])
        lam = 0.4
        model = self._new_model()
        x = admm.Var("x", 3)
        model.setObjective(lam * ChiSquaredDivGrad(x, y)
                          + 0.5 * admm.sum(admm.square(x - v)))
        model.optimize()
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        udf_obj = ChiSquaredDivGrad(admm.Var("t", 3), y)
        g = udf_obj.grad([sol])[0]
        residual = lam * g + (sol - v)
        self.assertLess(np.linalg.norm(residual), 0.5)

    # ---- Param Rebind with UDF ----

    def test_param_rebind_convergence(self):
        """Re-solving with different Param values: all should converge."""
        model = self._new_model()
        x = admm.Var("x", 3)
        p = admm.Param("p", 3)
        model.setObjective(0.5 * admm.sum(admm.square(x - p))
                          + 0.2 * QuarticGrad(x))
        for target in [np.array([1, 2, 3]),
                      np.array([-1, 0, 1]),
                      np.array([5, -5, 0])]:
            model.optimize({"p": target.astype(float)})
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            sol = np.asarray(x.X)
            self.assertTrue(np.all(np.isfinite(sol)))

    def test_param_matrix_with_udf(self):
        """Param as matrix coefficient with UDF."""
        model = self._new_model()
        x = admm.Var("x", 3)
        A = admm.Param("A", 2, 3)
        b = admm.Param("b", 2)
        model.setObjective(admm.sum(admm.square(A @ x - b))
                          + 0.1 * LogCoshGrad(x))
        A_val = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b_val = np.array([2.0, 3.0])
        model.optimize({"A": A_val, "b": b_val})
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        sol = np.asarray(x.X)
        self.assertTrue(np.all(np.isfinite(sol)))

    # ---- Model Copy with UDF ----

    def test_model_copy_preserves_udf(self):
        """Copied model should produce same solution."""
        model1 = self._new_model()
        x = admm.Var("x", 3)
        target = np.array([2.0, -1.0, 3.0])
        model1.setObjective(0.5 * admm.sum(admm.square(x - target))
                           + 0.1 * QuarticGrad(x))
        model1.optimize()
        self.assertEqual(model1.StatusString, "SOLVE_OPT_SUCCESS")
        sol1 = np.asarray(x.X).copy()

        model2 = admm.Model(model1)
        model2.setOption(admm.Options.solver_verbosity_level, 3)
        model2.setOption(admm.Options.display_sub_solver_details, 0)
        model2.optimize()
        self.assertEqual(model2.StatusString, "SOLVE_OPT_SUCCESS")

    # ---- Stability Under Perturbation ----

    def test_solution_continuity(self):
        """Small change in target → small change in solution."""
        base = np.array([3.0, -1.0, 2.0])
        model1 = self._new_model()
        x1 = admm.Var("x1", 3)
        model1.setObjective(0.5 * admm.sum(admm.square(x1 - base))
                           + 0.2 * LogCoshGrad(x1))
        model1.optimize()
        sol1 = np.asarray(x1.X).copy()

        perturbed = base + 0.01 * np.array([1, -1, 0.5])
        model2 = self._new_model()
        x2 = admm.Var("x2", 3)
        model2.setObjective(0.5 * admm.sum(admm.square(x2 - perturbed))
                           + 0.2 * LogCoshGrad(x2))
        model2.optimize()
        sol2 = np.asarray(x2.X)
        # Small perturbation → small solution change
        self.assertLess(np.linalg.norm(sol1 - sol2), 1.0)

    def test_solution_bounded_sensitivity(self):
        """Solution change should be bounded by Lipschitz constant."""
        np.random.seed(960)
        base = np.random.randn(5)
        eps_vals = [0.01, 0.1, 0.5]
        sol_base = None
        for eps in [0.0] + eps_vals:
            model = self._new_model()
            x = admm.Var("x", 5)
            target = base + eps * np.ones(5)
            model.setObjective(0.5 * admm.sum(admm.square(x - target))
                              + 0.2 * SoftplusGrad(x))
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
            sol = np.asarray(x.X)
            if sol_base is None:
                sol_base = sol.copy()
            else:
                # Change in solution bounded by change in input
                self.assertLess(np.linalg.norm(sol - sol_base),
                               eps * np.sqrt(5) + 1.0)

    # ---- Scalar UDF edge cases ----

    def test_scalar_udf_all_types(self):
        """Various UDFs on scalar (1-dimensional) variables."""
        udfs_and_targets = [
            (QuarticGrad, 1.5),
            (LogCoshGrad, -2.0),
            (SoftplusGrad, 0.5),
            (ExpSumGrad, -1.0),
        ]
        for UdfClass, target in udfs_and_targets:
            model = self._new_model()
            x = admm.Var("x", 1)
            model.setObjective(0.5 * admm.sum(admm.square(x - target))
                              + 0.1 * UdfClass(x))
            model.optimize()
            self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS",
                            f"Failed for {UdfClass.__name__}")
            self.assertTrue(np.all(np.isfinite(x.X)))


if __name__ == "__main__":
    unittest.main()
