import admm
import numpy as np
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import udf


class ASTProblemTestCase(unittest.TestCase):

    def _new_model(self):
        model = admm.Model()
        model.setOption(admm.Options.solver_verbosity_level, 3)
        model.setOption(admm.Options.display_sub_solver_details, 0)
        return model

    # min   0.5 * ||x - y||_2^2 + lam * ||x||_4^4
    #
    # where  ||x||_4^4 = sum_i x_i^4
    def test_user_defined_quartic_penalty(self):
        observed_value = np.array([2.0, -1.0, 0.5])
        penalty_weight = 0.1

        model = self._new_model()
        x = admm.Var("x", len(observed_value))

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value))
            + penalty_weight * udf.QuarticPenalty(x)
        )

        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   0.5 * ||x - y||_2^2 + lam * ||x||_0
    # s.t.  0 <= x <= 1
    #
    # where  ||x||_0 = #{i : x_i != 0}
    def test_user_defined_l0_penalty_vector(self):
        observed_values = np.array([0.2, 1.7, 0.6, 1.9])
        penalty_weight = 1.0
        expected_x = np.array([0.0, 1.0, 0.0, 1.0])
        expected_obj = 2.85

        model = self._new_model()
        x = admm.Var("x", len(observed_values))

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_values))
            + penalty_weight * udf.L0Norm(x)
        )
        model.addConstr(x >= 0)
        model.addConstr(x <= 1)
        model.setOption(admm.Options.admm_max_iteration, 10000)

        model.optimize()

        x_value = np.asarray(x.X)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.allclose(x_value, expected_x, atol=1e-2))
        self.assertLessEqual(np.max(x_value), 1.0 + 1e-6)
        self.assertLess(abs(model.ObjVal - expected_obj), 1e-2)

    # min   0.5 * ||X - Y||_F^2 + G(X)
    #
    # where  G(X) = #{j : ||X[:,j]||_2 > 0}  (number of nonzero columns)
    def test_user_defined_group_sparsity_penalty(self):
        observed_value = np.array([[0.2, 2.0, 0.3], [0.1, 1.0, 0.4]])
        expected_x = np.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]])
        expected_obj = 1.15

        model = self._new_model()
        x = admm.Var("x", 2, 3)

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value))
            + udf.GroupSparsityPenalty(x)
        )

        model.optimize()

        x_value = np.asarray(x.X)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.allclose(x_value, expected_x, atol=1e-2))
        self.assertLess(abs(model.ObjVal - expected_obj), 1e-2)

    # min   0.5 * ||X - Y||_F^2 + lam * rank(X)
    def test_user_defined_matrix_rank_penalty(self):
        observed_value = np.array([[2.0, 0.0], [0.0, 0.5]])
        penalty_weight = 0.5
        expected_x = np.array([[2.0, 0.0], [0.0, 0.0]])
        expected_obj = 0.625

        model = self._new_model()
        x = admm.Var("x", 2, 2)

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value))
            + penalty_weight * udf.RankPenalty(x)
        )

        model.optimize()

        x_value = np.asarray(x.X)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.allclose(x_value, expected_x, atol=1e-2))
        self.assertLess(abs(model.ObjVal - expected_obj), 1e-2)

    # min   0.5 * ||x - y||_2^2 + delta_{S^{n-1}}(x)
    #
    # where  S^{n-1} = {x in R^n : ||x||_2 = 1}
    def test_user_defined_unit_sphere_indicator(self):
        cases = [
            (np.array([3.0, 4.0]), np.array([0.6, 0.8]), 8.0),
            (np.array([0.1, 0.0]), np.array([1.0, 0.0]), 0.405),
        ]

        for observed_value, expected_x, expected_obj in cases:
            with self.subTest(observed_value=observed_value.tolist()):
                model = self._new_model()
                x = admm.Var("x", 2)

                model.setObjective(
                    0.5 * admm.sum(admm.square(x - observed_value))
                    + udf.UnitSphereIndicator(x)
                )

                model.optimize()

                x_value = np.asarray(x.X)
                self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
                self.assertTrue(np.allclose(x_value, expected_x, atol=1e-2))
                self.assertLess(abs(np.linalg.norm(x_value) - 1.0), 1e-6)
                self.assertLess(abs(model.ObjVal - expected_obj), 1e-2)

    # min   0.5 * ||X - Y||_F^2 + delta_{O(n)}(X)
    #
    # where  O(n) = {X in R^{n x n} : X^T X = I}
    def test_user_defined_orthogonal_matrix_indicator(self):
        observed_value = np.array([[2.0, 0.0], [0.0, 0.5]])
        expected_x = np.eye(2)
        expected_obj = 0.625

        model = self._new_model()
        x = admm.Var("x", 2, 2)

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value))
            + udf.OrthogonalMatrixIndicator(x)
        )

        model.optimize()

        x_value = np.asarray(x.X)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.allclose(x_value, expected_x, atol=1e-2))
        self.assertTrue(np.allclose(x_value.T @ x_value, np.eye(2), atol=1e-6))
        self.assertLess(abs(model.ObjVal - expected_obj), 1e-2)

    # min   0.5 * ||x - y||_2^2 + delta_{Delta_r}(x)
    #
    # where  Delta_r = {x in R^n : x_i >= 0, sum_i x_i = r}
    def test_user_defined_simplex_indicator(self):
        observed_value = np.array([0.2, -0.1, 0.7])

        model = self._new_model()
        x = admm.Var("x", len(observed_value))

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value))
            + udf.SimplexIndicator(x, radius=1.0)
        )

        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   0.5 * ||x - y||_2^2 + delta_{B0_k}(x)
    #
    # where  B0_k = {x in R^n : ||x||_0 <= k}
    def test_user_defined_l0_ball_indicator(self):
        observed_value = np.array([0.2, -1.5, 0.7, 3.0])
        expected_x = np.array([0.0, -1.5, 0.0, 3.0])
        expected_obj = 0.265

        model = self._new_model()
        x = admm.Var("x", len(observed_value))

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value))
            + udf.L0BallIndicator(x, k=2)
        )

        model.optimize()

        x_value = np.asarray(x.X)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.allclose(x_value, expected_x, atol=1e-2))
        self.assertLessEqual(np.count_nonzero(np.abs(x_value) > 1e-8), 2)
        self.assertLess(abs(model.ObjVal - expected_obj), 1e-2)

    # min   0.5 * ||x - y||_2^2 + lam * ||x||_{1/2}
    #
    # where  ||x||_{1/2} = sum_i sqrt(|x_i|)
    def test_user_defined_lhalf_penalty(self):
        observed_value = np.array([0.2, 1.0, 2.0])
        penalty_weight = 0.5
        expected_x = np.array([0.0, 0.7015158583813426, 1.8144020185805392])
        expected_obj = 1.1740511186579512

        model = self._new_model()
        x = admm.Var("x", len(observed_value))

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value))
            + penalty_weight * udf.LHalfNorm(x)
        )

        model.optimize()

        x_value = np.asarray(x.X)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.allclose(x_value, expected_x, atol=1e-2))
        self.assertLess(abs(model.ObjVal - expected_obj), 1e-2)

    # min   0.5 * ||X - Y||_F^2 + delta_{rank(X) <= r}(X)
    def test_user_defined_rank_r_indicator(self):
        observed_value = np.array([[3.0, 0.0], [0.0, 1.0]])
        expected_x = np.array([[3.0, 0.0], [0.0, 0.0]])
        expected_obj = 0.5

        model = self._new_model()
        x = admm.Var("x", 2, 2)

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value))
            + udf.RankRIndicator(x, rank_bound=1)
        )

        model.optimize()

        x_value = np.asarray(x.X)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.allclose(x_value, expected_x, atol=1e-2))
        self.assertLessEqual(np.linalg.matrix_rank(x_value, tol=1e-8), 1)
        self.assertLess(abs(model.ObjVal - expected_obj), 1e-2)

    # min   0.5 * ||x - y||_2^2 + delta_{{0,1}^n}(x)
    def test_user_defined_binary_indicator(self):
        observed_value = np.array([0.2, 0.8, 1.4, -0.3])
        expected_x = np.array([0.0, 1.0, 1.0, 0.0])
        expected_obj = 0.165

        model = self._new_model()
        x = admm.Var("x", len(observed_value))

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value)) + udf.BinaryIndicator(x)
        )

        model.optimize()

        x_value = np.asarray(x.X)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.allclose(x_value, expected_x, atol=1e-2))
        self.assertTrue(
            np.all(np.logical_or(np.isclose(x_value, 0.0), np.isclose(x_value, 1.0)))
        )
        self.assertLess(abs(model.ObjVal - expected_obj), 1e-2)

    # min   0.5 * ||X - Y||_F^2 + ||X||_{2,1}
    #
    # where  ||X||_{2,1} = sum_j ||X[:,j]||_2
    def test_user_defined_l21_norm(self):
        observed_value = np.array([[3.0, 0.3], [4.0, 0.4], [0.0, 0.0]])

        model = self._new_model()
        x = admm.Var("x", 3, 2)

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value)) + udf.L21Norm(x)
        )

        model.optimize()

        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")

    # min   0.5 * ||X - Y||_F^2 + delta_{St(m,n)}(X)
    #
    # where  St(m,n) = {X in R^{m x n} : X^T X = I_n},  m >= n
    def test_user_defined_stiefel_indicator(self):
        observed_value = np.array([[2.0, 0.0], [0.0, 0.5], [0.0, 0.0]])
        expected_x = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        expected_obj = 0.625

        model = self._new_model()
        x = admm.Var("x", 3, 2)

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value)) + udf.StiefelIndicator(x)
        )

        model.optimize()

        x_value = np.asarray(x.X)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.allclose(x_value, expected_x, atol=1e-2))
        self.assertTrue(np.allclose(x_value.T @ x_value, np.eye(2), atol=1e-6))
        self.assertLess(abs(model.ObjVal - expected_obj), 1e-2)

    # min   0.5 * ||x - y||_2^2 + sum_i p_SCAD(x_i; alpha, a)
    #
    # where  p_SCAD(t) = alpha * |t|                                    if |t| <= alpha
    #                    (-t^2 + 2*a*alpha*|t| - alpha^2) / (2*(a-1))  if alpha < |t| <= a*alpha
    #                    (a+1) * alpha^2 / 2                            if |t| > a*alpha
    def test_user_defined_scad_penalty(self):
        observed_value = np.array([0.3, 1.0, 1.5])
        expected_x = np.array([0.0, 0.7176470588235291, 1.5])
        expected_obj = 0.7292352941176471

        model = self._new_model()
        x = admm.Var("x", len(observed_value))

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value))
            + udf.SCADPenalty(x, alpha=0.4, a=3.7)
        )

        model.optimize()

        x_value = np.asarray(x.X)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.allclose(x_value, expected_x, atol=1e-2))
        self.assertLess(abs(model.ObjVal - expected_obj), 1e-2)

    # min   0.5 * ||x - y||_2^2 + sum_i m_MCP(x_i; lam, beta)
    #
    # where  m_MCP(t) = lam * |t| - t^2 / (2*beta)  if |t| <= beta * lam
    #                   beta * lam^2 / 2              if |t| > beta * lam
    def test_user_defined_mcp_penalty(self):
        observed_value = np.array([0.2, 0.5, 1.0])
        expected_x = np.array([0.0, 0.2, 1.0])
        expected_obj = 0.295

        model = self._new_model()
        x = admm.Var("x", len(observed_value))

        model.setObjective(
            0.5 * admm.sum(admm.square(x - observed_value))
            + udf.MCPPenalty(x, lam=0.4, beta=2.0)
        )

        model.optimize()

        x_value = np.asarray(x.X)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        self.assertTrue(np.allclose(x_value, expected_x, atol=1e-2))
        self.assertLess(abs(model.ObjVal - expected_obj), 1e-2)

    # min   0.5 * ||A x - b||_2^2 + lam * ||x||_0
    # s.t.  x >= 0
    #
    # where  ||x||_0 = #{i : x_i != 0}
    #
    # L0-regularized nonnegative regression: a sensing matrix A
    # maps x to observations b with additive noise; the L0 penalty
    # promotes sparse recovery while the nonnegativity constraint
    # restricts the feasible set.
    def test_user_defined_l0_regression(self):
        np.random.seed(42)
        n, m, k = 20, 30, 3
        x_true = np.zeros(n)
        support = np.random.choice(n, k, replace=False)
        x_true[support] = np.random.rand(k) * 2 + 0.5
        A = np.random.randn(m, n)
        b = A @ x_true + 0.01 * np.random.randn(m)
        lam = 0.5

        model = self._new_model()
        x = admm.Var("x", n)
        model.setObjective(
            0.5 * admm.sum(admm.square(A @ x - b)) + lam * udf.L0Norm(x)
        )
        model.addConstr(x >= 0)

        model.setOption(admm.Options.admm_max_iteration, 5000)

        model.optimize()

        x_value = np.asarray(x.X)
        self.assertEqual(model.StatusString, "SOLVE_OPT_SUCCESS")
        # The recovered support should match the true support
        recovered_support = set(np.where(np.abs(x_value) > 1e-6)[0])
        true_support = set(support)
        self.assertEqual(recovered_support, true_support,
                         f"Support mismatch: recovered {recovered_support} vs true {true_support}")
        self.assertTrue(np.allclose(x_value[support], x_true[support], atol=0.1))

if __name__ == "__main__":

    unittest.main()
