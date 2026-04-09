import admm
import numpy as np


class GraphLaplacianSmoothing(admm.UDFBase):
    r"""Graph Laplacian smoothing: penalizes variation over a graph.

    Mathematical definition:

        f(x) = Σ_{(i,j) ∈ E} (xᵢ − xⱼ)²
             = xᵀ L x

    where E is the edge set of an undirected graph and L is the
    graph Laplacian matrix.

    Behavior:
        x constant:  f(x) = 0            (minimum — no variation)
        x varies across edges:  f(x) > 0  (penalizes disagreement)

    Gradient:

        ∇f(x)ᵢ = 2 · Σ_{j: (i,j) ∈ E} (xᵢ − xⱼ)

    Equivalently, ∇f(x) = 2Lx where L is the Laplacian matrix.

    Properties:
    - Convex (quadratic form with positive semidefinite Laplacian)
    - Null space: constant vectors (f(c·1) = 0)
    - Generalizes total variation to arbitrary graph topologies
    - For a path graph (chain), reduces to Σ(xᵢ₊₁ − xᵢ)²

    Used in:
    - Graph signal processing (smooth graph signals)
    - Semi-supervised learning (label propagation)
    - Spatial statistics (Gaussian Markov random fields)
    - Image segmentation (pixel graphs)
    - Social network analysis (opinion dynamics)

    Parameters
    ----------
    arg : admm.Var or expression
        The graph signal x.
    edges : list of (int, int)
        Edge list of the undirected graph.
    """

    def __init__(self, arg, edges):
        self.arg = arg
        self.edges = edges

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float).ravel()
        s = 0.0
        for i, j in self.edges:
            s += (x[i] - x[j]) ** 2
        return float(s)

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float).ravel()
        g = np.zeros_like(x)
        for i, j in self.edges:
            g[i] += 2.0 * (x[i] - x[j])
            g[j] -= 2.0 * (x[i] - x[j])
        return [g.reshape(arglist[0].shape)]
