"""Associative-memory models used in the project.

All models store K patterns (rows of a K x N matrix ``X``) and recall a stored
pattern from a damaged cue.  They differ only in how sharply they focus on the
best-matching memory (the "separation function" of Millidge et al., 2022):

    model                       separation of overlap m      sharpness dial
    ------------------------    -------------------------    --------------
    ClassicalHopfield           linear  (F(m) = m^2)         none
    DenseAM(n)                  polynomial F(m) = m^n        power n
    ModernHopfield(beta)        softmax(beta * m)            inverse temp. beta

Binary models (ClassicalHopfield, DenseAM) work on bipolar states (+1/-1) and
use asynchronous single-neuron updates, as in Hopfield (1982) and
Krotov & Hopfield (2016).  ModernHopfield works on continuous unit-norm states
and uses the one-step update of Ramsauer et al. (2021).

Every recall method works on a *batch* of cues (B x N) at once so that many
trials can be simulated cheaply.
"""
from __future__ import annotations

import numpy as np


def _sign(x: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Sign function that keeps the previous state when the input is exactly 0."""
    out = np.sign(x)
    zero = out == 0
    if np.any(zero):
        out[zero] = previous[zero]
    return out


def unit_rows(X: np.ndarray) -> np.ndarray:
    """Scale every row of X to unit Euclidean norm."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# --------------------------------------------------------------------------
# Classical Hopfield network (baseline)
# --------------------------------------------------------------------------
class ClassicalHopfield:
    """Hopfield (1982) network with the Hebbian rule W = X^T X / N, w_ii = 0."""

    name = "Classical"

    def __init__(self, patterns: np.ndarray):
        X = np.asarray(patterns, dtype=np.float64)
        self.X = X
        self.K, self.N = X.shape
        self.W = X.T @ X / self.N
        np.fill_diagonal(self.W, 0.0)

    @property
    def n_synapses(self) -> int:
        """Number of distinct (symmetric) connections between N neurons."""
        return self.N * (self.N - 1) // 2

    def energy(self, S: np.ndarray) -> np.ndarray:
        S = np.atleast_2d(S)
        return -0.5 * np.einsum("bi,ij,bj->b", S, self.W, S)

    def recall(self, cues, max_sweeps: int = 30, rng=None):
        """Asynchronous updates in random order until no neuron changes."""
        rng = np.random.default_rng(rng)
        S = np.array(np.atleast_2d(cues), dtype=np.float64)
        for _ in range(max_sweeps):
            changed = False
            for i in rng.permutation(self.N):
                h = S @ self.W[i]
                new = _sign(h, S[:, i])
                if np.any(new != S[:, i]):
                    changed = True
                    S[:, i] = new
            if not changed:
                break
        return S

    def recall_one_step(self, cues):
        """Single synchronous update (used for auto vs hetero comparisons)."""
        S = np.atleast_2d(cues).astype(np.float64)
        return _sign(S @ self.W, S)


# --------------------------------------------------------------------------
# Dense associative memory (Krotov & Hopfield, 2016)
# --------------------------------------------------------------------------
class DenseAM:
    """Dense associative memory with energy E = -sum_mu F(xi_mu . sigma).

    F(x) = x^n (polynomial).  n = 2 is exactly the classical Hopfield network.
    The update for neuron i compares the energy with sigma_i = +1 and -1:

        sigma_i <- sign( sum_mu F(+xi_mu,i + m'_mu) - F(-xi_mu,i + m'_mu) )

    where m'_mu = sum_{j != i} xi_mu,j sigma_j.  Overlaps are kept up to date
    incrementally so that one sweep costs O(N K B).
    """

    def __init__(self, patterns: np.ndarray, n: int = 3):
        self.X = np.asarray(patterns, dtype=np.float64)
        self.K, self.N = self.X.shape
        self.n = int(n)
        self.name = f"Dense n={self.n}"

    def F(self, x):
        return x ** self.n

    def dF(self, x):
        """Derivative of F, used as the separation function in one-step reads."""
        return self.n * x ** (self.n - 1)

    @property
    def n_synapses(self) -> int:
        """Synapses in the two-layer form of Krotov & Hopfield (2021):
        one hidden 'memory neuron' per pattern, connected to all N inputs."""
        return self.N * self.K

    def energy(self, S):
        S = np.atleast_2d(S)
        return -np.sum(self.F(S @ self.X.T), axis=1)

    def recall(self, cues, max_sweeps: int = 30, rng=None):
        rng = np.random.default_rng(rng)
        S = np.array(np.atleast_2d(cues), dtype=np.float64)
        X = self.X
        # Normalise overlaps by N to keep numbers small (sign is unaffected).
        scale = 1.0 / self.N
        M = (S @ X.T) * scale  # B x K overlaps
        for _ in range(max_sweeps):
            changed = False
            for i in rng.permutation(self.N):
                xi = X[:, i] * scale  # K
                m_rest = M - S[:, i:i + 1] * xi  # remove neuron i
                drive = np.sum(self.F(m_rest + xi) - self.F(m_rest - xi), axis=1)
                new = _sign(drive, S[:, i])
                flip = new != S[:, i]
                if np.any(flip):
                    changed = True
                    M[flip] += (new[flip] - S[flip, i])[:, None] * xi
                    S[flip, i] = new[flip]
            if not changed:
                break
        return S

    def recall_one_step(self, cues):
        S = np.atleast_2d(cues).astype(np.float64)
        m = S @ self.X.T / self.N
        return _sign(self.dF(m) @ self.X, S)

    def hetero_one_step(self, cues, targets):
        """Read out the partner patterns Z given cues matched against X."""
        S = np.atleast_2d(cues).astype(np.float64)
        m = S @ self.X.T / self.N
        out = self.dF(m) @ np.asarray(targets, dtype=np.float64)
        return _sign(out, np.ones_like(out))


class ClassicalAsDense(DenseAM):
    """The classical network written as a DenseAM with n = 2 (for checks)."""

    def __init__(self, patterns):
        super().__init__(patterns, n=2)
        self.name = "Classical (n=2)"


# --------------------------------------------------------------------------
# Modern Hopfield network (Ramsauer et al., 2021)
# --------------------------------------------------------------------------
def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z: np.ndarray, axis: int = -1) -> np.ndarray:
    zmax = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(zmax, axis) + np.log(np.sum(np.exp(z - zmax), axis=axis))


class ModernHopfield:
    """Continuous modern Hopfield network.

    Energy:  E(xi) = -lse(beta, X xi) + 0.5 xi.xi + const
    Update:  xi_new = X^T softmax(beta X xi)

    Patterns and states are scaled to unit length so that X xi is a cosine
    similarity in [-1, 1] and beta has the same meaning for every data set.
    """

    def __init__(self, patterns: np.ndarray, beta: float = 32.0):
        self.X = unit_rows(patterns)
        self.K, self.N = self.X.shape
        self.beta = float(beta)
        self.name = f"Modern b={self.beta:g}"

    @property
    def n_synapses(self) -> int:
        return self.N * self.K

    def energy(self, S):
        S = unit_rows(np.atleast_2d(S))
        lse = logsumexp(self.beta * S @ self.X.T, axis=1) / self.beta
        return -lse + 0.5 * np.sum(S * S, axis=1)

    def attention(self, S):
        """Weights each stored pattern receives for cue(s) S (B x K)."""
        return softmax(self.beta * unit_rows(np.atleast_2d(S)) @ self.X.T, axis=1)

    def recall(self, cues, max_iter: int = 10, tol: float = 1e-6, rng=None):
        S = unit_rows(np.atleast_2d(cues))
        for _ in range(max_iter):
            new = softmax(self.beta * S @ self.X.T, axis=1) @ self.X
            if np.max(np.abs(new - S)) < tol:
                S = new
                break
            S = new
        return S

    def recall_one_step(self, cues):
        S = unit_rows(np.atleast_2d(cues))
        return softmax(self.beta * S @ self.X.T, axis=1) @ self.X

    def hetero_one_step(self, cues, targets):
        S = unit_rows(np.atleast_2d(cues))
        return softmax(self.beta * S @ self.X.T, axis=1) @ unit_rows(targets)

    def familiarity(self, probes):
        """Log-sum-exp match of a probe to memory (= minus the energy term)."""
        S = unit_rows(np.atleast_2d(probes))
        return logsumexp(self.beta * S @ self.X.T, axis=1) / self.beta


class LinearAssociator:
    """Classical hetero-associative memory W = Z^T X / N (linear separation).

    This is the hetero-associative counterpart of the classical Hopfield
    network (Kohonen 1972; Hopfield 1982 rule applied between two layers).
    """

    name = "Classical"

    def __init__(self, keys, targets):
        self.X = np.asarray(keys, dtype=np.float64)
        self.Z = np.asarray(targets, dtype=np.float64)
        self.K, self.N = self.X.shape
        self.W = self.Z.T @ self.X / self.N

    def hetero_one_step(self, cues, targets=None):
        S = np.atleast_2d(cues).astype(np.float64)
        out = S @ self.W.T
        return _sign(out, np.ones_like(out))


class NearestNeighbour:
    """Return the stored pattern most similar to the cue.

    This is the beta -> infinity limit of the modern network and the best that
    any similarity-based memory can do; used as a reference ceiling.
    """

    name = "Nearest neighbour"

    def __init__(self, patterns):
        self.X = np.asarray(patterns, dtype=np.float64)

    def recall(self, cues, rng=None):
        S = unit_rows(np.atleast_2d(cues))
        return self.X[np.argmax(S @ unit_rows(self.X).T, axis=1)]


def build_model(spec: dict, patterns: np.ndarray):
    """Create a model from a small spec dict, e.g. {"kind": "dense", "n": 3}."""
    kind = spec["kind"]
    if kind == "classical":
        return ClassicalHopfield(patterns)
    if kind == "dense":
        return DenseAM(patterns, n=spec["n"])
    if kind == "modern":
        return ModernHopfield(patterns, beta=spec["beta"])
    if kind == "nn":
        return NearestNeighbour(patterns)
    raise ValueError(f"unknown model kind {kind!r}")


def spec_label(spec: dict) -> str:
    kind = spec["kind"]
    if kind == "classical":
        return "Classical"
    if kind == "dense":
        return f"Dense n={spec['n']}"
    if kind == "nn":
        return "Nearest neighbour"
    return f"Modern β={spec['beta']:g}"
