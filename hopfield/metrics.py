"""Scoring recall: success, mix-ups and blends."""
from __future__ import annotations

import numpy as np

SUCCESS_SIM = 0.95  # cosine similarity needed to count as correct recall


def cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of A and every row of B."""
    A = np.atleast_2d(A).astype(np.float64)
    B = np.atleast_2d(B).astype(np.float64)
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return An @ Bn.T


def classify_outcomes(final: np.ndarray, memories: np.ndarray, targets: np.ndarray,
                      thr: float = SUCCESS_SIM):
    """Label every trial as 'correct', 'mixup' (another memory) or 'blend'.

    correct : nearest memory is the target and similarity >= thr
    mixup   : nearest memory is a different stored memory with similarity >= thr
    blend   : the state is not close to any single stored memory (a spurious /
              mixture state -- the model's version of a false memory)
    """
    sims = cosine(final, memories)
    best = np.argmax(sims, axis=1)
    best_sim = sims[np.arange(len(best)), best]
    target_sim = sims[np.arange(len(best)), targets]
    out = np.full(len(best), "blend", dtype=object)
    close = best_sim >= thr
    out[close & (best == targets)] = "correct"
    out[close & (best != targets)] = "mixup"
    return out, best, target_sim


def success_rate(final, memories, targets, thr: float = SUCCESS_SIM) -> float:
    labels, _, _ = classify_outcomes(final, memories, targets, thr)
    return float(np.mean(labels == "correct"))


def capacity_from_curve(Ks, rates, level: float = 0.9) -> float:
    """Largest K on the grid whose success rate is still >= level.

    Uses log-linear interpolation between the last passing and first failing K.
    Returns the largest tested K if the curve never drops below the level
    (i.e. a lower bound), and 0 if even the smallest K fails.
    """
    Ks = np.asarray(Ks, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if rates[0] < level:
        return 0.0
    for j in range(1, len(Ks)):
        if rates[j] < level:
            k0, k1, r0, r1 = Ks[j - 1], Ks[j], rates[j - 1], rates[j]
            t = (r0 - level) / max(r0 - r1, 1e-12)
            return float(np.exp(np.log(k0) + t * (np.log(k1) - np.log(k0))))
    return float(Ks[-1])
