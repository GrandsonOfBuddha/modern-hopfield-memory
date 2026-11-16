"""Experiment 3 (Task 2, Q3): auto- vs hetero-association with words.

Words are GloVe vectors turned into N = 512 bit +-1 codes (random-hyperplane
hashing keeps their semantic similarity).  Four memory tasks, same network
size, same cues damaged by flipping 20% of the bits:

  auto (recurrent) : store K words, recall a word from its damaged code by
                     running the full dynamics to a fixed point
  auto (one step)  : the same, but with a single update
  hetero related   : store K word pairs (a -> b) where b is a's closest
                     semantic neighbour (e.g. doctor -> physician); given a
                     damaged a, read out b in one step
  hetero unrelated : the same target words re-paired at random

In the shared framework of Millidge et al. (2022) all models compute
    output = P . sep(X . cue)
with P = X for auto- and P = Z (partners) for hetero-association, so the
comparison isolates the effect of the projection step.

Run:  python experiments/exp3_words.py [--quick]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import save_json, RESULTS
from hopfield import build_model, spec_label, success_rate, capacity_from_curve
from hopfield.models import LinearAssociator
from hopfield.data import WordCodes, flip_noise

N = 512
NOISE = 0.20
MODELS = [
    {"kind": "classical"},
    {"kind": "dense", "n": 3},
    {"kind": "dense", "n": 4},
    {"kind": "modern", "beta": 10},
    {"kind": "modern", "beta": 30},
    {"kind": "modern", "beta": 100},
]
K_GRID = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
TASKS = ["auto_recurrent", "auto_onestep", "hetero_related", "hetero_unrelated"]
POOL = (100, 10100)  # cue words: frequency ranks 100..10100


def neighbour_table(wc, k=10):
    """For every pool word, its k nearest GloVe neighbours (by vector cosine)."""
    path = RESULTS / "glove_neighbours.npy"
    if path.exists():
        return np.load(path)
    V = wc.vectors / np.linalg.norm(wc.vectors, axis=1, keepdims=True)
    lo, hi = POOL
    nb = np.zeros((hi - lo, k), dtype=int)
    for s in range(lo, hi, 1000):
        sims = V[s:s + 1000] @ V.T
        sims[np.arange(len(sims)), np.arange(s, s + len(sims))] = -np.inf
        nb[s - lo:s - lo + len(sims)] = np.argsort(-sims, axis=1)[:, :k]
    np.save(path, nb)
    return nb


def make_pairs(K, rng, nb):
    """K cue words, each with a distinct related partner (its nearest unused neighbour)."""
    lo, hi = POOL
    order = rng.permutation(hi - lo)
    used, cues, targets = set(), [], []
    for j in order:
        a = lo + j
        if a in used:
            continue
        for b in nb[j]:
            if b not in used and b != a:
                cues.append(a)
                targets.append(int(b))
                used.update((a, int(b)))
                break
        if len(cues) == K:
            break
    return np.array(cues), np.array(targets)


def run(spec, seeds, n_cues, grid):
    wc = WordCodes(N=N, seed=0)
    nb = neighbour_table(wc)
    curves = {t: [] for t in TASKS}
    done = {t: False for t in TASKS}
    for K in grid:
        hits = {t: 0.0 for t in TASKS}
        for s in range(seeds):
            rng = np.random.default_rng(777 * s + K)
            a_idx, b_idx = make_pairs(K, rng, nb)
            A, B = wc.codes[a_idx], wc.codes[b_idx]
            B_unrel = B[rng.permutation(K)]
            tgt = rng.choice(K, size=min(n_cues, K), replace=False)
            cue = flip_noise(A[tgt], NOISE, rng)
            auto = build_model(spec, A)
            if not done["auto_recurrent"]:
                hits["auto_recurrent"] += success_rate(auto.recall(cue, rng=rng), A, tgt)
            if not done["auto_onestep"]:
                hits["auto_onestep"] += success_rate(auto.recall_one_step(cue), A, tgt)
            for task, Z in (("hetero_related", B), ("hetero_unrelated", B_unrel)):
                if done[task]:
                    continue
                if spec["kind"] == "classical":
                    out = LinearAssociator(A, Z).hetero_one_step(cue)
                else:
                    out = auto.hetero_one_step(cue, Z)
                hits[task] += success_rate(out, Z, tgt)
        for t in TASKS:
            if not done[t]:
                r = hits[t] / seeds
                curves[t].append((K, r))
                done[t] = r < 0.05
        print(spec_label(spec), K, {t: round(curves[t][-1][1], 2) for t in TASKS}, flush=True)
        if all(done.values()):
            break
    out = {"model": spec_label(spec), "spec": spec, "tasks": {}}
    for t in TASKS:
        Ks, rs = zip(*curves[t])
        out["tasks"][t] = {"K": list(Ks), "success": list(rs),
                           "capacity": capacity_from_curve(Ks, rs)}
    return out


def code_stats():
    """Average similarity of codes for related vs random word pairs."""
    wc = WordCodes(N=N, seed=0)
    nb = neighbour_table(wc)
    rng = np.random.default_rng(0)
    a, b = make_pairs(1000, rng, nb)
    A, B = wc.codes[a], wc.codes[b]
    rel = float(np.mean(np.sum(A * B, 1) / N))
    perm = rng.permutation(1000)
    unrel = float(np.mean(np.sum(A * B[perm], 1) / N))
    examples = [(wc.words[i], wc.words[j]) for i, j in zip(a[:12], b[:12])]
    return {"related_pair_cosine": rel, "random_pair_cosine": unrel,
            "example_related_pairs": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    seeds = 4 if args.quick else 20
    grid = K_GRID[:7] if args.quick else K_GRID
    stats = code_stats()  # also builds the neighbour table once
    print(stats)
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(run, MODELS, [seeds] * len(MODELS), [10] * len(MODELS),
                          [grid] * len(MODELS)))
    save_json({"N": N, "noise": NOISE, "seeds": seeds, "cues_per_seed": 10,
               "code_stats": stats, "results": res}, "exp3_words.json")


if __name__ == "__main__":
    main()
