"""Experiment 1 (Q1): storage capacity on random patterns.

For every model and network size N we store K random +-1 patterns, damage a
stored pattern by flipping 10% of its bits and run the recall dynamics.
A trial succeeds when the final state is closest to the right memory with
cosine similarity >= 0.95.  Capacity = largest K recalled correctly in at
least 90% of trials (20 random pattern sets x 5 cues = 100 trials per K).

Reproduces: the classical limit K ~ 0.14 N (Amit et al., 1985), the
polynomial capacity K ~ N^(n-1) of dense memories (Krotov & Hopfield, 2016)
and the exponential capacity of the modern network (Ramsauer et al., 2021).

Run:  python experiments/exp1_capacity.py [--quick]
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import save_json
from hopfield import build_model, spec_label, capacity_from_curve, success_rate
from hopfield.data import random_patterns, flip_noise

NOISE = 0.10
MODELS = [
    {"kind": "classical"},
    {"kind": "dense", "n": 3},
    {"kind": "dense", "n": 4},
    {"kind": "modern", "beta": 8},
    {"kind": "modern", "beta": 32},
    {"kind": "modern", "beta": 128},
]
N_LIST = [16, 32, 64, 128, 256]
# extra small sizes for the modern network, whose capacity grows so fast that
# it only drops below the largest tested K (2^17) when N is small
N_MODERN_EXTRA = [8, 12, 20, 24, 40, 48]


def k_grid(kmax):
    ks = sorted({int(round(2 ** (e / 4))) for e in range(4, int(4 * np.log2(kmax)) + 1)})
    return [k for k in ks if k >= 2]


def run_curve(spec, N, seeds, cues, kmax):
    t0 = time.time()
    Ks, rates = [], []
    fails = 0
    for K in k_grid(kmax):
        ok = []
        for s in range(seeds):
            rng = np.random.default_rng(1000 * s + K)
            X = random_patterns(K, N, rng)
            tgt = rng.choice(K, size=min(cues, K), replace=False)
            cue = flip_noise(X[tgt], NOISE, rng)
            model = build_model(spec, X)
            out = model.recall(cue, rng=rng)
            ok.append(success_rate(out, X, tgt) * len(tgt))
        n_trials = seeds * min(cues, K)
        rate = float(np.sum(ok) / n_trials)
        Ks.append(K)
        rates.append(rate)
        fails = fails + 1 if rate < 0.5 else 0
        if fails >= 2:
            break
    cap = capacity_from_curve(Ks, rates)
    print(f"{spec_label(spec):14s} N={N:4d} capacity={cap:9.1f} "
          f"(max K tested {Ks[-1]}, {time.time() - t0:.0f}s)", flush=True)
    return {"model": spec_label(spec), "spec": spec, "N": N, "K": Ks,
            "success": rates, "capacity": cap, "reached_end": rates[-1] >= 0.9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer seeds, smaller grid")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    seeds = 4 if args.quick else 20
    cues = 5
    jobs = []
    for N in sorted(N_LIST + N_MODERN_EXTRA):
        for spec in MODELS:
            if N in N_MODERN_EXTRA and spec["kind"] != "modern":
                continue
            kmax = 2 ** 17 if spec["kind"] == "modern" else 2 ** 15
            if args.quick:
                kmax = min(kmax, 2 ** 12)
            jobs.append((spec, N, seeds, cues, kmax))
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(run_curve, *zip(*jobs)))
    save_json({"noise": NOISE, "seeds": seeds, "cues_per_seed": cues,
               "threshold_success": 0.9, "curves": results},
              "exp1_capacity.json")


if __name__ == "__main__":
    main()
