"""Experiment 2 (Task 1, Q1 + Q2): recalling handwritten digits.

Binarised MNIST digits (N = 784 pixels, +1 ink / -1 background) are stored
in each model.  A stored digit is damaged in one of two ways
    noise    : 20% of the pixels flipped
    occlude  : bottom half of the image hidden (set to background)
and the model has to restore it.  K (number of stored digits) goes from 10 to
20 000.  Every failure is classified as
    mix-up (same class)  : the network recalled another stored digit of the same class
    mix-up (other class) : it recalled a stored digit of a different class (e.g. 4 -> 9)
    blend                : it settled in a state that is not any stored digit
20 pattern sets x 5 cues = 100 trials per point; identical memories and cues
for all models.

Run:  python experiments/exp2_mnist.py [--quick]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import save_json, RESULTS
from hopfield import build_model, spec_label, classify_outcomes
from hopfield.data import load_mnist, sample_mnist, flip_noise, occlude_bottom

MODELS = [
    {"kind": "classical"},
    {"kind": "dense", "n": 4},
    {"kind": "dense", "n": 8},
    {"kind": "dense", "n": 16},
    {"kind": "modern", "beta": 30},
    {"kind": "modern", "beta": 100},
    {"kind": "modern", "beta": 300},
    {"kind": "modern", "beta": 1000},
    {"kind": "nn"},  # reference: beta -> infinity
]
K_GRID = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
CUES = {"noise": 0.20, "occlude": 0.5}


def make_trial(X, y, K, seed, n_cues, cue_type):
    rng = np.random.default_rng(10_000 * seed + K)
    P, lab = sample_mnist(X, y, K, rng)
    tgt = rng.choice(K, size=min(n_cues, K), replace=False)
    if cue_type == "noise":
        cue = flip_noise(P[tgt], CUES["noise"], rng)
    else:
        cue = occlude_bottom(P[tgt], frac=CUES["occlude"])
    return P, lab, tgt, cue, rng


def run(spec, cue_type, seeds, n_cues, k_grid):
    X, y = load_mnist()
    rows = []
    confusion = np.zeros((10, 10), dtype=int)  # cue class -> recalled class (other-class mix-ups)
    low = 0
    for K in k_grid:
        counts = {"correct": 0, "mixup_same": 0, "mixup_other": 0, "blend": 0}
        for s in range(seeds):
            P, lab, tgt, cue, rng = make_trial(X, y, K, s, n_cues, cue_type)
            out = build_model(spec, P).recall(cue, rng=rng)
            labels, best, _ = classify_outcomes(out, P, tgt)
            for l, b, t in zip(labels, best, tgt):
                if l == "mixup":
                    if lab[b] == lab[t]:
                        counts["mixup_same"] += 1
                    else:
                        counts["mixup_other"] += 1
                        confusion[lab[t], lab[b]] += 1
                else:
                    counts[l] += 1
        n = sum(counts.values())
        frac = {k: v / n for k, v in counts.items()}
        rows.append({"K": K, **frac})
        low = low + 1 if frac["correct"] < 0.05 else 0
        print(f"{spec_label(spec):14s} {cue_type:8s} K={K:6d} correct={frac['correct']:.2f} "
              f"same={frac['mixup_same']:.2f} other={frac['mixup_other']:.2f} "
              f"blend={frac['blend']:.2f}", flush=True)
        if low >= 2:  # stop once recall has collapsed
            break
    return {"model": spec_label(spec), "spec": spec, "cue": cue_type, "rows": rows,
            "confusion_other_class": confusion.tolist()}


def examples():
    """A few cue -> recall examples for the figure and the demo."""
    X, y = load_mnist()
    ex = {}
    for K in (100, 1000):
        for cue_type in CUES:
            P, lab, tgt, cue, rng = make_trial(X, y, K, 0, 4, cue_type)
            outs = [build_model(sp, P).recall(cue, rng=0) for sp in MODELS]
            ex[f"K{K}_{cue_type}"] = {"target": P[tgt], "cue": cue,
                                       "recalled": np.stack(outs)}
    np.savez_compressed(RESULTS / "exp2_examples.npz",
                        labels=np.array([spec_label(s) for s in MODELS]),
                        **{f"{k}_{f}": v[f] for k, v in ex.items() for f in v})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    seeds = 3 if args.quick else 20
    grid = K_GRID[:6] if args.quick else K_GRID
    jobs = [(sp, c, seeds, 5, grid) for sp in MODELS for c in CUES]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(run, *zip(*jobs)))
    X, y = load_mnist()
    rng = np.random.default_rng(0)
    P, _ = sample_mnist(X, y, 1000, rng)
    C = P @ P.T / P.shape[1]
    mean_overlap = float(C[np.triu_indices(len(P), 1)].mean())
    save_json({"cues": CUES, "seeds": seeds, "cues_per_seed": 5,
               "mean_pairwise_cosine": mean_overlap, "results": res}, "exp2_mnist.json")
    examples()


if __name__ == "__main__":
    main()
