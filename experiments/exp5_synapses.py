"""Experiment 5 (Q1, cost): capacity per synapse.

Counting synapses
  classical : N(N-1)/2 symmetric connections between the N neurons
  dense / modern, written as a biologically plausible two-layer network
  (Krotov & Hopfield, 2021): one hidden "memory neuron" per stored pattern,
  each connected to all N feature neurons -> N*K synapses.

So a modern network that stores K memories always needs N*K synapses: its
memories-per-synapse is exactly 1/N, whatever K is.  The classical network
stores ~0.14 N memories in ~N^2/2 synapses, i.e. ~0.28/N per synapse.

Matched-budget test: give every model the synapse budget of a classical
network of the same size, S = N(N-1)/2.  A two-layer model can then afford
K_h = S / N = (N-1)/2 hidden neurons, i.e. store (N-1)/2 memories.  We store
exactly that many random patterns and test recall from 10%-noisy cues.

Run:  python experiments/exp5_synapses.py   (needs results/exp1_capacity.json)
"""
import numpy as np

from common import save_json, load_json
from hopfield import build_model, spec_label, success_rate
from hopfield.data import random_patterns, flip_noise

N_LIST = [64, 128, 256, 512]
MODELS = [{"kind": "classical"}, {"kind": "dense", "n": 3},
          {"kind": "modern", "beta": 32}]


def matched_budget(seeds=20, cues=10):
    rows = []
    for N in N_LIST:
        S = N * (N - 1) // 2
        K = S // N  # memories affordable by a two-layer net with S synapses
        for spec in MODELS:
            r = []
            for s in range(seeds):
                rng = np.random.default_rng(s)
                X = random_patterns(K, N, rng)
                tgt = rng.choice(K, size=cues, replace=False)
                cue = flip_noise(X[tgt], 0.10, rng)
                r.append(success_rate(build_model(spec, X).recall(cue, rng=rng), X, tgt))
            rows.append({"N": N, "synapses": S, "K": K, "model": spec_label(spec),
                         "success": float(np.mean(r))})
            print(rows[-1], flush=True)
    return rows


def per_synapse():
    """Capacity per synapse from the Experiment 1 capacities."""
    d = load_json("exp1_capacity.json")
    rows = []
    for c in d["curves"]:
        N, K = c["N"], c["capacity"]
        if K <= 0:
            continue
        if c["spec"]["kind"] == "classical":
            syn = N * (N - 1) // 2
        else:
            syn = N * K  # one hidden neuron (N synapses) per stored memory
        rows.append({"model": c["model"], "N": N, "capacity": K,
                     "lower_bound": c["reached_end"], "synapses": syn,
                     "memories_per_synapse": K / syn,
                     "bits_per_synapse": K * N / syn})
    return rows


def main():
    save_json({"matched_budget": matched_budget(), "per_synapse": per_synapse()},
              "exp5_synapses.json")


if __name__ == "__main__":
    main()
