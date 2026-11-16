"""Experiment 4 (Q2): DRM false memories.

In the DRM task people study words such as bed, rest, tired, dream ... and
then "remember" the never-presented critical lure SLEEP (Roediger &
McDermott, 1995).  False recall grows with the number of studied associates
(Robinson & Roediger, 1997).

Model version:
  * study  : store the first k associates of a DRM list (k = 3, 6, 9, 12, 15)
             as +-1 word codes (N = 512).  The lure is never stored.
  * recall : start the network from a random state (unconstrained free
             recall), let it settle, and read out the nearest word in a
             20 000-word lexicon.  The output is a studied word (veridical),
             the critical lure (false recall) or another word (intrusion).
  * recognition (k = 15): compare the familiarity (-energy) of studied
             words, the critical lure and an unrelated lure (another list's lure).

31 DRM lists x 3 random word codings x 40 random starts per condition.

Run:  python experiments/exp4_drm.py [--quick]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import save_json
from hopfield import build_model, spec_label, unit_rows
from hopfield.models import logsumexp
from hopfield.data import WordCodes
from hopfield.drm_lists import DRM_LISTS

N = 512
LIST_LENGTHS = [3, 6, 9, 12, 15]
BETAS = [1, 2, 3, 5, 7, 10, 20, 50, 100]
# Binary models use +-1 word codes ("codes").  The modern network is defined
# for continuous patterns, so it is run on the unit-length GloVe vectors
# themselves ("glove") and, as a control, on the same +-1 codes.
MODELS = ([{"kind": "classical", "space": "codes"}]
          + [{"kind": "dense", "n": n, "space": "codes"} for n in (3, 4, 8)]
          + [{"kind": "modern", "beta": b, "space": "glove"} for b in BETAS]
          + [{"kind": "modern", "beta": b, "space": "codes"} for b in BETAS])


class Space:
    """Word patterns and lexicon decoding in one representation."""

    def __init__(self, space, code_seed):
        self.wc = WordCodes(N=N, seed=code_seed)
        self.glove = space == "glove"
        if self.glove:
            V = self.wc.vectors
            self.lex = V / np.linalg.norm(V, axis=1, keepdims=True)
        else:
            self.lex = self.wc.codes / np.sqrt(N)
        self.dim = self.lex.shape[1]

    def __call__(self, words):
        return self.lex[[self.wc.index[w] for w in words]] * (1 if self.glove else np.sqrt(N))

    def random_starts(self, n, rng):
        if self.glove:
            return unit_rows(rng.standard_normal((n, self.dim)))
        return rng.choice([-1.0, 1.0], size=(n, self.dim))

    def decode(self, states):
        sims = unit_rows(states) @ self.lex.T
        return [self.wc.words[j] for j in np.argmax(sims, axis=1)]


def familiarity(spec, X, probes):
    """-Energy of a probe state (higher = feels more familiar)."""
    m = unit_rows(probes) @ unit_rows(X).T  # cosine overlaps
    if spec["kind"] == "classical":
        return np.sum(m ** 2, axis=1)
    if spec["kind"] == "dense":
        return np.sum(m ** spec["n"], axis=1)
    b = spec["beta"]
    return logsumexp(b * m, axis=1) / b


def run(spec, code_seeds, n_starts):
    lures = list(DRM_LISTS)
    recall = {k: {"veridical": 0, "lure": 0, "other": 0} for k in LIST_LENGTHS}
    per_list = {l: {"lure": 0, "n": 0} for l in lures}  # at k = 15
    recog = {"pct_studied_below_lure": [], "pct_studied_below_unrelated": [],
             "lure_minus_unrelated": [], "studied_minus_unrelated": []}
    examples = {}
    for cs in code_seeds:
        wc = Space(spec.get("space", "codes"), cs)
        for li, lure in enumerate(lures):
            words = DRM_LISTS[lure]
            for k in LIST_LENGTHS:
                X = wc(words[:k])
                rng = np.random.default_rng(10_000 * cs + 100 * li + k)
                starts = wc.random_starts(n_starts, rng)
                out = build_model(spec, X).recall(starts, rng=rng)
                decoded = wc.decode(out)
                for w in decoded:
                    key = "lure" if w == lure else "veridical" if w in words[:k] else "other"
                    recall[k][key] += 1
                    if k == 15:
                        per_list[lure]["lure"] += key == "lure"
                        per_list[lure]["n"] += 1
                if k == 15 and cs == code_seeds[0] and lure in ("sleep", "window", "king", "cup"):
                    vals, cnt = np.unique(decoded, return_counts=True)
                    examples[lure] = dict(sorted(zip(vals.tolist(), cnt.tolist()),
                                                 key=lambda t: -t[1])[:5])
            # recognition after studying the full list
            X = wc(words)
            unrelated = lures[(li + 1) % len(lures)]
            f_st = familiarity(spec, X, X)
            f_lu = familiarity(spec, X, wc([lure]))[0]
            f_un = familiarity(spec, X, wc([unrelated]))[0]
            recog["pct_studied_below_lure"].append(float(np.mean(f_st < f_lu)))
            recog["pct_studied_below_unrelated"].append(float(np.mean(f_st < f_un)))
            recog["lure_minus_unrelated"].append(float(f_lu - f_un))
            recog["studied_minus_unrelated"].append(float(np.mean(f_st) - f_un))
    rec = {}
    for k, c in recall.items():
        n = sum(c.values())
        rec[k] = {key: v / n for key, v in c.items()}
    lists = {l: v["lure"] / v["n"] for l, v in per_list.items()}
    summary = {key: float(np.mean(v)) for key, v in recog.items()}
    label = spec_label(spec) + (" (GloVe)" if spec.get("space") == "glove" else "")
    print(label, {k: round(v["lure"], 2) for k, v in rec.items()},
          "recog:", {k: round(v, 2) for k, v in summary.items()}, flush=True)
    return {"model": label, "spec": spec, "recall_by_length": rec,
            "false_recall_by_list_k15": lists, "recognition": summary,
            "examples_k15": examples}


def lure_centroid_similarity():
    """How close each lure is to the average GloVe vector of its list.

    rank_overall: rank of the lure among *all* lexicon words (studied ones
    included) -- rank 1 means a perfect blend of the list reads out as the lure.
    """
    wc = WordCodes(N=N, seed=0)
    V = wc.vectors / np.linalg.norm(wc.vectors, axis=1, keepdims=True)
    out = {}
    for lure, words in DRM_LISTS.items():
        c = V[[wc.index[w] for w in words]].mean(0)
        c /= np.linalg.norm(c)
        sims = V @ c
        rank = int(np.sum(sims > sims[wc.index[lure]])) + 1
        out[lure] = {"cosine": float(sims[wc.index[lure]]), "rank_overall": rank}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    seeds = [0] if args.quick else [0, 1, 2]
    starts = 10 if args.quick else 40
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(run, MODELS, [seeds] * len(MODELS), [starts] * len(MODELS)))
    save_json({"N": N, "list_lengths": LIST_LENGTHS, "code_seeds": seeds,
               "starts": starts, "lure_centroid": lure_centroid_similarity(),
               "results": res}, "exp4_drm.json")


if __name__ == "__main__":
    main()
