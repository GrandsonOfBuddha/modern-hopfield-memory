"""Live demo for the presentation.

1) digits : store K MNIST digits, damage one, and show what every model recalls
       python demo.py digits --K 500 --cue noise
       python demo.py digits --K 100 --cue occlude --seed 3
2) drm    : study a DRM word list and see whether each model "remembers" the lure
       python demo.py drm --list sleep
3) words  : recall a word pair (hetero-association) from a noisy cue
       python demo.py words --K 1000

Add --save out.png to write the digits figure instead of opening a window.
"""
import argparse

import numpy as np

from hopfield import build_model, spec_label, cosine
from hopfield.data import load_mnist, sample_mnist, flip_noise, occlude_bottom, WordCodes
from hopfield.drm_lists import DRM_LISTS

DIGIT_MODELS = [{"kind": "classical"}, {"kind": "dense", "n": 8}, {"kind": "dense", "n": 16},
                {"kind": "modern", "beta": 30}, {"kind": "modern", "beta": 300}]


def digits(args):
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)
    X, y = load_mnist()
    P, lab = sample_mnist(X, y, args.K, rng)
    t = int(rng.integers(args.K))
    cue = flip_noise(P[t:t + 1], 0.2, rng) if args.cue == "noise" else occlude_bottom(P[t:t + 1])
    panels = [("stored digit", P[t]), ("cue", cue[0])]
    print(f"Stored {args.K} digits; target is a '{lab[t]}'.")
    for spec in DIGIT_MODELS:
        out = build_model(spec, P).recall(cue, rng=0)[0]
        sims = cosine(out, P)[0]
        j = int(np.argmax(sims))
        verdict = ("correct" if j == t and sims[j] >= 0.95 else
                   f"recalled another '{lab[j]}'" if sims[j] >= 0.95 else "blend / spurious")
        print(f"  {spec_label(spec):14s} -> {verdict}  (similarity to target {sims[t]:.2f})")
        panels.append((f"{spec_label(spec)}\n{verdict}", out))
    fig, axs = plt.subplots(1, len(panels), figsize=(2 * len(panels), 2.4))
    for ax, (title, im) in zip(axs, panels):
        im = im / max(np.abs(im).max(), 1e-12)
        ax.imshow(im.reshape(28, 28), cmap="gray_r", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"K = {args.K} stored digits, cue = {args.cue}")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print("saved", args.save)
    else:
        plt.show()


def drm(args):
    words = DRM_LISTS[args.list]
    rng = np.random.default_rng(args.seed)
    print(f"Studied list: {', '.join(words)}")
    print(f"Critical lure (never studied): {args.list.upper()}\n")
    wc = WordCodes(N=512, seed=0)
    V = wc.vectors / np.linalg.norm(wc.vectors, axis=1, keepdims=True)
    X = V[[wc.index[w] for w in words]]
    for beta in (1, 3, 5, 10, 50):
        net = build_model({"kind": "modern", "beta": beta}, X)
        out = net.recall(rng.standard_normal((20, X.shape[1])))
        rec = [wc.words[j] for j in np.argmax(out @ V.T, axis=1)]
        vals, cnt = np.unique(rec, return_counts=True)
        top = ", ".join(f"{v}{' (LURE!)' if v == args.list else ''} x{c}"
                        for v, c in sorted(zip(vals, cnt), key=lambda t: -t[1])[:4])
        fam = net.familiarity(V[[wc.index[args.list]]])[0]
        fam_st = net.familiarity(X).mean()
        print(f"Modern beta={beta:<3g} recalls: {top:55s} "
              f"familiarity lure {fam:.2f} vs studied {fam_st:.2f}")


def words(args):
    rng = np.random.default_rng(args.seed)
    wc = WordCodes(N=512, seed=0)
    idx = rng.choice(np.arange(100, 10100), size=2 * args.K, replace=False)
    A, B = wc.codes[idx[:args.K]], wc.codes[idx[args.K:]]
    t = int(rng.integers(args.K))
    cue = flip_noise(A[t:t + 1], 0.2, rng)
    print(f"Stored {args.K} random word pairs. Cue: '{wc.words[idx[t]]}' (20% of bits flipped),"
          f" correct partner: '{wc.words[idx[args.K + t]]}'")
    from hopfield.models import LinearAssociator
    nets = [("Classical", LinearAssociator(A, B))] + \
           [(spec_label(s), build_model(s, A)) for s in
            ({"kind": "dense", "n": 3}, {"kind": "modern", "beta": 30})]
    for name, net in nets:
        out = net.hetero_one_step(cue, B)
        j = int(np.argmax(cosine(out, B)[0]))
        print(f"  {name:14s} -> '{wc.words[idx[args.K + j]]}'"
              f"{'  correct' if j == t else '  WRONG'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("digits")
    d.add_argument("--K", type=int, default=500)
    d.add_argument("--cue", choices=["noise", "occlude"], default="noise")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--save", default=None)
    r = sub.add_parser("drm")
    r.add_argument("--list", default="sleep", choices=sorted(DRM_LISTS))
    r.add_argument("--seed", type=int, default=0)
    w = sub.add_parser("words")
    w.add_argument("--K", type=int, default=1000)
    w.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"digits": digits, "drm": drm, "words": words}[args.cmd](args)


if __name__ == "__main__":
    main()
