"""Larger, simpler versions of the report figures for the presentation.

The report figures are sized for an IEEE column and are unreadable on a projector,
so each slide gets its own figure with one message, big fonts and labels written
next to the lines instead of a legend.

Run:  python experiments/make_slide_figures.py      (writes figures/slides/*.png)
"""
import numpy as np

from common import plt, load_json, color_for, RESULTS, FIGURES
from hopfield.drm_lists import HUMAN_MEAN_FALSE_RECALL

OUT = FIGURES / "slides"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.linewidth": 1.0, "lines.linewidth": 2.4, "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Carlito", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "cm",
})

GREY = "#555555"
DENSE = "#3182bd"
MODERN = "#e6550d"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


def label_end(ax, x, y, text, color, dx=1.08, dy=1.0, **kw):
    ax.text(x * dx, y * dy, text, color=color, va="center", fontsize=14, **kw)


# --------------------------------------------------------------------------
def classical_collapse():
    """Motivation: classical recall collapses near K = 0.14N for every N."""
    curves = load_json("exp1_capacity.json")["curves"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Ns = [32, 64, 128, 256]
    shades = ["#bdbdbd", "#969696", "#636363", "#252525"]
    for N, col in zip(Ns, shades):
        c = next(c for c in curves if c["model"] == "Classical" and c["N"] == N)
        K = np.array(c["K"])
        keep = K / N <= 0.5
        ax.plot(K[keep] / N, np.array(c["success"])[keep], "-", color=col, label=f"N = {N}")
    ax.axvline(0.138, color=MODERN, ls="--", lw=1.6)
    ax.text(0.130, 0.12, "theory: 0.138 N", color=MODERN, fontsize=14, ha="right")
    ax.set_xlabel("memories per neuron  K / N")
    ax.set_ylabel("fraction recalled correctly")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "s_classical_collapse")


def sharpness():
    m = np.linspace(-1, 1, 401)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(m, m, color=GREY)
    ax.plot(m, 8 * m ** 7 / 8, color=DENSE)
    ax.plot(m, np.exp(50 * (m - 1)), color=MODERN)
    ax.axhline(0, color="k", lw=0.6)
    ax.text(-0.45, -0.85, "classical: linear", color=GREY, fontsize=14)
    ax.annotate("dense: $m^{n}$  (n = 8)", xy=(0.86, 0.34), xytext=(0.30, -0.45),
                color=DENSE, fontsize=14, arrowprops=dict(arrowstyle="->", color=DENSE, lw=1.3))
    ax.text(0.30, 0.86, "modern: $e^{\\beta m}$\n(β = 50)", color=MODERN, fontsize=14,
            ha="right")
    ax.set_xlabel("overlap m between cue and a stored memory")
    ax.set_ylabel("pull of that memory")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, 0, 1])
    save(fig, "s_sharpness")


def capacity():
    curves = load_json("exp1_capacity.json")["curves"]
    show = [("Classical", GREY, "classical"), ("Dense n=3", "#6baed6", "dense n = 3"),
            ("Dense n=4", DENSE, "dense n = 4"), ("Modern β=8", "#fdae6b", "modern β = 8"),
            ("Modern β=32", MODERN, "modern β = 32")]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for lab, col, name in show:
        cs = sorted([c for c in curves if c["model"] == lab and c["capacity"] > 0],
                    key=lambda c: c["N"])
        Ns = np.array([c["N"] for c in cs])
        caps = np.array([c["capacity"] for c in cs])
        lb = np.array([c["reached_end"] for c in cs])
        ax.plot(Ns[~lb], caps[~lb], "o-", ms=6, color=col)
        if lb.any():
            ax.plot(Ns[lb], caps[lb], "^", ms=9, color=col, mfc="white", mew=2)
            ax.plot([Ns[~lb][-1], Ns[lb][0]], [caps[~lb][-1], caps[lb][0]], ":", color=col)
        last = (Ns[~lb][-1], caps[~lb][-1]) if not lb.any() else (Ns[lb][-1], caps[lb][-1])
        label_end(ax, last[0], last[1], name, col, dx=1.15)
    Ns = np.array([16, 256])
    ax.plot(Ns, 0.138 * Ns, "--", color="k", lw=1.2)
    ax.text(70, 5.2, "0.138 N", fontsize=13, rotation=17)
    ax.plot([9.3], [5e5], "^", ms=9, color="#444444", mfc="white", mew=2)
    ax.text(11, 5e5, "never failed on the tested range", fontsize=12, color="#444444", va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(7, 900)
    ax.set_ylim(1.5, 1.5e6)
    ax.set_xlabel("number of neurons N")
    ax.set_ylabel("capacity (memories, 90% recall)")
    save(fig, "s_capacity")


def mnist_noise():
    res = load_json("exp2_mnist.json")["results"]
    show = [("Classical", GREY, "classical"), ("Dense n=4", "#9ecae1", "dense n = 4"),
            ("Dense n=16", DENSE, "dense n = 16"), ("Modern β=30", "#fdae6b", "modern β = 30"),
            ("Modern β=100", "#f16913", "modern β = 100"),
            ("Modern β=300", "#a63603", "modern β = 300")]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for lab, col, name in show:
        r = next(r for r in res if r["model"] == lab and r["cue"] == "noise")
        K = np.array([row["K"] for row in r["rows"]])
        y = np.array([row["correct"] for row in r["rows"]])
        ax.plot(K, y, "o-", ms=5, color=col)
    nn = next(r for r in res if r["model"] == "Nearest neighbour" and r["cue"] == "noise")
    ax.plot([row["K"] for row in nn["rows"]], [row["correct"] for row in nn["rows"]], "--",
            color="k", lw=1.2)
    # labels placed by hand where the curves are easy to tell apart
    ax.text(13, 0.04, "classical, dense n = 4", color=GREY, fontsize=13)
    ax.text(700, 0.30, "dense n = 16", color=DENSE, fontsize=13)
    ax.text(28, 0.30, "β = 30", color="#e08a3c", fontsize=13)
    ax.text(5000, 0.72, "β = 100", color="#f16913", fontsize=13)
    ax.text(1200, 1.05, "β = 300  =  nearest neighbour (dashed)", color="#a63603", fontsize=13)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlabel("number of stored digits K")
    ax.set_ylabel("fraction recalled correctly")
    save(fig, "s_mnist_noise")


def mnist_examples():
    z = np.load(RESULTS / "exp2_examples.npz", allow_pickle=True)
    labels = list(z["labels"])
    keep = ["Classical", "Dense n=16", "Modern β=30", "Modern β=300"]
    heads = ["stored", "cue", "classical", "dense\nn = 16", "modern\nβ = 30", "modern\nβ = 300"]
    rows = [("K100_noise", 1, "20% noise"), ("K1000_noise", 0, "20% noise"),
            ("K100_occlude", 2, "half hidden")]
    fig, axs = plt.subplots(len(rows), len(heads), figsize=(6.6, 3.9))
    for i, (key, j, rowname) in enumerate(rows):
        imgs = [z[f"{key}_target"][j], z[f"{key}_cue"][j]] + \
               [z[f"{key}_recalled"][labels.index(k)][j] for k in keep]
        for a, (ax, im) in enumerate(zip(axs[i], imgs)):
            im = im / max(np.max(np.abs(im)), 1e-12)
            ax.imshow(im.reshape(28, 28), cmap="gray_r", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(heads[a], fontsize=13)
        k = key.split("_")[0][1:]
        axs[i][0].set_ylabel(f"K = {k}\n{rowname}", fontsize=12)
    fig.subplots_adjust(wspace=0.06, hspace=0.12)
    save(fig, "s_mnist_examples")


def mnist_outcomes():
    res = load_json("exp2_mnist.json")["results"]
    show = [("Classical", "classical"), ("Dense n=4", "n = 4"), ("Dense n=16", "n = 16"),
            ("Modern β=30", "β = 30"), ("Modern β=100", "β = 100"), ("Modern β=300", "β = 300")]
    cats = [("correct", "#4d9a45", "correct"), ("mixup_same", "#9ecae1", "another real digit"),
            ("mixup_other", None, None), ("blend", "#c0392b", "blend (no real digit)")]
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    x = np.arange(len(show))
    bottom = np.zeros(len(show))
    for key, col, name in cats:
        vals = []
        for lab, _ in show:
            r = next(r for r in res if r["model"] == lab and r["cue"] == "noise")
            row = next((row for row in r["rows"] if row["K"] == 100), r["rows"][-1])
            vals.append(row[key])
        vals = np.array(vals)
        if key == "mixup_other":   # merge both mix-up kinds into one colour
            ax.bar(x, vals, bottom=bottom, color="#9ecae1", width=0.7)
        else:
            ax.bar(x, vals, bottom=bottom, color=col, width=0.7, label=name)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([s for _, s in show], fontsize=12)
    ax.set_ylabel("fraction of trials (K = 100)")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2,
              fontsize=12, handlelength=1.0, columnspacing=1.0)
    save(fig, "s_mnist_outcomes")


def drm():
    d = load_json("exp4_drm.json")
    res = d["results"]
    ks = d["list_lengths"]
    glove = {r["spec"]["beta"]: r for r in res if r["model"].endswith("(GloVe)")}

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    for b, col in ((1, "#fdae6b"), (3, MODERN), (5, "#a63603"), (10, "#7f2704")):
        y = [glove[b]["recall_by_length"][str(k)]["lure"] for k in ks]
        ax.plot(ks, y, "o-", ms=6, color=col)
        ax.text(15.4, y[-1] - (0.02 if b == 10 else 0), f"β = {b:g}", color=col, va="center", fontsize=13)
    cl = next(r for r in res if r["model"] == "Classical")
    ax.plot(ks, [cl["recall_by_length"][str(k)]["lure"] for k in ks], "s-", ms=5, color=GREY)
    ax.text(15.4, 0.06, "classical", color=GREY, va="center", fontsize=13)
    ax.plot([15], [HUMAN_MEAN_FALSE_RECALL], "*", ms=17, color="k", zorder=5)
    ax.annotate("humans (0.46)", xy=(15, HUMAN_MEAN_FALSE_RECALL), xytext=(11.3, 0.66), fontsize=13,
                arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    ax.set_xticks(ks)
    ax.set_xlim(2, 17.8)
    ax.set_ylim(-0.03, 0.75)
    ax.set_xlabel("studied words per list")
    ax.set_ylabel("P(recalling the lure)")
    save(fig, "s_drm_length")

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    bs = sorted(glove)
    fr = [glove[b]["recall_by_length"]["15"]["lure"] for b in bs]
    vr = [glove[b]["recall_by_length"]["15"]["veridical"] for b in bs]
    ax.plot(bs, fr, "o-", ms=6, color=MODERN)
    ax.plot(bs, vr, "o-", ms=6, color="#4d9a45")
    ax.axhline(HUMAN_MEAN_FALSE_RECALL, color="k", ls=":", lw=1.2)
    ax.text(30, HUMAN_MEAN_FALSE_RECALL + 0.04, "humans: 0.46", fontsize=13)
    ax.text(1.1, 0.70, "lure recalled", color=MODERN, fontsize=13)
    ax.text(14, 0.88, "a studied word\nrecalled", color="#4d9a45", fontsize=13)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("sharpness β")
    ax.set_ylabel("probability (15-word lists)")
    save(fig, "s_drm_beta")


def equations():
    """The three update rules typeset with mathtext, for the method slide."""
    fig = plt.figure(figsize=(6.4, 2.7))
    rows = [(GREY, r"classical:   $\sigma_i \leftarrow \mathrm{sgn}\left(\sum_\mu \xi^\mu_i\, m_\mu\right)$"),
            (DENSE, r"dense:   $\sigma_i \leftarrow \mathrm{sgn}\sum_\mu \left[F(\xi^\mu_i + m'_\mu) - F(-\xi^\mu_i + m'_\mu)\right],\ \ F(x)=x^n$"),
            (MODERN, r"modern:   $\xi^{\mathrm{new}} = X^{\top}\,\mathrm{softmax}(\beta\, X \xi)$")]
    for i, (col, tex) in enumerate(rows):
        fig.text(0.0, 0.84 - i * 0.34, tex, color=col, fontsize=17, va="center")
    fig.text(0.0, -0.08, r"$m_\mu$ = overlap of the state with memory $\mu$", fontsize=13,
             color="#444444")
    save(fig, "s_equations")


def synapse_budget():
    """Memories stored when every model gets the synapses of a classical network."""
    d = load_json("exp5_synapses.json")
    cap = {c["N"]: c["capacity"] for c in load_json("exp1_capacity.json")["curves"]
           if c["model"] == "Classical"}
    Ns = [r["N"] for r in d["matched_budget"] if r["model"] == "Classical" and r["N"] in cap]
    two = [r["K"] for r in d["matched_budget"] if r["model"] == "Modern β=32" and r["N"] in Ns]
    cl = [cap[N] for N in Ns]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(len(Ns))
    b1 = ax.bar(x - 0.2, cl, 0.38, color="#7f7f7f")
    b2 = ax.bar(x + 0.2, two, 0.38, color=MODERN)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}",
                    ha="center", fontsize=14)
    ax.text(-0.42, 118, "classical network (its capacity)", color="#5a5a5a", fontsize=14)
    ax.text(-0.42, 106, "modern / dense, same synapses (all recalled)", color=MODERN, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N = {N}\n{N * (N - 1) // 2:,} synapses".replace(",", " ") for N in Ns])
    ax.set_ylabel("memories stored")
    ax.set_ylim(0, 135)
    save(fig, "s_synapses")


def words():
    res = {r["model"]: r for r in load_json("exp3_words.json")["results"]}
    tasks = [("auto_recurrent", "auto, run to convergence", "#636363"),
             ("auto_onestep", "auto, one step", "#bdbdbd"),
             ("hetero_related", "hetero, related pairs", MODERN),
             ("hetero_unrelated", "hetero, unrelated pairs", "#fdae6b")]
    models = [("Dense n=4", "dense n = 4"), ("Modern β=10", "modern β = 10")]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    w = 0.2
    for i, (t, name, col) in enumerate(tasks):
        vals = [res[m]["tasks"][t]["capacity"] for m, _ in models]
        bars = ax.bar(np.arange(2) + (i - 1.5) * w, vals, w * 0.95, color=col, label=name)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 20, f"{b.get_height():.0f}",
                    ha="center", fontsize=12)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels([n for _, n in models])
    ax.set_ylabel("capacity (words or pairs)")
    ax.set_ylim(0, 1700)
    ax.legend(frameon=False, loc="upper right", fontsize=13, handlelength=1.0)
    save(fig, "s_words")


if __name__ == "__main__":
    for f in (classical_collapse, sharpness, capacity, mnist_noise, mnist_examples,
              mnist_outcomes, drm, equations, synapse_budget, words):
        f()
