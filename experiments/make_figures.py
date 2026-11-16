"""Make every figure of the report from the saved results/*.json files.

Run:  python experiments/make_figures.py
"""
import numpy as np

from common import plt, load_json, savefig, color_for, RESULTS
from hopfield.drm_lists import HUMAN_MEAN_FALSE_RECALL

SINGLE = 3.45  # IEEE column width (inches)
DOUBLE = 7.1


def modern_betas(labels):
    return [float(l.split("=")[1].split()[0]) for l in labels if l.startswith("Modern")]


# --------------------------------------------------------------------------
def fig_sharpness():
    """How strongly each model weights a memory as a function of its overlap."""
    m = np.linspace(-1, 1, 401)
    fig, ax = plt.subplots(figsize=(SINGLE, 1.7))
    curves = [("Classical", m, color_for("Classical")),
              ("Dense n=3", 3 * m ** 2, color_for("Dense n=3")),
              ("Dense n=8", 8 * m ** 7, color_for("Dense n=8"))]
    for lab, w, c in curves:
        ax.plot(m, w / np.max(np.abs(w)), color=c, lw=1.4, label=lab)
    for b in (2, 10, 50):
        w = np.exp(b * (m - 1))
        ax.plot(m, w, color=color_for(f"Modern β={b}", [2, 50]), lw=1.4,
                label=f"Modern β={b}")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("overlap of cue with a stored memory (cosine)")
    ax.set_ylabel("pull of that memory\n(normalised)")
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.set_ylim(-1.05, 1.05)
    savefig(fig, "fig1_sharpness")


# --------------------------------------------------------------------------
def fig_capacity():
    d = load_json("exp1_capacity.json")
    syn = load_json("exp5_synapses.json")
    curves = d["curves"]
    labels = sorted({c["model"] for c in curves}, key=lambda l: (l[0] != "C", l))
    betas = modern_betas(labels)
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.05))

    ax = axs[0]
    for lab in labels:
        c = next((c for c in curves if c["model"] == lab and c["N"] == 64), None)
        if c is None:
            continue
        ax.plot(np.array(c["K"]) / 64, c["success"], "-", ms=2, lw=1.3,
                color=color_for(lab, betas), label=lab)
    ax.axvline(0.138, ls=":", color="k", lw=0.8)
    ax.text(0.15, 0.05, "0.138N", fontsize=6.5)
    ax.axhline(0.9, ls="--", color="grey", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("load K / N   (N = 64)")
    ax.set_ylabel("recall success")
    ax.set_title("(a) recall vs number of memories")
    ax.legend(frameon=False, fontsize=5.5, loc="lower right")

    ax = axs[1]
    for lab in labels:
        cs = sorted([c for c in curves if c["model"] == lab and c["capacity"] > 0],
                    key=lambda c: c["N"])
        Ns = np.array([c["N"] for c in cs])
        caps = np.array([c["capacity"] for c in cs])
        lb = np.array([c["reached_end"] for c in cs])
        col = color_for(lab, betas)
        ax.plot(Ns[~lb], caps[~lb], "o-", ms=3, lw=1.2, color=col, label=lab)
        if lb.any():
            ax.plot(Ns[lb], caps[lb], "^", ms=4, color=col, mfc="none")
    Ns = np.array([16, 256])
    ax.plot(Ns, 0.138 * Ns, ":", color="k", lw=0.8, label="0.138N (theory)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of neurons N")
    ax.set_ylabel("capacity (90% recall)")
    ax.set_title("(b) capacity grows with N")
    ax.text(2 ** 6.3, 4, "0.138N", fontsize=6)
    ax.text(2 ** 3.1, 3e4, "△ = lower bound", fontsize=6)

    ax = axs[2]
    rows = syn["per_synapse"]
    for lab in labels:
        rs = sorted([r for r in rows if r["model"] == lab], key=lambda r: r["N"])
        if not rs:
            continue
        ax.plot([r["N"] for r in rs], [r["memories_per_synapse"] * r["N"] for r in rs],
                "o-", ms=3, lw=1.2, color=color_for(lab, betas), label=lab)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("number of neurons N")
    ax.set_ylabel("memories per synapse  × N")
    ax.set_title("(c) cost: memories per synapse")
    ax.text(2 ** 3.1, 0.88, "dense & modern: one hidden neuron\n(N synapses) per memory",
            fontsize=6, va="top")
    ax.text(2 ** 3.1, 0.45, "classical: N(N−1)/2 synapses", fontsize=6, color="#555555")
    fig.tight_layout(w_pad=1.0)
    savefig(fig, "fig2_capacity")


# --------------------------------------------------------------------------
def fig_mnist():
    d = load_json("exp2_mnist.json")
    res = d["results"]
    labels = list(dict.fromkeys(r["model"] for r in res))
    betas = modern_betas(labels)
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.05),
                            gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, cue, title in ((axs[0], "noise", "(a) 20% pixels flipped"),
                           (axs[1], "occlude", "(b) bottom half hidden")):
        for lab in labels:
            r = next(r for r in res if r["model"] == lab and r["cue"] == cue)
            K = [row["K"] for row in r["rows"]]
            y = [row["correct"] for row in r["rows"]]
            nn = lab == "Nearest neighbour"
            ax.plot(K, y, "--" if nn else "-", lw=1.0 if nn else 1.3,
                    marker="" if nn else ("x" if lab == "Classical" else "o"), ms=2.5,
                    color="k" if nn else color_for(lab, betas), label=lab)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("stored digits K")
        ax.set_title(title)
    axs[0].set_ylabel("recall success")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6, ncol=9, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), handlelength=1.5, columnspacing=1.0)

    ax = axs[2]
    K_show = 100
    cats = [("correct", "#4daf4a", "correct"), ("mixup_same", "#a6cee3", "other digit, same class"),
            ("mixup_other", "#1f78b4", "other digit, other class"), ("blend", "#e41a1c", "blend / spurious")]
    show = [l for l in labels if l != "Modern β=1000"]
    x = np.arange(len(show))
    bottom = np.zeros(len(show))
    for key, col, name in cats:
        vals = []
        for lab in show:
            r = next(r for r in res if r["model"] == lab and r["cue"] == "noise")
            row = next((row for row in r["rows"] if row["K"] == K_show), None)
            if row is None:  # recall had already collapsed: use last measured point
                row = r["rows"][-1]
            vals.append(row[key])
        ax.bar(x, vals, bottom=bottom, color=col, width=0.75, label=name)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("Modern ", "").replace("Dense ", "").replace("Nearest neighbour", "NN")
                        .replace("Classical", "Class.") for l in show], rotation=45, fontsize=6)
    ax.set_ylabel("fraction of trials")
    ax.set_title(f"(c) outcomes, K = {K_show}, noisy cues")
    ax.legend(frameon=False, fontsize=5.3, loc="upper left", ncol=2, columnspacing=0.6,
              handlelength=1.0)
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.5, 1])
    fig.tight_layout(w_pad=0.8)
    savefig(fig, "fig3_mnist")


def fig_mnist_examples():
    z = np.load(RESULTS / "exp2_examples.npz", allow_pickle=True)
    labels = list(z["labels"])
    keep = ["Classical", "Dense n=8", "Dense n=16", "Modern β=30", "Modern β=300"]
    cols = ["target", "cue"] + keep
    rows = [("K100_noise", 0), ("K100_noise", 1), ("K1000_noise", 0), ("K100_occlude", 2),
            ("K1000_occlude", 0)]
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(SINGLE, 0.55 * len(rows) + 0.25))
    for i, (key, j) in enumerate(rows):
        imgs = [z[f"{key}_target"][j], z[f"{key}_cue"][j]] + \
               [z[f"{key}_recalled"][labels.index(k)][j] for k in keep]
        for a, (ax, im) in enumerate(zip(axs[i], imgs)):
            im = im / max(np.max(np.abs(im)), 1e-12)  # modern states are unit length
            ax.imshow(im.reshape(28, 28), cmap="gray_r", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(cols[a].replace("Modern ", "").replace("Dense ", ""), fontsize=6)
        axs[i][0].set_ylabel(key.split("_")[0].replace("K", "K="), fontsize=6)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    savefig(fig, "fig4_mnist_examples")


# --------------------------------------------------------------------------
def fig_words():
    d = load_json("exp3_words.json")
    res = d["results"]
    tasks = [("auto_recurrent", "auto (recurrent)", "#636363"),
             ("auto_onestep", "auto (one step)", "#bdbdbd"),
             ("hetero_related", "hetero, related pairs", "#e6550d"),
             ("hetero_unrelated", "hetero, unrelated pairs", "#fdae6b")]
    fig, ax = plt.subplots(figsize=(SINGLE, 1.9))
    x = np.arange(len(res))
    w = 0.2
    for i, (t, name, col) in enumerate(tasks):
        caps = []
        for r in res:
            c = r["tasks"][t]["capacity"]
            caps.append(max(c, 1))
        bars = ax.bar(x + (i - 1.5) * w, caps, w, color=col, label=name)
        for b, r in zip(bars, res):
            if r["tasks"][t]["success"][-1] >= 0.9:  # never failed: lower bound
                ax.text(b.get_x() + w / 2, b.get_height() * 1.1, "≥", ha="center",
                        fontsize=5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r["model"].replace("Modern ", "").replace("Dense ", "") for r in res],
                       fontsize=6.5)
    ax.set_ylabel("capacity (words or pairs)")
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper left")
    ax.set_ylim(5, 4e4)
    savefig(fig, "fig5_words")


# --------------------------------------------------------------------------
def fig_drm():
    d = load_json("exp4_drm.json")
    res = d["results"]
    by = {r["model"]: r for r in res}
    glove = [r for r in res if r["model"].endswith("(GloVe)")]
    codes_modern = [r for r in res if r["model"].startswith("Modern") and "GloVe" not in r["model"]]
    binary = [r for r in res if not r["model"].startswith("Modern")]
    ks = d["list_lengths"]
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.05))

    ax = axs[0]
    gb = [r["spec"]["beta"] for r in glove]
    for r in glove:
        if r["spec"]["beta"] in (1, 3, 5, 7, 10):
            ax.plot(ks, [r["recall_by_length"][str(k)]["lure"] for k in ks], "o-", ms=3,
                    color=color_for(f"Modern β={r['spec']['beta']}", gb),
                    label=f"Modern β={r['spec']['beta']:g}")
    r = by["Classical"]
    ax.plot(ks, [r["recall_by_length"][str(k)]["lure"] for k in ks], "s-", ms=3,
            color=color_for("Classical"), label="Classical (codes)")
    ax.set_xlabel("studied associates k")
    ax.set_ylabel("P(recall critical lure)")
    ax.set_title("(a) false recall grows with list length")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", ncol=2, columnspacing=0.8)
    ax.set_ylim(-0.02, 0.95)
    ax.set_xticks(ks)

    ax = axs[1]
    b = [r["spec"]["beta"] for r in glove]
    ax.plot(b, [r["recall_by_length"]["15"]["lure"] for r in glove], "o-", ms=3,
            color="#d95f0e", label="false recall")
    ax.plot(b, [r["recall_by_length"]["15"]["veridical"] for r in glove], "o-", ms=3,
            color="#31a354", label="veridical")
    ax.plot(b, [r["recognition"]["pct_studied_below_lure"] for r in glove], "s--", ms=3,
            color="#756bb1", label="lure more\nfamiliar")
    ax.axhline(HUMAN_MEAN_FALSE_RECALL, color="k", ls=":", lw=0.8)
    ax.text(11, HUMAN_MEAN_FALSE_RECALL + 0.03, "human false recall (0.46)", fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlabel("sharpness β (modern network, GloVe)")
    ax.set_ylabel("probability (k = 15)")
    ax.set_title("(b) sharp recall removes false memory")
    ax.legend(frameon=False, fontsize=5.3, loc="lower right", bbox_to_anchor=(1.03, 0.05),
              handlelength=1.2)

    ax = axs[2]
    names = ["Classical", "Dense n=3", "Dense n=4", "Dense n=8"] + \
            [r["model"] for r in codes_modern if r["spec"]["beta"] in (1, 2, 5, 20)] + \
            [r["model"] for r in glove if r["spec"]["beta"] in (1, 2, 5, 20)]
    vals = [by[n]["recognition"]["pct_studied_below_lure"] for n in names]
    cols = [color_for(n.split(" (")[0], [1, 100]) for n in names]
    ax.barh(np.arange(len(names)), vals, color=cols)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([n.replace("Modern ", "Modern ").replace(" (GloVe)", ", GloVe")
                        if "GloVe" in n else n + (", codes" if n.startswith("Modern") else "")
                        for n in names], fontsize=5.5)
    ax.invert_yaxis()
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel("P(lure feels more familiar\nthan a studied word)")
    ax.set_title("(c) false recognition")
    fig.tight_layout(w_pad=0.8)
    savefig(fig, "fig6_drm")


def fig_synapse_budget():
    """Slide figure: memories stored at the synapse budget of a classical network."""
    d = load_json("exp5_synapses.json")
    cap = {c["N"]: c["capacity"] for c in load_json("exp1_capacity.json")["curves"]
           if c["model"] == "Classical"}
    Ns = [r["N"] for r in d["matched_budget"] if r["model"] == "Classical" and r["N"] in cap]
    two_layer = [r["K"] for r in d["matched_budget"] if r["model"] == "Modern β=32" and r["N"] in Ns]
    classical = [cap[N] for N in Ns]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = np.arange(len(Ns))
    b1 = ax.bar(x - 0.2, classical, 0.38, color=color_for("Classical"), label="classical (capacity)")
    b2 = ax.bar(x + 0.2, two_layer, 0.38, color="#e6550d",
                label="dense / modern, same synapses\n(all recalled correctly)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N = {N}\n{N * (N - 1) // 2:,} synapses" for N in Ns])
    ax.set_ylabel("memories stored")
    ax.legend(frameon=False, loc="upper left")
    savefig(fig, "slide_synapse_budget")


if __name__ == "__main__":
    for f in (fig_sharpness, fig_capacity, fig_mnist, fig_mnist_examples, fig_words, fig_drm,
              fig_synapse_budget):
        try:
            f()
        except FileNotFoundError as e:
            print(f"skipping {f.__name__}: {e}")
