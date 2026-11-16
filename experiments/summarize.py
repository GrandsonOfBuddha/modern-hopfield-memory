"""Print the key numbers quoted in the report (and save results/summary.json)."""
import numpy as np

from common import load_json, save_json
from hopfield import capacity_from_curve


def main():
    out = {}
    d = load_json("exp1_capacity.json")
    print("== Exp 1: capacity on random patterns (10% noise, 90% criterion)")
    cap = {}
    for c in d["curves"]:
        cap.setdefault(c["model"], {})[c["N"]] = (round(c["capacity"], 1), c["reached_end"])
    for m, v in cap.items():
        print(f"  {m:14s}", "  ".join(f"N={N}:{k}{'+' if lb else ''}" for N, (k, lb) in sorted(v.items())))
    # log-log slopes for classical / dense, using sizes where capacity was measured
    slopes = {}
    for m, v in cap.items():
        pts = [(N, k) for N, (k, lb) in sorted(v.items()) if not lb and k > 0 and N >= 32]
        if len(pts) >= 2 and not m.startswith("Modern"):
            Ns, ks = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
            slopes[m] = float(np.polyfit(Ns, ks, 1)[0])
        if m.startswith("Modern"):
            pts = [(N, k) for N, (k, lb) in sorted(v.items()) if not lb and k > 0]
            if len(pts) >= 2:
                slopes[m + " (log K per neuron)"] = float(np.polyfit([p[0] for p in pts],
                                                                     np.log([p[1] for p in pts]), 1)[0])
    print("  slopes:", {k: round(s, 2) for k, s in slopes.items()})
    out["exp1"] = {"capacity": cap, "slopes": slopes}

    d = load_json("exp2_mnist.json")
    print(f"\n== Exp 2: MNIST (mean pairwise cosine {d['mean_pairwise_cosine']:.2f})")
    e2 = {}
    for r in d["results"]:
        K = [row["K"] for row in r["rows"]]
        y = [row["correct"] for row in r["rows"]]
        e2.setdefault(r["model"], {})[r["cue"]] = capacity_from_curve(K, y)
    for m, v in e2.items():
        print(f"  {m:18s} capacity noise={v['noise']:.0f}  occlude={v['occlude']:.0f}")
    conf = {}
    for r in d["results"]:
        if r["model"] == "Modern β=300" and r["cue"] == "occlude":
            C = np.array(r["confusion_other_class"])
            top = np.dstack(np.unravel_index(np.argsort(-C.ravel())[:6], C.shape))[0]
            conf = [(int(a), int(b), int(C[a, b])) for a, b in top]
            print("  most common cross-class mix-ups (cue -> recalled, count):", conf)
    out["exp2"] = {"capacity": e2, "occlude_confusions_beta300": conf}

    d = load_json("exp3_words.json")
    print("\n== Exp 3: words, auto vs hetero (N=512, 20% noise)", d["code_stats"])
    e3 = {}
    for r in d["results"]:
        e3[r["model"]] = {t: (round(v["capacity"]), v["success"][-1] >= 0.9)
                          for t, v in r["tasks"].items()}
        print(f"  {r['model']:14s}", {t: f"{c}{'+' if lb else ''}" for t, (c, lb) in e3[r['model']].items()})
    out["exp3"] = e3

    d = load_json("exp4_drm.json")
    print("\n== Exp 4: DRM")
    e4 = {}
    for r in d["results"]:
        e4[r["model"]] = {"lure_by_k": {k: round(v["lure"], 3) for k, v in r["recall_by_length"].items()},
                          "veridical_k15": round(r["recall_by_length"]["15"]["veridical"], 3),
                          "recog_lure_beats_studied": round(r["recognition"]["pct_studied_below_lure"], 3),
                          "recog_unrel_beats_studied": round(r["recognition"]["pct_studied_below_unrelated"], 3)}
        print(f"  {r['model']:22s}", e4[r["model"]])
    lc = d["lure_centroid"]
    n1 = sum(v["rank_overall"] == 1 for v in lc.values())
    print(f"  lure is the nearest GloVe word to its list centroid in {n1}/{len(lc)} lists")
    r3 = next(r for r in d["results"] if r["model"] == "Modern β=3 (GloVe)")
    fl = r3["false_recall_by_list_k15"]
    x = np.array([lc[l]["cosine"] for l in fl])
    y = np.array([fl[l] for l in fl])
    rho = float(np.corrcoef(x, y)[0, 1])
    print(f"  beta=3: per-list false recall vs lure-centroid cosine r={rho:.2f}; window={fl['window']}, "
          f"king={fl['king']}, cup={fl['cup']}")
    out["exp4"] = {"models": e4, "lure_nearest_centroid": n1, "n_lists": len(lc),
                   "beta3_corr_centroid": rho, "beta3_per_list": fl}

    d = load_json("exp5_synapses.json")
    print("\n== Exp 5: matched synapse budget")
    for r in d["matched_budget"]:
        print(f"  N={r['N']:4d} S={r['synapses']:6d} K={r['K']:4d} {r['model']:12s} success={r['success']:.2f}")
    out["exp5"] = d["matched_budget"]
    save_json(out, "summary.json")


if __name__ == "__main__":
    main()
