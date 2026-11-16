"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# One BLAS thread per process: we parallelise over jobs instead.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# One colour per model family, darker = sharper.
COLORS = {
    "Classical": "#7f7f7f",
    "Dense n=3": "#9ecae1", "Dense n=4": "#6baed6", "Dense n=8": "#3182bd",
    "Dense n=16": "#08519c",
}
MODERN_CMAP = plt.get_cmap("Oranges")


def color_for(label: str, betas=None) -> str:
    if label in COLORS:
        return COLORS[label]
    if label.startswith("Modern"):
        b = float(label.split("=")[1])
        betas = betas or [b]
        lo, hi = min(betas), max(betas)
        t = 0.5 if hi == lo else (math.log(b) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return matplotlib.colors.to_hex(MODERN_CMAP(0.4 + 0.55 * t))
    return "black"


def save_json(obj, name: str):
    path = RESULTS / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print("saved", path)


def load_json(name: str):
    with open(RESULTS / name) as f:
        return json.load(f)


def savefig(fig, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    print("saved", FIGURES / f"{name}.png")
