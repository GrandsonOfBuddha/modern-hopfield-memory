# When Does Memory Break?
### Sharpness, capacity and false memories in classical, dense and modern Hopfield networks

Final project for *Introduction to Neural and Cognitive Modelling* (2026). Author: Soham Chatterjee (2023101092).

This repository compares three associative-memory models on the same memories and the same damaged cues:

| Model | Recall rule | "Sharpness" dial |
|---|---|---|
| Classical Hopfield network (Hopfield 1982) | every memory pulls in proportion to its overlap | none (linear) |
| Dense associative memory (Krotov & Hopfield 2016) | energy `-Σ F(ξ·σ)`, `F(x)=xⁿ` | power `n` |
| Modern Hopfield network (Ramsauer et al. 2021) | `ξ_new = Xᵀ softmax(β X ξ)` (= attention) | inverse temperature `β` |

It asks three questions:

* **Q1 – capacity and cost.** Do modern networks escape the `0.14N` limit, and do they still win when we count synapses?
* **Q2 – false memories.** Do they mix up similar memories or "remember" things never stored, like people do in the DRM paradigm?
* **Q3 – auto vs hetero.** Is recalling a paired partner harder than restoring the item itself?

The report is in [`report/report.pdf`](report/report.pdf) and the slides are in [`slides/`](slides/).

---

## Main results

| | Result |
|---|---|
| Random patterns | Classical capacity ≈ 0.12N (theory 0.138N). Dense memories scale as N^2.2 (n=3) and N^3.4 (n=4). Modern capacity is exponential: ln K ≈ 0.45N, and K > 2¹⁷ for N ≥ 32. At β=8 it stops at about 130 memories whatever N is. |
| Synapse cost | Two-layer modern/dense networks need N synapses per memory. At the synapse budget of a classical network they store 0.5N instead of 0.12N memories, a constant factor of about 4, not an exponential gain. |
| MNIST | The classical network fails at K=10 by falling into one spurious "average digit". With β ≥ 300 the modern network equals a perfect nearest-neighbour lookup up to 20 000 digits. Blunt models make *blends*; sharp models make *mix-ups* with real digits. |
| Words | Hetero-association with unrelated pairs is **easier** than auto-association (dense n=4: 1371 vs 1040). Related pairs are hardest (702). Iterating the dynamics never helped. |
| DRM | On GloVe vectors a blunt modern network (β≈3) falsely recalls the lure with probability 0.48 (humans: 0.46). False recall grows with list length, as in Robinson & Roediger (1997). From β ≥ 10 it disappears. |

---

## Quick start

```bash
git clone https://github.com/GrandsonOfBuddha/modern-hopfield-memory.git
cd modern-hopfield-memory
python -m pip install -r requirements.txt     # numpy, matplotlib, pytest
python -m pytest -q tests                     # 8 unit tests, < 1 s
```

### Live demo (used in the presentation)

```bash
python demo.py digits --K 500 --cue noise     # damaged digit -> what each model recalls
python demo.py digits --K 100 --cue occlude   # bottom half hidden
python demo.py drm --list sleep               # does the network "remember" SLEEP?
python demo.py words --K 1000                 # hetero-associative word-pair recall
```

`demo.py digits` opens a window. Add `--save out.png` to write the figure to a file instead. On the first run MNIST (about 10 MB) is downloaded to `data/raw/`.

### Reproduce every number and figure

```bash
./run_all.sh            # about 40 min on a 10-core laptop
./run_all.sh --quick    # about 2 min smoke test (fewer seeds; OVERWRITES results/)
```

Or run single experiments from the repository root:

| Script | What it does | Output |
|---|---|---|
| `experiments/exp1_capacity.py` | capacity vs N on random patterns | `results/exp1_capacity.json` |
| `experiments/exp2_mnist.py` | MNIST recall, mix-ups and blends | `results/exp2_mnist.json`, `exp2_examples.npz` |
| `experiments/exp3_words.py` | auto vs hetero association with words | `results/exp3_words.json` |
| `experiments/exp4_drm.py` | DRM false recall and recognition | `results/exp4_drm.json` |
| `experiments/exp5_synapses.py` | synapse-matched comparison (needs exp1) | `results/exp5_synapses.json` |
| `experiments/make_figures.py` | all report figures | `figures/*.pdf`, `figures/*.png` |
| `experiments/make_slide_figures.py` | larger versions of the figures for the slides | `figures/slides/*.png` |
| `experiments/summarize.py` | prints every number quoted in the report | `results/summary.json` |

Logs from the runs used in the report are in `results/logs/`.

---

## Repository layout

```
hopfield/            the models and data code
  models.py          ClassicalHopfield, DenseAM, ModernHopfield, LinearAssociator, NearestNeighbour
  data.py            random patterns, cue damage, MNIST loader, GloVe -> ±1 word codes
  metrics.py         success / mix-up / blend scoring, capacity estimate
  drm_lists.py       31 DRM lists (Roediger & McDermott 1995; Stadler et al. 1999)
experiments/         one script per experiment + figures + summary
tests/               unit tests (e.g. dense n=2 == classical, energy never increases)
data/glove_subset.npz  20 077 GloVe words (100-d) used in the word experiments (7 MB)
results/             JSON results of the runs in the report
figures/             figures of the report
report/              LaTeX source and PDF of the report
slides/              presentation
demo.py              live demo
```

## Data

* **MNIST** (LeCun et al., 1998): downloaded automatically from the PyTorch mirror `ossci-datasets.s3.amazonaws.com/mnist/`. Pixels > 0.5 become +1 (ink), the rest −1.
* **GloVe** 6B 100-d (Pennington et al., 2014), via the gensim-data release `glove-wiki-gigaword-100` (130 MB). The processed subset `data/glove_subset.npz` is included, so the download is only needed to rebuild it (delete the file and run any word experiment).
* **DRM lists**: transcribed into `hopfield/drm_lists.py`. Human reference values (Stadler et al., 1999): window 0.65, king 0.10, mean over 36 lists 0.46.

## Model weights

These models have no trained parameters: the stored patterns *are* the weights (`W = XᵀX/N` for the classical network, and the memory matrix `X` for the dense and modern networks). Every pattern set comes from a fixed random seed, so all memories are regenerated exactly when the scripts run.

## Implementation notes

* Binary models use asynchronous updates in random order until no neuron changes (at most 30 sweeps), vectorised over a batch of cues.
* The dense update uses the exact energy difference `Σ F(ξᵢ + m') − F(−ξᵢ + m')` with incrementally updated overlaps, so a sweep costs O(NKB).
* Modern-network patterns are scaled to unit length, so β means the same thing on every data set.
* Success means the nearest stored memory is the target **and** cosine ≥ 0.95. Capacity is the largest K with ≥ 90% success (log-interpolated). Each point uses 20 pattern sets.

## References

Hopfield (1982) PNAS · Amit, Gutfreund & Sompolinsky (1985) PRL · Krotov & Hopfield (2016) NeurIPS · Demircigil et al. (2017) J Stat Phys · Ramsauer et al. (2021) ICLR · Krotov & Hopfield (2021) ICLR · Millidge et al. (2022) ICML · Roediger & McDermott (1995) JEP:LMC · Robinson & Roediger (1997) Psych Sci · Stadler, Roediger & McDermott (1999) Mem Cogn · Arndt & Hirshman (1998) JML · Charikar (2002) STOC.

The authors' code ([Krotov](https://github.com/DimaKrotov/Dense_Associative_Memory), [Ramsauer et al.](https://github.com/ml-jku/hopfield-layers)) was **not** used. All models were implemented from the equations in the papers.
