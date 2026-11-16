"""Data: random patterns, MNIST digits and GloVe word codes.

Downloads go to ``data/raw`` (git-ignored).  A small processed word file
``data/glove_subset.npz`` is committed so that the word experiments can be
re-run without the 130 MB GloVe download.
"""
from __future__ import annotations

import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np

from .drm_lists import DRM_LISTS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
GLOVE_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
             "glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz")


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


# --------------------------------------------------------------------------
# Random patterns and cue damage
# --------------------------------------------------------------------------
def random_patterns(K: int, N: int, rng) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0]), size=(K, N))


def flip_noise(P: np.ndarray, frac: float, rng) -> np.ndarray:
    """Flip exactly round(frac*N) randomly chosen signs in every row."""
    P = np.array(P, dtype=np.float64, copy=True)
    B, N = P.shape
    n_flip = int(round(frac * N))
    for b in range(B):
        idx = rng.choice(N, size=n_flip, replace=False)
        P[b, idx] *= -1
    return P


def occlude_bottom(P: np.ndarray, side: int = 28, frac: float = 0.5,
                   fill: float = -1.0) -> np.ndarray:
    """Hide the bottom part of square images (pixels set to background -1)."""
    P = np.array(P, dtype=np.float64, copy=True)
    rows = int(round(side * frac))
    img = P.reshape(-1, side, side)
    img[:, side - rows:, :] = fill
    return img.reshape(P.shape)


# --------------------------------------------------------------------------
# MNIST
# --------------------------------------------------------------------------
def load_mnist(binary: bool = True):
    """Return (patterns, labels) for the 60 000 MNIST training digits.

    Pixels > 0.5 become +1 (ink), the rest -1 (background).
    """
    img_f = _download(MNIST_URL + "train-images-idx3-ubyte.gz",
                      RAW / "train-images-idx3-ubyte.gz")
    lab_f = _download(MNIST_URL + "train-labels-idx1-ubyte.gz",
                      RAW / "train-labels-idx1-ubyte.gz")
    with gzip.open(img_f, "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    with gzip.open(lab_f, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8).astype(int)
    if binary:  # int8 keeps the full set small; models convert to float64
        return np.where(images > 127, 1, -1).astype(np.int8), labels
    return images.astype(np.float64) / 255.0, labels


def sample_mnist(X, y, K: int, rng):
    """Draw K digits with (as far as possible) equal numbers per class."""
    per = int(np.ceil(K / 10))
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False)
                          for c in range(10)])
    idx = rng.permutation(idx)[:K]
    return X[idx].astype(np.float64), y[idx]


# --------------------------------------------------------------------------
# Words (GloVe -> bipolar codes)
# --------------------------------------------------------------------------
def build_glove_subset(n_vocab: int = 20000) -> Path:
    """Keep the n_vocab most frequent alphabetic words plus every DRM word."""
    out = DATA / "glove_subset.npz"
    if out.exists():
        return out
    src = _download(GLOVE_URL, RAW / "glove-wiki-gigaword-100.gz")
    needed = set(DRM_LISTS) | {w for ws in DRM_LISTS.values() for w in ws}
    words, vecs = [], []
    with gzip.open(src, "rt", encoding="utf8") as f:
        next(f)  # header: "400000 100"
        for line in f:
            parts = line.rstrip().split(" ")
            w = parts[0]
            keep = (len(words) < n_vocab and w.isalpha()) or w in needed
            if keep and w not in words[-1:]:
                words.append(w)
                vecs.append(np.asarray(parts[1:], dtype=np.float32))
            if len(words) >= n_vocab and needed.issubset(words):
                break
    missing = needed.difference(words)
    if missing:
        print("warning: DRM words missing from GloVe:", sorted(missing))
    np.savez_compressed(out, words=np.array(words), vecs=np.stack(vecs))
    return out


class WordCodes:
    """Bipolar word patterns from GloVe vectors by random-hyperplane hashing.

    code(w) = sign(R (v_w - mean)), R ~ N(0,1)^{N x 100}.  The angle between two
    codes approximates the angle between the GloVe vectors (Charikar, 2002), so
    semantic similarity is kept while the patterns become +-1 like the
    neurons of the binary models.
    """

    def __init__(self, N: int = 512, seed: int = 0):
        d = np.load(build_glove_subset())
        self.words = [str(w) for w in d["words"]]
        self.index = {w: i for i, w in enumerate(self.words)}
        V = d["vecs"].astype(np.float64)
        self.vectors = V
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((V.shape[1], N))
        self.codes = np.sign((V - V.mean(axis=0)) @ R)
        self.codes[self.codes == 0] = 1.0
        self.N = N

    def __call__(self, words):
        return self.codes[[self.index[w] for w in words]]

    def has(self, w):
        return w in self.index

    def nearest_words(self, states, k: int = 1):
        """Decode states to the closest words in the whole lexicon (cosine)."""
        S = np.atleast_2d(states)
        S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        C = self.codes / np.sqrt(self.N)
        sims = S @ C.T
        top = np.argsort(-sims, axis=1)[:, :k]
        return [[self.words[j] for j in row] for row in top], np.take_along_axis(sims, top, 1)
