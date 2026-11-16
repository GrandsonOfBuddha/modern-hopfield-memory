import numpy as np

from hopfield import (ClassicalHopfield, DenseAM, ModernHopfield, LinearAssociator,
                      cosine, capacity_from_curve, classify_outcomes)
from hopfield.data import random_patterns, flip_noise, occlude_bottom


def test_dense_n2_equals_classical():
    """Dense AM with F(x)=x^2 must follow exactly the classical dynamics."""
    rng = np.random.default_rng(0)
    X = random_patterns(12, 100, rng)
    cues = flip_noise(X[:5], 0.2, rng)
    a = ClassicalHopfield(X).recall(cues, rng=1)
    b = DenseAM(X, n=2).recall(cues, rng=1)
    assert np.array_equal(a, b)


def test_classical_recalls_below_capacity():
    rng = np.random.default_rng(1)
    X = random_patterns(5, 200, rng)  # load 0.025, far below 0.138
    out = ClassicalHopfield(X).recall(flip_noise(X, 0.1, rng), rng=0)
    assert np.allclose(cosine(out, X).diagonal(), 1.0)


def test_classical_energy_never_increases():
    rng = np.random.default_rng(2)
    X = random_patterns(20, 100, rng)
    net = ClassicalHopfield(X)
    s = flip_noise(X[:1], 0.3, rng)
    e0 = net.energy(s)[0]
    s1 = net.recall(s, max_sweeps=1, rng=0)
    assert net.energy(s1)[0] <= e0 + 1e-9


def test_dense_beats_classical_above_classical_capacity():
    rng = np.random.default_rng(3)
    X = random_patterns(60, 100, rng)  # load 0.6 >> 0.138
    cues = flip_noise(X[:20], 0.1, rng)
    t = np.arange(20)
    c, _, _ = classify_outcomes(ClassicalHopfield(X).recall(cues, rng=0), X, t)
    d, _, _ = classify_outcomes(DenseAM(X, n=3).recall(cues, rng=0), X, t)
    assert np.mean(d == "correct") > 0.9 > np.mean(c == "correct")


def test_modern_one_step_retrieval():
    rng = np.random.default_rng(4)
    X = random_patterns(2000, 64, rng)
    net = ModernHopfield(X, beta=64)
    cues = flip_noise(X[:50], 0.1, rng)
    out = net.recall_one_step(cues)
    lab, _, _ = classify_outcomes(out, X, np.arange(50))
    assert np.mean(lab == "correct") > 0.95


def test_modern_low_beta_blends():
    rng = np.random.default_rng(5)
    X = random_patterns(50, 64, rng)
    out = ModernHopfield(X, beta=0.5).recall(X[:5])
    lab, _, _ = classify_outcomes(out, X, np.arange(5))
    assert np.all(lab == "blend")


def test_hetero_readouts():
    rng = np.random.default_rng(6)
    A = random_patterns(10, 200, rng)
    B = random_patterns(10, 200, rng)
    for net in (LinearAssociator(A, B), DenseAM(A, n=3), ModernHopfield(A, beta=50)):
        out = net.hetero_one_step(A, B)
        assert np.all(np.argmax(cosine(out, B), axis=1) == np.arange(10))


def test_occlusion_and_capacity_helpers():
    img = np.ones((1, 784))
    occ = occlude_bottom(img)
    assert occ.reshape(28, 28)[14:].max() == -1 and occ.reshape(28, 28)[:14].min() == 1
    assert capacity_from_curve([10, 20, 40], [1.0, 1.0, 0.5]) > 20
    assert capacity_from_curve([10, 20], [0.5, 0.2]) == 0.0
    assert capacity_from_curve([10, 20], [1.0, 1.0]) == 20
