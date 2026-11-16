#!/usr/bin/env bash
# Reproduce every result and figure of the report (about 30-40 min on a laptop).
# Use ./run_all.sh --quick for a ~2 minute smoke test with fewer seeds.
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q tests
cd experiments
python exp1_capacity.py "$@"
python exp2_mnist.py "$@"
python exp3_words.py "$@"
python exp4_drm.py "$@"
python exp5_synapses.py
python make_figures.py
