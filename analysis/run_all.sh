#!/bin/sh
# Reproduce the technical report's log-derived numbers and figures. Run from analysis/: sh run_all.sh
# Needs Python 3 with matplotlib, numpy, scipy and pillow (pip install matplotlib numpy scipy pillow).
# Outputs: stats.json, stats_logs.json, esr_fit.json, data/*.csv, figures/*.{pdf,png} (all in analysis/).
set -e
PY=${PYTHON:-python3}
$PY -I scripts/analyze.py > /dev/null
$PY -I scripts/guidance_check.py > /dev/null
$PY -I scripts/esr_fit.py > /dev/null
$PY -I scripts/build_stats.py
$PY -I scripts/make_figures.py
