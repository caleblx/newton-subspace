#!/bin/bash
# everything in order, sequential so the timings don't share the GPU
# PY=./env/bin/python scripts/run_all.sh   (or PY=./gpu.sh on the cluster)
cd "$(dirname "$0")/.."
PY=${PY:-./env/bin/python}
set -e
$PY scripts/test_kernels.py
$PY scripts/test_correctness.py
$PY scripts/exp_basis_size.py        > results/log_exp_basis_size.txt 2>&1
$PY scripts/exp_stiffness.py         > results/log_exp_stiffness.txt 2>&1
$PY scripts/analyze_projection.py    > results/log_analyze_projection.txt 2>&1
$PY scripts/exp_cubature.py 4        > results/log_exp_cubature_4.txt 2>&1
$PY scripts/exp_solver_settings.py 2 > results/log_exp_solver_settings_2.txt 2>&1
$PY scripts/exp_resolution.py 1 2 3 4 5 > results/log_exp_resolution.txt 2>&1
$PY scripts/summarize.py
$PY scripts/make_figures.py
$PY scripts/render_comparison.py 3 twist
$PY scripts/render_comparison.py 3 diag_load
$PY scripts/export_playback.py 3
$PY scripts/build_demo.py bunny
