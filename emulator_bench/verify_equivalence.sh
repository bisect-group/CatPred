#!/usr/bin/env bash
# Checks that the CatPred wrapper matches the original catpred code. Usage: verify_equivalence.sh [--gpu]
# Expected env: mldb (torch, rdkit, pytest, scikit-learn, scipy, tensorboard). The script does not create envs or install packages.
# CPU mode: tiny committed fixture (tests/fixtures), the original ESM2 650M path on the CPU and the original run_training.
#   It needs the esm2_t33_650M_UR50D weights in ~/.cache/torch/hub (downloaded on first use) and fair-esm 2.0.0
#   (unpacked from the pinned wheel under the test scratch folder). Any skipped test fails the run.
# --gpu mode: also runs the GPU tests on cuda:0. They need the EMULaToR data and the built ESM cache
#   (emulator_bench/cache_embeddings.py). They compare the fast trainer with the original loop on a real kcat split.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected conda env: mldb"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
pytest_args=(-q -rs -p no:cacheprovider --color=no -o addopts="")
out=$(CUDA_VISIBLE_DEVICES="" python -m pytest tests "${pytest_args[@]}" -m "not gpu" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if grep -Eq "[0-9]+ skipped" <<<"$out"; then
    echo "FAIL: tests were skipped"
    exit 1
fi
if [[ ${1:-} == --gpu ]]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m pytest tests "${pytest_args[@]}" -s -m gpu
fi
