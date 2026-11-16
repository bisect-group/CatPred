#!/usr/bin/env bash
# Reproduces the CatPred paper runs: ESM2 + ProGRES caches, training (kcat, km, ki; 3 seeds), result tables.
# Usage: emulator_bench/run.sh [--dry-run] [cache|train|aggregate|all]   (train also evaluates)
# Env: PY GPUS RUNS_PER_GPU SEEDS DATA_ROOT OUT_ROOT EMULATOR_ROOT CACHE_DEVICE DRY_RUN
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STEP=all
for a in "$@"; do
  case $a in --dry-run) DRY_RUN=1 ;; cache | train | aggregate | all) STEP=$a ;; *) echo "bad argument: $a" >&2; exit 2 ;; esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

PY=${PY:-python}
GPUS=${GPUS:-1 2 3}
RUNS_PER_GPU=${RUNS_PER_GPU:-4}
SEEDS=${SEEDS:-666 777 888}
CACHE_DEVICE=${CACHE_DEVICE:-cuda:0}
EMULATOR_ROOT=${EMULATOR_ROOT:-$(realpath ../..)}
DATA_ROOT=${DATA_ROOT:-$EMULATOR_ROOT/data/processed/datasets/kinetic_dataset}
OUT_ROOT=${OUT_ROOT:-$EMULATOR_ROOT/outputs/baselines/CatPred}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CATPRED_BENCH_OUT_ROOT=$OUT_ROOT
P=(--data_root "$DATA_ROOT" --out_root "$OUT_ROOT")
EGNN=$OUT_ROOT/embeddings/progres_embeds/catpred_progres_embeds_dict.pt                # kcat, km
EGNN_KI=$OUT_ROOT/embeddings/progres_embeds/catpred_progres_embeds_dict_kcat_km_ki.pt  # ki
FLAGS=(--workers 1 --ensemble_size 1 --skip_pad --fused_attn)
B=emulator_bench

if want cache; then
  for s in 0 1; do run $PY $B/cache_embeddings.py "${P[@]}" --shard $s --num_shards 2 & done
  wait
  run $PY $B/cache_structures.py --out_root "$OUT_ROOT" --data_root "$DATA_ROOT" --tasks kcat km --device "$CACHE_DEVICE"
  run $PY $B/cache_structures.py --out_root "$OUT_ROOT" --data_root "$DATA_ROOT" --tasks kcat km ki --tag kcat_km_ki --device "$CACHE_DEVICE"
fi

if want train; then
  # shellcheck disable=SC2086
  run $PY $B/launch_parallel_retrain.py "${P[@]}" --tasks kcat km --seeds $SEEDS --gpus $GPUS --runs_per_gpu "$RUNS_PER_GPU" "${FLAGS[@]}" --egnn_path "$EGNN"
  # shellcheck disable=SC2086
  run $PY $B/launch_parallel_retrain.py "${P[@]}" --tasks ki --seeds $SEEDS --gpus $GPUS --runs_per_gpu "$RUNS_PER_GPU" "${FLAGS[@]}" --egnn_path "$EGNN_KI" \
    --split_groups random_splits_grouped_sequence random_splits_grouped_smiles enzyme_sequence_splits enzyme_structure_splits substrate_splits conformer_cosine_splits
  # Ki uniprot_time: 8% of test rows (viral polyproteins) have no structure, so the hit-rate floor is 0.9.
  for s in $SEEDS; do
    run mkdir -p "$OUT_ROOT/ki/uniprot_time_splits/all"
    CUDA_VISIBLE_DEVICES=${GPUS%% *} run $PY $B/train_run.py "${P[@]}" --task ki --split_group uniprot_time_splits --split_name all --seed "$s" \
      --esm_dir "$OUT_ROOT/embeddings/esm2_t33_650M_UR50D_v4" --epochs 30 "${FLAGS[@]}" --egnn_path "$EGNN_KI" --min_hit_rate 0.9
  done
fi

# shellcheck disable=SC2086
want aggregate && run $PY $B/aggregate_catpred_results.py "${P[@]}" --seeds $SEEDS
exit 0
