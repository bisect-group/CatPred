# CatPred EMULaToR bench

This folder runs CatPred with its original published settings on the EMULaToR kinetic splits.
It does only multi-seed evaluation. The model, loss, features and hyperparameters are the original ones.

## Scope

- Tasks: `kcat`, `km`, `ki` from `data/processed/datasets/kinetic_dataset/<task>`. Unified datasets are never used.
- Splits: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `enzyme_sequence_splits/threshold_<t>`, `substrate_splits/threshold_<t>`. Thresholds come from disk.
- Seeds: 666, 777, 888.
- Enzyme features (as in `scripts/reproduce_fig5_catpred.sh`): Seq-Attn plus ESM2 for kcat and km. Seq-Attn plus proGRES (E-GNN) for ki.

## Settings

The defaults of `train_single_target_tvt.py` are the original ones: 30 epochs, batch size 16, ensemble of 10, MVE loss,
Adam (init 1e-4, max 1e-3, final 1e-4, 2 warmup epochs, Noam schedule), dropout 0, seq_embed_dim 36, 6 attention heads,
sequence length 2048, fp32. The best epoch is chosen by validation RMSE.

## Work tree

The dataset folder is read only. Staged splits, caches and results go in `--base_dir`
(default `<EMULaToR>/outputs/baselines/CatPred`, or `CATPRED_BENCH_BASE_DIR`).

- `<base_dir>/<task>/<split>/{train,val,test}.parquet`: staged splits with a structure column.
- `<base_dir>/embeddings/esm2`: ESM2 tensors per unique sequence.
- `<base_dir>/embeddings/progres`: proGRES features per structure.
- `<base_dir>/embeddings/fair_esm`: isolated Meta fair-esm install. Override with `CATPRED_FAIR_ESM_PATH`.
- `<base_dir>/<task>/<split>/catpred_results/original_defaults/seed_<s>/`: results and checkpoints.

## Steps

1. `align_structures.py` stages the splits and picks one structure per row (PDBe, then AlphaFold, then ESMFold2 fallback). Ki rows need a structure. Kcat and km use sequences only.
2. `cache_embeddings.py` builds the ESM2 cache (kcat, km) and runs proGRES once (ki).
3. `train_single_target_tvt.py` trains and evaluates one seed on one split.
4. `launch_parallel_retrain.py` runs steps 1 to 3 for all seeds, splits and tasks. It can use several GPUs and resumes finished runs.
5. `aggregate_catpred_results.py` writes mean and std over seeds.

## Speed-ups that keep behaviour

Inline data loading, ESM feature cache, a fused-Adam option and a faster ESM forward pass. These do not change the model or the settings.
The ESM cache embeds one sequence per call, as the original does. Its key carries a version string, so older caches (batch-padded) are not reused.
The trainer runs the native per-member test predict, as the original does. It uses torch RNG, so skipping it would change later ensemble members. The `--skip_native_test_eval` flag is removed.
Fused Adam is off by default (`--optimizer_fused off`) because it is not bit-identical to the original optimizer.
Rows whose sequence has a residue outside the original vocabulary (the original raises `KeyError`) are dropped from every split. The counts are in `run_summary.csv` (`dropped_unknown_residue_*`).
Mixed precision is off by default because the original code used fp32.

## Smoke tests

`--limit_rows N`, `--epochs` and `--ensemble_size` on the launcher shrink a run. Do not use them for results.

See `commands.txt` for the commands.
