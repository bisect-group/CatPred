# CatPred emulator_bench report

## Method summary

CatPred (Boorla and Maranas, Nat. Commun. 2025) predicts kcat, Km and Ki from an enzyme sequence and a substrate SMILES.
A D-MPNN encodes the substrate.
The enzyme side uses Seq-Attn (rotary embedding plus multi-head attention).
It can add ESM2 features (1280-d) and proGRES E-GNN structure features (128-d).
The model has a probabilistic head trained with the MVE loss.
The final predictor is an ensemble of 10 models.
Released models: kcat and Km use Seq-Attn plus ESM2. Ki uses Seq-Attn plus proGRES (scripts/reproduce_fig5_catpred.sh).
The wrapper now only runs multi-seed evaluation with these original settings.

## Original settings

| Setting | Value | Source |
|---|---|---|
| Epochs | 30 | scripts/reproduce_fig5_catpred.sh |
| Batch size | 16 | same script (the paper does not state it) |
| Ensemble | 10 | same script, paper |
| Loss | mve | same script, paper |
| seq_embed_dim / heads | 36 / 6 | same script |
| init / max / final lr | 1e-4 / 1e-3 / 1e-4 | catpred train args defaults |
| Warmup | 2 epochs, Noam scheduler | catpred train args defaults |
| Dropout, weight decay | 0, 0 | catpred train args defaults |
| Metric, model selection | rmse, best validation epoch checkpoint | catpred run_training |
| Precision | fp32 | original code |
| Sequence length | 2048 | catpred esm_utils |
| Seeds | 666 777 888 | task rule (paper seeds) |

## What the wrapper does now

`launch_parallel_retrain.py` runs, for each seed, split and task:
1. `align_structures.py` stages the splits into `--base_dir` and picks one structure per row. The dataset tree is read only.
2. `cache_embeddings.py` builds the ESM2 cache (kcat, km) and runs proGRES once (ki).
3. `train_single_target_tvt.py` trains and evaluates. Features follow `ENZYME_FEATURES` in `common.py`.
4. `aggregate_catpred_results.py` writes mean and std over seeds.

Splits are discovered from disk: the two grouped random splits, `uniprot_time_splits`, and every `enzyme_sequence_splits/threshold_*` and `substrate_splits/threshold_*`.
Paths default to the EMULaToR layout. The work tree is `<EMULaToR>/outputs/baselines/CatPred` (or `CATPRED_BENCH_BASE_DIR`).

## Files

Removed: `tune_optuna.py`, `launch_parallel_optuna.py`, `retrain_from_optuna_best.py`, `export_optuna_best.py`, `run_split_benchmarks.py`, `precompute_features.py`, `kill_bench.py`, `bench_scheduler.py`, `bench_dataloader.py`, `progres.txt`, `progres.md`, `original_catpred_retrain_hparams.json`, `Plan.md`.

Changed: `common.py`, `align_structures.py`, `cache_embeddings.py`, `train_single_target_tvt.py`, `predict_single_target.py`, `bench_training_control.py`, `bench_fast_loader.py`, `launch_parallel_retrain.py`, `aggregate_catpred_results.py`, `fair_esm_shim.py`, `README.md`, `commands.txt`.

## Bugs fixed and deviations removed

- The old wrapper changed the original settings: batch size 32, cosine scheduler, early stopping, validation interval, smart batching, cached batch graphs. All are removed. Defaults are the original ones.
- The old precompute stage cached batches and fixed batch composition across epochs. It is removed.
- Best checkpoint now follows the original rule (validation rmse every epoch).
- Another user's home path and Optuna paths, keys and text are removed.
- The default split group `random_splits` is replaced by the five real groups.
- The aligner edited split files in place. It now writes a staged copy in the work tree.
- Ki structure lookup now uses the ESMFold2 directory and keys the fallback by `uniprot_id`.
- Checkpoint loading failed for a work tree outside the repo (untrusted path). The trainer now trusts its own `out_dir`.
- The ESM cache could be written outside `--base_dir`. The launcher cache now defaults to `<base_dir>/embeddings`.

## Speed-ups and why behaviour is unchanged

- Fused Adam: same update rule, faster kernel, but not bit-identical to `Adam`. It is off by default (`--optimizer_fused off`). Use `auto` or `on` only when exact equality is not needed.
- ESM fast forward: same ESM2 weights. The cache embeds one sequence per call, as the original does, so features match the original exactly.
- Attention uses the default `need_weights=True`, as the original does.
- Native per-member test predict runs, as in the original. It consumes torch RNG, so the later ensemble members match the original.
- Sequences truncated to 2048: the original limit.
- Inline data loader and ESM cache: only change where data comes from.
- Mixed precision is off by default. It is available only by flag.

## Unknown amino acids

The original raises `KeyError` on residues outside its 20-letter vocabulary (`catpred/models/model.py:374`).
The wrapper drops these rows from train, val and test, then counts them. It does not map them to pad.
`run_summary.csv` has `dropped_unknown_residue_train`, `_val` and `_test`. The filtered splits are in `<out_dir>/filtered_inputs/`.
Sequences with unknown residues in the 1d tables (unique sequences, all splits):

| Task | Unique sequences affected | Unique sequences | Letters |
|---|---|---|---|
| kcat | 53 | 20,593 | B, U, X |
| km | 147 | 29,085 | B, O, U, X, Z |
| ki | 29 | 5,938 | B, O, U, X |

The ESM and molgraph caches live in versioned folders (cache version 3). Their keys are the sequence and the input file signature. A filtered file has its own signature.
Check: a 62/16/15 smoke run with 2 train and 1 test rows changed to `X` dropped 2, 0 and 1 rows. The 64/16/16 comparison drops none and still matches the original bit for bit.

## Commands

See `commands.txt`. Full run, all seeds, splits and tasks:

```
/home/adhil/.conda/envs/mldb/bin/python emulator_bench/launch_parallel_retrain.py --gpus 0 1
/home/adhil/.conda/envs/mldb/bin/python emulator_bench/aggregate_catpred_results.py --seeds 666 777 888
```

## Changes after review

- `bench_training_control.py`: when `training_complete.json` and `model.pt` exist for an ensemble member, the epoch loop is skipped (`start_epoch = args.epochs`). Before, a resume logged "reusing" but trained the member again. Checked on a copy of the earlier smoke output: `--ensemble_size 2` after a finished 1-member run kept `model_0/model.pt` unchanged (same mtime), trained only `model_1`, and wrote the final results.
- `align_structures.py`: each staged split file now has a fingerprint under `<base_dir>/_structure_alignment_cache/staged_fingerprints/` (source size and mtime, `--limit_rows`, identity threshold, structure or plain). An existing staged file is skipped only if its fingerprint matches. A smoke staging with `--limit_rows` is therefore redone by a full run, and the reverse too. Checked on kcat uniprot_time_splits: limit 16, then full (55,029 train rows restaged), then a repeat (skipped, same mtime), then limit 16 (restaged to 16 rows).
- Not changed: `launch_parallel_retrain.py` still defines `_run_step` twice; unknown amino acids are dropped (see Unknown amino acids); the four patch modules stay.

## Smoke test

Env: mldb python, GPU 1, outputs under `scratchpad/bench_smoke/CatPred`.
fair-esm was installed into a scratch folder (not into mldb) and used through `CATPRED_FAIR_ESM_PATH`.
Run: kcat and ki, `random_splits_grouped_sequence`, seed 666, 64 rows, 1 epoch, ensemble 1.
Both runs finished with exit code 0 and wrote `final_results_test.csv`.
kcat test RMSE 1.616. ki test RMSE 1.908 (the E-GNN path ran; readout input 464 = 300 + 128 + 36).
The aggregator ran on the result. Nothing was written under `EMULaToR/data`.
Not run: km, other split groups, full-size training.

## Open issues

- The paper mentions bf16, but the original code used fp32. Default is fp32.
- The ensemble of 10 with 30 epochs for 3 seeds, all splits and tasks is very costly.
- proGRES reads only the first chain of a PDB file.
- The structure source order (PDBe, AlphaFold, ESMFold2) is the wrapper author's choice.
- The paper does not state batch size. Code value 16 is used.
- The paper text says 20 epochs for tuning and 30 for production. 30 is used.
- `catpred/data/esm_utils.py` imports `emulator_bench.fair_esm_shim`. This was kept.
- A finished ensemble is reused only when its `run_settings.json` matches the epochs, ensemble size and staged split files. The staging fingerprint includes the 3D candidate table signature. Runs without `run_settings.json` are retrained.
