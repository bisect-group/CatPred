# CatPred EMULaToR bench

This folder runs CatPred with its original published settings on the EMULaToR kinetic splits.
It does only multi-seed evaluation. Tasks: `kcat` and `km`. Seeds: 666, 777, 888.
The model, loss, features and hyperparameters are the original ones (`scripts/reproduce_fig5_catpred.sh`):
Seq-Attn plus ESM2, 30 epochs, batch size 16, ensemble of 10, MVE loss, Noam schedule with 2 warmup epochs, fp32,
best-validation-RMSE checkpoint for each member, sequence length 2048.
One addition, by user decision: ProGRES structure vectors (128-d) join the ESM2 features through the original
`--add_pretrained_egnn_feats` path, for kcat and Km. The released script uses them only for Ki.
The production run also uses `--amp` and `--skip_pad` (see Speed).

## Splits and runs

`launch_parallel_retrain.py` finds the split folders on disk under `data/processed/datasets/kinetic_dataset/<task>`:
`random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, and the `threshold_*` folders of
`enzyme_sequence_splits`, `enzyme_structure_splits`, `substrate_splits` and `conformer_cosine_splits`.
That is 7 folders per task, 14 in all, and 42 runs with 3 seeds. The dataset tree is read only.

## Steps

1. `cache_embeddings.py` builds the ESM2 cache once (all unique kcat and km sequences). It takes about 35 minutes with 2 processes on one GPU, and 68 GB of disk.
2. `cache_structures.py` picks one structure per sequence, embeds it with progres 0.2.7 and writes the dict `catpred_progres_embeds_dict.pt` (about 3 minutes, 25 MB).
3. `launch_parallel_retrain.py --egnn_path <dict>` starts one `train_run.py` process per run.
4. `train_run.py` trains the 10 members of one run, writes the predictions and the metrics.
5. `aggregate_catpred_results.py` rebuilds the result tables from the saved predictions.

## Structures

One structure per unique sequence, in this order (same rule as the DEKP baseline):
1. Experimental: the dataset row's `pdb_chain` in the mmCIF file (`mmcif/<ID>_updated.cif`), best `pdb_coverage` among the rows of the sequence in `<task>_kinetic_params_3d.parquet`. The chain is written to `structures/chains/<ID>_<chain>.pdb`.
2. AlphaFold: `alphafold/AF-<accession>-F1-model_v<n>.pdb`, newest version.
3. ESMFold2: `esmfold2_pdb/ESMFold2-<id>.pdb`.
4. ESM3: `esm/ESM3-open-small-<id>.pdb`, only if the file exists.

`<id>` is the UniProt id with mutations (`P12345|A1B`) or the accession. The ESM3 files (955) live only on the farm in `data/intermediate/esm`;
here they were copied to `outputs/baselines/CatPred/structures/esm`. Pass `--esm3_dir` if the folder is elsewhere.
A sequence without a structure is not in the dict. The original model then gives it the mean vector (`pretrained_egnn_feats_avg`), and the row stays.
The dict key is the sequence key (sha1) that the ESM cache also uses. The progres vectors are cached by sequence key and file fingerprint, so the build resumes and a changed structure file is embedded again.
Coverage per task is in `structure_coverage.csv` and the chosen file for each sequence in `structure_table.csv` (both in `embeddings/progres_embeds`).

## Files

- `fast_train.py`: the fast trainer. It uses the original model, loss, scheduler and data classes. See `REPORT.md` for what differs.
- `esm_cache.py`: packed cache of per-residue ESM2 vectors.
- `structures.py`: the structure rule and the mmCIF chain extraction. `cache_structures.py`: progres vectors and the dict. `progres_shim.py`: unpacks the pinned progres 0.2.7 wheel and weights.
- `reference_run.py`: runs the original `run_training` for the tests and the benchmarks.
- `bench_speed.py`: step rate of the original loop and of the fast trainer.
- `verify_equivalence.sh`: tests against the original code (`../tests`). `--gpu` adds the GPU tests.

## Work tree

Default `<EMULaToR>/outputs/baselines/CatPred` (`--out_root` or `CATPRED_BENCH_OUT_ROOT`).

- `embeddings/esm2_t33_650M_UR50D_v4`: ESM2 cache. The folder name holds the cache version (`CACHE_VERSION` in `common.py`).
- `embeddings/fair_esm`: fair-esm 2.0.0, unpacked from the pinned wheel. It is never installed into the environment.
- `embeddings/progres`: progres 0.2.7 (wheel sha256 checked) and its trained weights (sha256 checked), used through `sys.path`. Never installed.
- `embeddings/progres_embeds`: `catpred_progres_embeds_dict.pt`, `vectors.jsonl` (resume cache), `structure_table.csv`, `structure_coverage.csv`, `manifest.json`.
- `structures/chains`, `structures/esm`: extracted experimental chains and the copied ESM3 files.
- `<task>/<split_group>/<split_name>/seed_<s>/`: one run. `split_name` is `threshold_<t>`, or `all` for a folder without thresholds.
  - `model_<k>/model.pt`: best-validation checkpoint of member k (original format). `model_<k>/history.csv`: loss sum and validation RMSE per epoch.
  - `pred_label_{train,val,test}.csv`, `final_results_{train,val,test}.csv`: ensemble predictions and metrics (log10 scale).
  - `run_summary.csv`: metrics, best epochs, and the counts of dropped rows. `run_settings.json`: the settings of the run.
  - `members/`: raw member results. `inputs/`: filtered split tables.
- `<task>/<split_group>/<split_name>/seed_<s>.log`: log of the run.
- `aggregate_catpred_results.csv`, `catpred_seed_results.csv`, `catpred_missing_results.csv`: summaries.

A run with the same settings is skipped. A run stopped halfway resumes at the next member. Changed settings start a new run. The old run folder moves to `<run_dir>.old-<old settings id>-<time>` and is never deleted. With `--egnn_path`, the run fails if fewer than 98% of the rows of any split have a structure vector (`--min_hit_rate`). The rates are in `run_summary.csv`. Split files are compared by content hash. The launcher checks that the ESM cache holds every sequence.

## Dropped rows

Rows are dropped and counted in `run_summary.csv`. `dropped_unknown_residue_*` counts rows with a residue outside the 20-letter vocabulary
(the original raises `KeyError`). `dropped_invalid_smiles_*` counts rows with a SMILES that has no heavy atom or does not parse
(the original `get_data` drops them silently).

## Speed

The default path is exact: the maths of the original code, same batches, same inits.
`--workers N` trains members in N processes on one GPU and does not change the result.
`--tf32`, `--amp`, `--fused_adam` and `--skip_pad` change the numbers. They are off by default in the code. See `REPORT.md` for their speed and effect.
`--skip_pad` keeps the maths (equal in float64 to 1e-15) and skips the ESM block of the first pooling layer at padded positions.
`--fused_attn` runs the attention core of the Seq-Attn block as one Triton kernel (GPU only, off by default). With the default `--attn_precision ieee` it keeps the maths of `nn.MultiheadAttention` (fp32 dot products, no TF32; max error 4.6e-6 against float64 for L up to 2047, the float32 original gives 2.2e-6) and the peak GPU memory of one process falls from 6.9 to 1.1 GB. `--attn_precision tf32` or `bf16` change the numbers (see `REPORT.md` section 13). `--compile` compiles the attentive pooling with `torch.compile(dynamic=True)` (no gain measured). Without a CUDA device these flags stop with an error.
The production command uses `--amp --skip_pad --egnn_path`. One process needs 8.5 GB of GPU memory, so a 24 GB A10 holds 2 member processes (`--workers 2`).

See `commands.txt` for the commands.
