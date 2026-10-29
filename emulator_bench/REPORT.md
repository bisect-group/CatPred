# CatPred emulator_bench report
Scope: kcat and km, 7 split folders each, seeds 666 777 888 (42 runs). Ki is out of scope.

## 1. Method summary
CatPred (Boorla and Maranas, Nat. Commun. 2025) predicts kcat and Km from an enzyme sequence and a substrate SMILES. A D-MPNN encodes the substrate. The enzyme side has a Seq-Attn block (embedding, rotary, 6-head attention) and per-residue ESM2 vectors (1280-d). Both go through an attentive pooling layer. The pooled vector and the substrate vector feed an FFN that outputs a mean and a variance. The loss is MVE on train-standardised log10 targets. The model is an ensemble of 10. Each member keeps the epoch with the best validation RMSE. The ensemble prediction is the average of the member means. The released kcat and Km models use Seq-Attn plus ESM2. The wrapper only runs multi-seed evaluation with these settings. It does no tuning.

One addition, by user decision: a 128-d ProGRES structure vector joins the pooled enzyme vector (original `--add_pretrained_egnn_feats`, `model.py:122-128, 400-408, 443`) for kcat and Km, next to the ESM2 features. The released script uses these vectors only for Ki. Production also uses bf16 autocast (`--amp`) and a padded-position skip in the first pooling layer (`--skip_pad`, section 6).

## 2. Original settings
| Setting | Value | Source |
|---|---|---|
| Epochs | 30 | `scripts/reproduce_fig5_catpred.sh:68`, `args.py:508` |
| Batch size | 16 | `reproduce_fig5_catpred.sh:67` (not in paper) |
| Ensemble | 10 | `reproduce_fig5_catpred.sh:67` |
| Loss | mve | `reproduce_fig5_catpred.sh:67`, `loss_functions.py:230` |
| Enzyme features | Seq-Attn + ESM2, `seq_embed_dim` 36, 6 heads | `reproduce_fig5_catpred.sh:54,67` |
| Optimizer | Adam, weight decay 0 | `utils.py:534` |
| LR | Noam, 1e-4 / 1e-3 / 1e-4, 2 warmup epochs, `train_data_size // batch_size` steps per epoch | `args.py:510-519`, `utils.py:547` |
| Dropout | 0 | `args.py:363` |
| Model selection | best validation RMSE, first minimum | `run_training.py:331` |
| Target transform | train targets standardised | `run_training.py:182` |
| Precision | fp32 (paper mentions bf16; code does not) | code |
| ESM2 | `esm2_t33_650M_UR50D`, layer 33, per residue, no cls/eos, fp32, one sequence per call, 2048 tokens | `esm_utils.py:72,153,159,162` |
| Structure features | Released script: Ki only (`reproduce_fig5_catpred.sh:52`). Ablation: ESM2 + ProGRES (`reproduce_ablation.sh:56`). This wrapper: kcat and Km (user decision). 128-d, key = protein record `name`, missing key gives the mean vector | `model.py:122-128, 226-227, 400-408` |
| Vocabulary | 20 letters; others raise `KeyError` | `model.py:370-374` |
| Seeds | 666 777 888; `--seed` and `--pytorch_seed` both set (the released script sets neither) | `args.py:302,307` |

## 3. What the wrapper does
1. `cache_embeddings.py` embeds every unique kcat and km sequence once, using the original `get_single_esm_repr`. Output: `embeddings/esm2_t33_650M_UR50D_v4` (30,120 sequences, 68 GB).
2. `cache_structures.py` picks one structure per unique sequence, embeds it with progres 0.2.7 and writes `catpred_progres_embeds_dict.pt` (section 12).
3. `launch_parallel_retrain.py` finds the split folders on disk (7 per task) and starts one `train_run.py` per task, seed and folder.
4. `train_run.py` reads the parquet splits, drops rows (section 6), builds datapoints with the original `get_data`, scales train targets with the original `StandardScaler`, and rebuilds the member inits and batch order of the original run. Then it starts `--workers` member processes that train with `fast_train.py`.
5. The run averages member predictions in order and writes metrics and predictions. `aggregate_catpred_results.py` rebuilds the tables from the saved predictions.

No split staging is used. Structure features are on only with `--egnn_path`.

## 4. Files
- Added: `fused_attention.py`, `profile_step.py`, `bench_attention.py`, `fast_train.py`, `train_run.py`, `esm_cache.py`, `reference_run.py`, `bench_speed.py`, `verify_equivalence.sh`, `structures.py`, `cache_structures.py`, `progres_shim.py`, `../tests/`.
- Rewritten: `common.py`, `cache_embeddings.py`, `fair_esm_shim.py`, `launch_parallel_retrain.py`, `aggregate_catpred_results.py`, `README.md`, `commands.txt`.
- Removed: `align_structures.py`, `build_tvt_data.py`, `train_single_target_tvt.py`, `predict_single_target.py`, `bench_amp.py`, `bench_fast_loader.py`, `bench_feature_cache.py`, `bench_optimizer.py`, `bench_speedups.py`, `bench_training_control.py`.
- No original file was edited.

## 5. Bugs fixed
- Staged splits and the structure stage are removed. kcat and km do not need them.
- All 7 split groups are found (the old wrapper found 5).
- Resume reuses a member only when the settings match.
- The aggregator reads the dataset tree.
- Invalid-SMILES drops are now counted.

## 6. Deviations, and why behaviour is unchanged
Dropped rows are counted per split in `run_summary.csv`.
- Unknown residue: the original raises `KeyError`. 53 kcat and 147 km unique sequences. Open question for the user.
- Invalid SMILES: the original `get_data` drops them silently (`data/utils.py:311`). All are `[H]` or `[H].[H]`.
- No missing targets.

| Task | Split folder | Rows tr/val/te | Unknown residue tr/val/te | Invalid SMILES tr/val/te |
|---|---|---|---|---|
| kcat | random_splits_grouped_sequence | 50,289 / 2,794 / 2,795 | 206 / 5 / 4 | 728 / 40 / 39 |
| kcat | random_splits_grouped_smiles | 50,301 / 2,788 / 2,789 | 184 / 12 / 19 | 807 / 0 / 0 |
| kcat | uniprot_time_splits | 50,526 / 2,800 / 2,552 | 201 / 4 / 10 | 755 / 24 / 28 |
| kcat | enzyme_sequence_splits/threshold_0.01 | 50,293 / 2,793 / 2,792 | 169 / 36 / 10 | 743 / 31 / 33 |
| kcat | enzyme_structure_splits/threshold_0.625 | 50,308 / 2,793 / 2,777 | 213 / 2 / 0 | 758 / 32 / 17 |
| kcat | substrate_splits/threshold_0.505 | 50,307 / 2,793 / 2,778 | 191 / 7 / 17 | 807 / 0 / 0 |
| kcat | conformer_cosine_splits/threshold_0.996 | 50,292 / 2,793 / 2,793 | 183 / 7 / 25 | 807 / 0 / 0 |
| km | random_splits_grouped_sequence | 70,954 / 3,942 / 3,935 | 363 / 22 / 17 | 7 / 1 / 0 |
| km | random_splits_grouped_smiles | 70,922 / 3,926 / 3,983 | 371 / 13 / 18 | 7 / 0 / 1 |
| km | uniprot_time_splits | 71,158 / 3,980 / 3,693 | 381 / 12 / 9 | 8 / 0 / 0 |
| km | enzyme_sequence_splits/threshold_0.05 | 70,954 / 3,941 / 3,936 | 267 / 26 / 109 | 2 / 3 / 3 |
| km | enzyme_structure_splits/threshold_0.69 | 70,963 / 3,941 / 3,927 | 391 / 10 / 1 | 3 / 3 / 2 |
| km | substrate_splits/threshold_0.52 | 70,949 / 3,941 / 3,941 | 358 / 17 / 27 | 8 / 0 / 0 |
| km | conformer_cosine_splits/threshold_0.997 | 70,949 / 3,941 / 3,941 | 350 / 15 / 37 | 8 / 0 / 0 |

The original has no padding mask, so the output depends on the longest sequence in the batch. The wrapper keeps the original batches and padding.

Exact speed-ups (default):
- ESM vectors come from memory-mapped fp32 files, copied by 2 background threads on a side stream.
- Each unique SMILES gets its `MolGraph` once, and batches are concatenated arrays (bit-equal to `BatchMolGraph`).
- The lean forward builds the padded tensors with 2 copies and one `index_copy_`.
- There is no per-step `.item()` or logging, and the all-ones masks and weights are left out.
- Members run in separate processes. Inits and batch order are rebuilt exactly. The sampler is one `Random(seed)` shared by all members, and the torch RNG replay (loader draws plus the second model build) is in `member_inits`.

Not possible with equal maths: CUDA graphs and `torch.compile` (shapes change every step), and ESM on the GPU (68 GB).

Options that change numbers (off by default):
| Option | Flag | Steps/s | Speed-up | Max loss diff (200 steps) | Max weight diff (200 steps) |
|---|---|---|---|---|---|
| exact | none | 29.5 | 1.00 | noise floor 4.1e-3 | 3.9e-3 |
| TF32 | `--tf32` | 39.8 | 1.35 | 2.4e-2 | 1.0e-2 |
| bf16 | `--amp` | 38.8 | 1.32 | 2.0e-2 | 1.0e-2 |
| fused Adam | `--fused_adam` | 29.8 | 1.01 | 5.0e-3 | 3.9e-3 |
| all three | | 39.0 | 1.32 | 3.0e-2 | 1.1e-2 |

The noise floor is two exact GPU runs with equal inputs (atomic adds in the backward of `index_select`). The effect of the options on the final metric is not measured. Fused SDPA attention was slower and was removed. (This table was measured on a shared GPU without structure features. See the next block for the production options.)

Padded-position skip (`--skip_pad`, user decision, off by default, on in the production command):
- The original attentive pooling has no padding mask. Its input at a padded position is `[0 (ESM, 1280), attn_out (36)]`, so the ESM block of `linear1` adds zero there. The softmax still covers every position, as in the original. `--skip_pad` applies the ESM block of `linear1` only to the real residues (`index_add`), and replaces `cat` plus `bmm` by two `bmm`. The maths is equal. Only the order of float operations changes.
- Test: model and batch in float64 on the CPU, exact path against skip path, with and without structure features, batches of 16, 5 and 1 (padded share 0.38, 0.27, 0). Output and gradient differences are 0 to 7.8e-16 (gradients up to 2.2), the machine epsilon. The test fails above 1e-12.
- In float32 the difference is rounding only (`allclose` rtol 1e-4 on output, 1e-3 on gradients).

Production options, real kcat `random_splits_grouped_sequence` train split, batch 16, structure features on, one process on a free RTX PRO 4500 Blackwell (32 GB), 40 s per row, prefetch 2 threads:
| Variant | Flags | Steps/s | Speed-up | Peak allocated / reserved (GB) |
|---|---|---|---|---|
| exact | none | 62.3 | 1.00 | 6.97 / 7.65 |
| bf16 | `--amp` | 81.2 | 1.30 | 6.95 / 8.53 |
| skip | `--skip_pad` | 79.4 | 1.27 | 6.97 / 9.29 |
| bf16 + skip | `--amp --skip_pad` | 90.7 | 1.46 | 6.95 / 8.41 |

Drift after 200 steps from the same init, against the exact-against-exact noise floor of the same call (several calls, because the floor changes from call to call):
| Option | Max loss diff | Max weight diff | Noise floor, loss | Noise floor, weights |
|---|---|---|---|---|
| `--amp` | 3.6e-2 (3 calls, same) | 9.8e-3 | 2.4e-7 to 4.8e-3 | 1.2e-4 to 3.1e-3 |
| `--skip_pad` | 1.6e-3 to 2.1e-2 | 3.2e-3 to 7.5e-3 | 2.4e-7 to 2.1e-2 | 1.3e-4 to 5.7e-3 |
| `--amp --skip_pad` | 1.8e-2 to 2.6e-2 | 9.0e-3 to 1.0e-2 | 2.5e-4 to 1.9e-2 | 1.5e-3 to 7.5e-3 |

`--skip_pad` alone stays inside the noise range of all calls (up to 2.1e-2 in loss and 7.5e-3 in weights). `--amp` and `--amp --skip_pad` drift about 1e-2 in weights, as `--amp` did before (1e-2). Both are above the usual noise, so they are deviations. Their effect on the final metric is not measured.

## 7. Commands
See `commands.txt`.

## 8. Smoke test
The launcher on GPU 0 with 64 rows, 1 epoch and 2 members finished for kcat and km in 6 s each (exit 0). The aggregator ran on the outputs. Dry run lists 42 runs. Real-size timing, kcat `random_splits_grouped_sequence`, seed 666:
- 1 member, 3 epochs, 1 process: 416.7 s (val RMSE 1.257, 1.179, 1.122; test RMSE 1.248, PCC 0.632).
- 2 members, 1 epoch, 2 processes: 253 s wall, 240 s per member (test RMSE 1.345).

Measured with the MPEK job on the GPU.

Production flags (`--amp --skip_pad --egnn_path`), 2 workers, 64 rows, 1 epoch, 2 members, kcat and km `random_splits_grouped_sequence`: exit 0 in 9 s each. `run_settings.json` holds the options and the sha256 of the dict. Dry run with these flags: 42 runs.

## 9. Equivalence verdict: EQUIVALENT
Exact path (no `--amp`, no `--skip_pad`), with and without structure features.
CPU (tiny fixture, original `run_training` as reference, deterministic algorithms): 36 tests pass, 0 skipped (`verify_equivalence.sh`, 170 s).
New in this run: with a ProGRES dict that misses a third of the sequences (they get the mean vector), forward, every gradient, per-step loss and weights (36 steps, 3 members) and member test predictions are bit-equal to the original with `--add_pretrained_egnn_feats`; the skip path equals the exact path in float64 (section 6); the run settings (amp, skip, dict hash) start a new run when they change (the old folder moves aside, nothing is deleted); 4 structure resolver tests; 1 progres test (vectors equal `progres.embed_structure`, resume, changed file re-embedded).
Before this run the 21 tests covered the following. Bit-equal results for ESM cache against the original call, 2100-residue truncation to 2047 rows, `BatchMolGraph` tensors, forward and every gradient, init weights of 3 members, batch order of 9 epochs, per-step loss and weights (36 steps), validation RMSE and best epoch, member and ensemble test predictions, and 1 against 3 processes. Resume and settings-change restart also pass.

GPU (`verify_equivalence.sh --gpu`, 7 tests pass, real kcat split, real ESM cache and real ProGRES dict, free GPU):
- ESM vectors equal the original call on GPU, and the forward on 5 real batches (with structure features) is bit-equal.
- After 20 steps the weight difference is 4.1e-5 for original against original, and 7.8e-5 for original against fast, with a loss difference of 1.9e-6 in both pairs.
- `--skip_pad` and `--amp --skip_pad` forwards stay close to the exact forward on 5 real batches (rtol 1e-3 and 0.1).
- Side-stream batches equal the CPU batches, and 2 processes match 1 within 1e-3.

ESM reuse: see "ESM2 reuse" above. Nothing reused. All 30,120 vectors computed with the original path.

## 10. Speed and projection
Measured on cuda:0 while MPEK used the GPU at 100%. Real kcat `random_splits_grouped_sequence` train split, batch 16.
| Variant | Steps/s | Note |
|---|---|---|
| Original loop | 9.1, 8.7 | 6.6 GB peak |
| Fast, 1 process | 29.5 | 3.2×; 7.0 GB peak |
| 1, 2 or 4 prefetch threads | 29.5 each | host not the limit |
| 2 processes (aggregate) | 49 | 29.5 + 19.5 |
| 3 processes (aggregate) | 54 | 23 GB total |

Profile: attention 50% (bmm 23%, softmax 16%, element-wise 9%), first pooling layer and substrate FFN 35%. The GPU is busy the whole wall time.

Work: 48.7M step-equivalents (kcat 20.0M, km 28.7M). One kcat run is about 0.97M steps, one km run about 1.36M.
| Layout | Steps/s | Days |
|---|---|---|
| Original loop | 9.1 | 62 |
| Fast, 1 proc | 29.5 | 19.1 |
| Fast, 2 proc | 49 | 11.5 |
| Fast, 3 proc | 54 | 10.4 (kcat 4.3, km 6.1) |
| TF32 + bf16, 1 proc | 39 | 14.4 |

Days = 48.7e6 / (steps_per_s × 86400). These are upper bounds for this machine. The authors' A100 figures (about 5 h per seed kcat, 15 h km) imply about 17 days on one such GPU.

Production layout. Measured on a free RTX PRO 4500 Blackwell (32 GB), structure features on, `--amp --skip_pad`, same split as above. This is not an A10.
| Processes on the GPU | Steps/s per process | Aggregate | GPU memory (nvidia-smi) |
|---|---|---|---|
| 1 | 91.2 | 91.2 | 8.5 GB |
| 2 | 42.8, 43.0 | 85.8 | 17.0 GB |
| 3 | 32.1, 28.2, 42.2 | 102.5 | 25.6 GB |
| 4 | out of memory (32 GB full) | | |

- One process: 6.95 GB peak allocated, 8.4 GB reserved, 8.5 GB in nvidia-smi (with the CUDA context). `--amp` does not lower the peak (6.95 against 6.97 GB for exact). The padded ESM tensor and the activations of the longest batch set the peak, and the fp32 weights stay.
- The farm A10 has 23,028 MiB (about 22.5 GB usable). 2 processes (17.0 GB) fit. 3 processes (25.6 GB) do not. So the production command uses 2 member processes per GPU, 1 run per GPU, `--gpus 0 1`.
- On this GPU one process already fills the compute (2 processes give no gain). An A10 is slower, so it is also likely compute bound, and `--workers 1` may be as fast. This is not measured. Measure on the A10 first (commands.txt item 7, then 2 concurrent calls).
- Projection with the exact numbers of this GPU: 48.7e6 / 91.2 / 86400 = 6.2 days on one such GPU, 3.1 days on two. For 2 A10s there is no measurement.

## 11. Open issues
- Wall clock on 2 A10 GPUs is not measured (section 10). Measure the step rate on the farm before the launch.
- `--skip_pad` (user decision) gives 1.27× alone and 1.46× with `--amp`. It keeps the maths and changes rounding. `--amp` drifts the weights by about 1e-2 in 200 steps, above the usual noise.
- The ProGRES vectors are from progres 0.2.7 and its Zenodo model (section 12). The authors' dict was not available, so equality with it is not checked.
- Kcat and Km with structure features is not in the released script (Ki only). It follows the ablation script. This is a user decision.
- GPU runs are not bit-reproducible. `--deterministic` exists but was only used on CPU.
- Dropped rows: unknown residues (user decision) and `[H]` substrates (original behaviour).
- Seeds: the released script uses seed 0. The wrapper sets both seeds.
- The paper says 20 epochs for tuning and 30 for production; 30 is used. Batch size 16 is from the code. The paper mentions bf16; the code uses fp32.
- Each member process needs 8.5 GB of GPU memory with `--amp --skip_pad` and structure features (section 10).
- The ESM cache is 68 GB. Run kcat first.
- `tensorboardX` is missing in mldb; reference runs use torch's `SummaryWriter`.
- Checkpoints load in the original `predict.py` only with `CATPRED_TRUSTED_DESERIALIZATION_ROOTS` set.
- Not run: a full 30-epoch member, a real-size km run, multi-GPU, an A10, and TF32 or bf16 in a full training.

## 12. Structures and ProGRES vectors
Source of the original vectors (user decision C):
- `catpred_progres_embeds_dict.pt` is not built by any script in this repo or its docs (searched `scripts/`, `README.md`, `environment.yml`, `modal_app.py`, `catpred/`). The scripts only read `$DATA_DIR/catpred_progres_embeds_dict.pt`.
- The model code gives the format: a dict from the protein record `name` to a 128-d tensor (`model.py:227` adds 128 to the FFN input; `model.py:400-408` looks up `each['name']`). `create_pdbrecords.py` sets `name` to the basename of the row's `pdbpath`. A name that is not in the dict gets the mean of all vectors.
- 128-d is the ProGRES embedding size (`embedding_size = 128` in progres). The only version pin is `progres==0.2.7` in `modal_app.py` (`environment.yml` has `progres` without a version). The README credits progres (pre-trained EGNN models) for these features. The exact model of the authors' dict is not documented. This wrapper uses progres 0.2.7 with its `trained_model.pt`.
- mldb has progres 1.1.0. That version is not used. `progres_shim.py` unpacks the wheel `progres-0.2.7-py3-none-any.whl` (sha256 5037ce4b...fd7cfe2, checked) and `trained_model.pt` (Zenodo record 13365312, sha256 3fa3de9a...6be3, md5 equal to the Zenodo md5) into `embeddings/progres`, and imports them through `sys.path`, like fair-esm. Dependencies (torch, torch_scatter, torch_geometric, einops, Biopython, pydantic) are already in mldb. Nothing was installed.
- Call: `progres.embed_structure(path, model=model)`: C-alpha atoms of the first chain, contacts under 10 A, 6 EGNN layers, unit-length 128-d output.
- Here the dict key is the sequence key (sha1), not a file name, because one structure is chosen per sequence. The model only needs the same key in the dict and in the record `name`.

Structure choice (user decision B), per unique sequence of the 14 split folders (vocabulary filter applied), over the rows of the sequence in `<task>_kinetic_params_3d.parquet`: best-coverage experimental chain (`pdb_chain`, mmCIF), else AlphaFold, else ESMFold2, else ESM3.

| Task | Unique sequences | Experimental | AlphaFold | ESMFold2 | ESM3 | None |
|---|---|---|---|---|---|---|
| kcat | 20,540 | 13,358 | 6,990 | 148 | 34 | 10 |
| km | 28,938 | 16,135 | 12,449 | 301 | 45 | 8 |
| union (dict) | 30,120 | 16,943 | 12,804 | 311 | 51 | 11 |

The dict holds 30,109 vectors (25 MB). The 11 sequences with no structure get the mean vector. Rows are kept. No structure failed in progres. The "unknown residue" sequences (53 kcat, 147 km) are already dropped before this step.
- Files: local `data/intermediate/{mmcif,alphafold,esmfold2_pdb}` hold every file needed (5,748 mmCIF, 9,771 AlphaFold, 311 ESMFold2). `processed_exp_pdb` is not used. The 955 ESM3 files exist only on the farm (`data/intermediate/esm`, 913 MB). They were copied read-only to `outputs/baselines/CatPred/structures/esm`. Experimental chains are in `structures/chains` (1.2 GB).
- The farm has far more structure files than this machine (for example 149,253 mmCIF files against 5,785). The resolver gave the same counts on the farm file lists (read-only `ls`) and on the local folders, and every chosen file exists locally.
- Vectors were checked against `progres.embed_structure(path)` called again, for 4 structures of each source, on the CPU and on the GPU: max abs difference 3e-8 to 7e-7 (float32 rounding; the ESM3 files are the longest). Bit-equal when the model and the device are the same (the unit test, on the CPU).
- The vector cache (`vectors.jsonl`) is keyed by sequence key and sha1 of the structure file. The build takes 157 s on the GPU for 30,120 structures and resumes.
- The experimental chain that the dataset names can differ from the sequence (mutants, tags). The AlphaFold and ESM structures of a mutant are those of the wild type (same rule as DEKP).

## 13. Fused attention kernel (`--fused_attn`)
Measured on the farm A10 (GPU 0, shared with another job at 90-100% use, so rates move by up to 1.5x between calls). Real kcat `random_splits_grouped_sequence` train split, batch 16, structure features, `--skip_pad`, fp32. Variants ran one after the other, in rounds. Read the ratios, not the absolute rates.

Profile of one step (torch.profiler, 40 steps, share of GPU kernel time; backward kernels count for their forward part):
| Part | `--skip_pad` | `--skip_pad --fused_attn` |
|---|---|---|
| Attention (in-projection, scores, softmax, value product, out-projection; forward and backward) | 66-67% | 38% |
| Attentive pooling (first layer on the 1316-d input, softmax, bmm) | 27% | 50% |
| D-MPNN encoder | 3.1% | 6.0% |
| Optimizer | 1.2-1.4% | 2.3% |
| Seq embedding, rotary, FFN (readout), loss, copies | 2.5% | 3.7% |
| Kernel time per step | 36 to 60 ms | 18 ms |

Attention is the largest part, so the kernel is worth building.

Design (`fused_attention.py`):
- Only the attention core is a kernel. The in-projection (weights and biases of `in_proj_weight` and `in_proj_bias`), the scale `sqrt(1/6)` on q (as `F.multi_head_attention_forward`) and the out-projection stay matmuls. The softmax covers every position, with no mask, as in the original. The unused attention weights of `need_weights=True` are not computed.
- Forward: FlashAttention style, one program per block of 64 queries and one (sample, head), online softmax over blocks of 64 keys, row log-sum-exp saved. No L x L matrix. Backward: probabilities are recomputed. One kernel gives dK and dV, one gives dQ (no atomic adds, so it is deterministic). The preprocessing `sum(dO * O)` is one torch op.
- The head dimension 6 is zero-padded to 16, not 8, because `tl.dot` needs at least 16. Zero columns change no dot product, so this is exact. fp32, `input_precision="ieee"` (no TF32).
- `--attn_precision tf32` uses `input_precision="tf32"`. `bf16` casts the tiles of every dot to bf16 with fp32 accumulation. Softmax, running max and sum, exp, outputs and gradients stay fp32.
- q and k are equal tensors in the original, so the fused path calls rotary once.
- The flag changes only `forward` in `fast_train.py`. The CPU path is unchanged. With `--fused_attn` and no CUDA device the code raises `RuntimeError` (`train_run.py` exits with a message for `--cpu`). Under `--amp` the kernel gets fp32 inputs (`custom_fwd(cast_inputs=float32)`).
- Run settings: `fused_attn` and `attn_precision` (and `compile`) enter `run_settings.json` only when used. Runs without the flags keep their settings id, so no finished run moves aside.

Equivalence tests (`tests/test_fused_attention.py`, random weights and biases, q, k and v differ, batch 4, forward and gradients of q, k, v and all 4 parameter tensors; error is max difference over the largest reference value; reference is the original module in float64 on the GPU):
| L | ieee, fused | ieee, float32 original | tf32 | bf16 |
|---|---|---|---|---|
| 1 (q and k gradients skipped, zero in exact maths) | 7.9e-7 | 1.5e-7 | 4.0e-3 | 2.3e-2 |
| 2 | 1.1e-6 | 7.1e-7 | 5.7e-3 | 2.1e-2 |
| 7, 33, 64, 129 | 1.1e-6 to 1.8e-6 | 1.0e-6 to 1.7e-6 | 2.9e-3 to 3.9e-3 | 0.97e-2 to 2.7e-2 |
| 500 | 1.8e-6 | 1.9e-6 | 4.5e-3 | 3.3e-2 |
| 2047 | 4.6e-6 | 2.2e-6 | 4.0e-3 | 2.4e-2 |
The ieee kernel gives the same result in two calls (bitwise). Test bounds: ieee 2e-5 and at most 10 times the float32 original; tf32 1e-2; bf16 5e-2. At L=1 the kernel leaves rounding noise (about 4e-7 of the largest gradient for ieee) in the gradient into q and k, where the exact value is zero. On real batches the full forward with the kernel stays within 1e-4 of the `--skip_pad` forward (tf32 2e-2, bf16 1e-1), also with `--compile`.

Kernel alone (`bench_attention.py`, forward plus backward of the whole attention block with projections, batch 16, median of 7 rounds, same GPU, shared):
| L | original (ms) | ieee | tf32 | bf16 |
|---|---|---|---|---|
| 300 | 4.5 | 0.89x | 0.89x | 0.87x |
| 800 | 24.5 | 2.8x | 5.0x | 4.5x |
| 2047 | 145 | 3.6x | 11.0x | 13.8x |
The original keeps the L x L weights: the peak of this benchmark at L=2047 was 6.5 GB. The kernel keeps none (full-step peak above). A short batch is slower than the original because the padded tensors and the launches cost more than the saved time.

Full step, 1 process (steps/s, rounds in order; variants alternate in each round):
| Round | `--skip_pad` | + `--fused_attn` | + tf32 | + bf16 |
|---|---|---|---|---|
| 1 | 29.9 | 30.5 | 33.9 | 32.3 |
| 2 | 18.9 | 31.4 | 31.3 | 31.9 |
| 3 | 20.1 | 31.1 | 33.4 | 37.5 |
Further paired calls (two more series): unfused 18.3, 29.8, 30.8, 20.6, 31.2 against fused 30.8, 26.8, 34.9, 27.9, 34.4. Median of all 8 calls: 25.2 against 31.0 (ratio 1.23). The ratio of a pair goes from 0.9 to 1.65, because the other job on the GPU slows the unfused step more. When the GPU is quiet, unfused and fused both reach about 30 steps/s. So the step then waits on something other than attention (probably the Python launch work and the host: the D-MPNN, the many small kernels). tf32 and bf16 add 4 to 7% over ieee in the full step, the kernel is not the limit any more. Honest summary: 1.0x to 1.6x, about 1.2x typical on this shared GPU.

Memory (peak allocated, one process): 6.9 GB without the kernel, 1.1 GB with it (reserved 9.2 against 1.6 GB). With `--compile` 0.83 GB.

Several processes on GPU 0, fused ieee (steps/s per process, aggregate): 2 processes 21.1 + 21.1 and 30.3 + 21.4 (aggregate 42 to 51); 4 processes 15 to 17.5 each (aggregate 59 to 63; 4 x 1.6 GB reserved). One process gives 27 to 35. The unfused 2-process case was not run: 2 x 9.2 GB reserved plus the 3.7 GB of the other job leave no margin on the 22.5 GB. So the kernel lets 4 or more processes share one A10 where only 2 fit before. Whether this raises the aggregate on a quiet GPU is not measured (the GPU was never quiet).

Drift after 200 steps from the same init (as `bench_speed.py --mode drift`: exact path against option path, exact against exact is the noise floor; two calls each; all with `--skip_pad`):
| Option | Max loss diff | Max weight diff | Noise floor, loss | Noise floor, weights |
|---|---|---|---|---|
| `--skip_pad` only | 2.4e-2, 3.7e-2 | 5.6e-3, 5.8e-3 | 3.6e-7, 3.2e-2 | 3.1e-4, 5.7e-3 |
| + `--fused_attn` (ieee) | 9.5e-3, 8.7e-3 | 7.6e-3, 4.1e-3 | 2.4e-7, 4.8e-7 | 1.1e-3, 1.8e-3 |
| + tf32 | 1.3e-2, 2.4e-2 | 9.5e-3, 8.9e-3 | 1.4e-2, 8.2e-3 | 5.6e-3, 3.8e-3 |
| + bf16 | 2.2e-2, 2.2e-2 | 7.6e-3, 7.6e-3 | 3.6e-7, 1.4e-2 | 7.5e-4, 5.6e-3 |
The noise floor changes from call to call, by up to 5 orders in loss (the atomic adds of the original backward, then training amplifies them). The ieee kernel with `--skip_pad` drifts as much as `--skip_pad` alone (weights 4e-3 to 7.6e-3, the same range). tf32 and bf16 are 8.5e-3 to 9.5e-3 in weights, also in the range of the `--skip_pad` noise, but they do change the numbers by design. The test run on the real split (`test_gpu.py`) gave ieee 1.2e-2 loss and 5.6e-3 weights, tf32 2.0e-2 and 8.5e-3, bf16 2.2e-2 and 7.6e-3, with noise floors up to 2.7e-3 in weights. The effect on the final metric is not measured.

`--compile` (`torch.compile(dynamic=True)`, off by default):
- The attentive pooling compiles well: 1 graph, 0 recompiles for all batch shapes, compile time 4 to 5 s (15 s with a cold Triton cache). Step rate: unfused 30.8, 20.6, 31.2 against compiled 20.1, 19.3, 30.5; fused 34.9, 27.9, 34.4 against fused and compiled 33.0, 32.3, 31.8. No gain (ratio 0.95 to 1.0 within noise). The matmuls dominate the pooling and stay in cuBLAS.
- The rotary embedding recompiles at every new longest sequence (it caches frequencies by length). It reached the recompile limit of 8 and fell back to eager, so it is not compiled.
- The whole `forward` recompiles every step: the D-MPNN receives Python lists of atom and bond ranges, and dynamo guards on their values. Each compile took 30 to 100 s. After 8 steps the limit stopped it (eager again). It needs the D-MPNN in `catpred/` to take tensors, which this wrapper must not edit.
- Compiled code gives equal maths with other rounding (fused ops). Not recommended: no speed gain.

Test results: CPU `verify_equivalence.sh` 38 passed, 0 skipped (the 2 new tests: CPU error of the kernel, settings keys). On the farm `python -m pytest tests -m gpu` with GPU 0: 40 passed, 1 failed. The failure is `test_cache_vectors_equal_original_esm_call_on_gpu` (ESM cache built on the Blackwell, rounding differs on the A10). `ruff check emulator_bench tests` passes.
