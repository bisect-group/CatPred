"""GPU checks on a real kcat split with the built ESM cache and ProGRES dict. Run with `verify_equivalence.sh --gpu`.

GPU training of the original code is not bit-reproducible: the backward pass of index_select uses atomic adds.
So the tests compare forward passes bit for bit, and gradients and weights against the noise of the original loop
(two runs of the original loop with the same inputs).
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import REAL_OUT_ROOT
from emulator_bench import fast_train, train_run
from emulator_bench.common import default_data_root, esm_cache_dir
from emulator_bench.cache_structures import dict_path
from emulator_bench.esm_cache import EsmCache, sequence_key
from emulator_bench.reference_run import Recorder

pytestmark = pytest.mark.gpu
SEED, STEPS = 666, 20
SPLIT = default_data_root() / "kcat" / "random_splits_grouped_sequence"
EGNN = dict_path(REAL_OUT_ROOT)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    assert torch.cuda.is_available(), "GPU mode needs a CUDA device"
    torch.set_num_threads(2)
    run_dir = tmp_path_factory.mktemp("real")
    args, splits, scaler, labels, dropped, invalid, inits = train_run.prepare_run(
        {s: SPLIT / f"{s}.parquet" for s in ("train", "val", "test")}, run_dir, esm_cache_dir(REAL_OUT_ROOT), SEED, 1, 1, egnn_path=EGNN)
    return args, splits, scaler, inits


def test_cache_vectors_equal_original_esm_call_on_gpu():
    from catpred.data.esm_utils import get_single_esm_repr

    cache = EsmCache(esm_cache_dir(REAL_OUT_ROOT))
    table = pd.read_parquet(SPLIT / "train.parquet", columns=["sequence"])
    for seq in table.sequence.drop_duplicates().sample(6, random_state=0):
        if seq in cache:
            assert np.array_equal(cache.get(seq), get_single_esm_repr(seq).cpu().numpy())


def original_forward(model, split, rows):
    from catpred.data.data import MoleculeDataset

    points = []
    for r in rows:
        d = split.dataset._data[r]
        d.protein_record = {"name": sequence_key(split.sequences[r]), "seq": split.sequences[r], "esm2_feats": torch.from_numpy(np.array(split.esm[r]))}
        points.append(d)
    batch = MoleculeDataset(points)
    return model(batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(),
                 batch.bond_descriptors(), batch.bond_features(), None, None)


def test_forward_equals_original_on_gpu_batches(real):
    from catpred.models import MoleculeModel

    args, splits, _, inits = real
    split = splits["train"]
    model = MoleculeModel(args)
    model.load_state_dict(inits[0])
    model = model.to(args.device)
    for rows in fast_train.batches_of(next(fast_train.epoch_orders(SEED, split.n, 1)), 16)[:5]:
        with torch.no_grad():
            reference = original_forward(model, split, rows)
            for db in fast_train.Prefetcher(split, [rows], args.device):
                assert torch.equal(reference, fast_train.forward(model, db))


def test_amp_and_skip_pad_forward_stay_close_to_exact_on_gpu(real):
    from catpred.models import MoleculeModel

    args, splits, _, inits = real
    split = splits["train"]
    model = MoleculeModel(args)
    model.load_state_dict(inits[0])
    model = model.to(args.device)
    for rows in fast_train.batches_of(next(fast_train.epoch_orders(SEED, split.n, 1)), 16)[:5]:
        with torch.no_grad():
            for db in fast_train.Prefetcher(split, [rows], args.device):
                exact = fast_train.forward(model, db)
                skip = fast_train.forward(model, db, fast_train.Options(skip_pad=True))
                assert torch.allclose(exact, skip, rtol=1e-3, atol=1e-4)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    amp = fast_train.forward(model, db, fast_train.Options(amp=True, skip_pad=True)).float()
                assert torch.allclose(exact, amp, rtol=0.1, atol=0.1)


def run_steps(args, splits, inits, fast: bool):
    """STEPS training steps from the first member init. Returns per-step losses and weights."""
    from catpred.models import MoleculeModel
    import catpred.train.loss_functions as loss_module
    from catpred.utils import build_lr_scheduler, build_optimizer

    split = splits["train"]
    rows = next(fast_train.epoch_orders(SEED, split.n, 1))[: STEPS * 16]
    model = MoleculeModel(args)
    model.load_state_dict(inits[0])
    model = model.to(args.device)
    optimizer = build_optimizer(model, args)
    scheduler = build_lr_scheduler(optimizer, args)
    with Recorder() as rec:
        loss_func = loss_module.get_loss_func(args)  # the recorder wraps it
        if fast:
            counts = {}
            for db in fast_train.Prefetcher(split, fast_train.batches_of(rows, 16), args.device):
                fast_train.train_step(model, db, loss_func, optimizer, scheduler, counts, fast_train.Options())
        else:
            from catpred.data.data import MoleculeDataLoader, MoleculeDataset, set_cache_graph
            from catpred.train.train import train

            set_cache_graph(False)
            points = []
            for r in rows:
                d = split.dataset._data[r]
                d.protein_record = {"name": sequence_key(split.sequences[r]), "seq": split.sequences[r], "esm2_feats": torch.from_numpy(np.array(split.esm[r]))}
                points.append(d)
            train(model, MoleculeDataLoader(MoleculeDataset(points), batch_size=16, num_workers=0), loss_func, optimizer, scheduler, args)
    return torch.stack([x.sum() for x in rec.losses]), torch.stack(rec.params)


def test_training_steps_match_original_within_gpu_noise(real):
    args, splits, _, inits = real
    loss_a, params_a = run_steps(args, splits, inits, fast=False)
    loss_b, params_b = run_steps(args, splits, inits, fast=False)
    loss_f, params_f = run_steps(args, splits, inits, fast=True)
    noise = (params_a - params_b).abs().max().item()
    gap = (params_a - params_f).abs().max().item()
    print(f"\n[gpu] {STEPS} steps: original vs original {noise:.3e}, original vs fast {gap:.3e} (max abs weight difference)")
    print(f"[gpu] max loss difference: original vs original {(loss_a - loss_b).abs().max().item():.3e}, "
          f"original vs fast {(loss_a - loss_f).abs().max().item():.3e}")
    assert gap <= max(10 * noise, 1e-6)
    assert torch.allclose(loss_a, loss_f, rtol=1e-4, atol=1e-5)


def test_side_stream_batches_equal_cpu_batches(real):
    args, splits, _, _ = real
    split = splits["train"]
    rows = next(fast_train.epoch_orders(SEED, split.n, 1))[:16]
    hb = fast_train.build_host_batch(split, rows)
    cpu = fast_train.assemble(hb, hb.tensors())
    for db in fast_train.Prefetcher(split, [rows], args.device):
        assert torch.equal(db.esm.cpu(), cpu.esm) and torch.equal(db.tokens.cpu(), cpu.tokens)
        assert all(torch.equal(a.cpu(), b) for a, b in zip(db.graph.get_components()[:5], cpu.graph.get_components()[:5]))


def test_members_in_separate_gpu_processes_match_one_process(tmp_path):
    """Two workers against one worker on the real split with a short run. Bit-equal with deterministic GPU kernels."""
    import gc
    import subprocess
    import sys

    gc.collect()
    torch.cuda.empty_cache()  # the earlier tests in this process hold cached GPU memory the workers need
    common = [sys.executable, str(train_run.__file__), "--task", "kcat", "--split_group", "random_splits_grouped_sequence",
              "--seed", str(SEED), "--epochs", "1", "--ensemble_size", "2", "--limit_rows", "800", "--esm_dir", str(esm_cache_dir(REAL_OUT_ROOT)),
              "--egnn_path", str(EGNN), "--deterministic"]
    for name, workers in (("one", 1), ("two", 2)):
        done = subprocess.run([*common, "--run_dir", str(tmp_path / name), "--workers", str(workers)], capture_output=True, text=True,
                              env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")})
        assert done.returncode == 0, done.stderr[-1500:]
    for k in range(2):
        a, b = np.load(tmp_path / "one" / "members" / f"member_{k}.npz"), np.load(tmp_path / "two" / "members" / f"member_{k}.npz")
        assert np.array_equal(a["pred_test"], b["pred_test"])


def test_real_dict_covers_every_split_and_esm3_files_resolve(capsys):
    from emulator_bench.common import SEQUENCE_COL, TASKS, discover_split_jobs, unknown_residue_mask
    from emulator_bench.structures import ESM3_PREFIX, StructureResolver

    esm3_dir = REAL_OUT_ROOT / "structures" / "esm"
    esm3 = StructureResolver.from_dirs(esm3_dir, esm3_dir, esm3_dir, esm3_dir, esm3_dir).esm3
    assert len(esm3) > 900 and all(p.name.startswith(ESM3_PREFIX) and p.exists() for p in list(esm3.values())[:50])
    vectors = torch.load(EGNN, weights_only=True)
    with capsys.disabled():
        for job in discover_split_jobs(default_data_root(), TASKS):
            rates = {}
            for split, path in job.paths().items():
                seqs = pd.read_parquet(path, columns=[SEQUENCE_COL])[SEQUENCE_COL]
                seqs = seqs[~np.array(unknown_residue_mask(seqs), dtype=bool)]
                rates[split] = float(np.mean([sequence_key(s) in vectors for s in seqs]))
            print(f"\n[hit rate] {job.task} {job.split_group}/{job.split_name}: " + " ".join(f"{k} {v:.4f}" for k, v in rates.items()))
            assert min(rates.values()) >= train_run.MIN_HIT_RATE


@pytest.mark.parametrize("options", [dict(fused_attn=True), dict(fused_attn=True, attn_precision="tf32"), dict(fused_attn=True, attn_precision="bf16"),
                                     dict(compile=True), dict(compile=True, fused_attn=True)])
def test_fused_attn_and_compile_forward_stay_close_to_skip_pad_on_gpu(real, options):
    from catpred.models import MoleculeModel

    args, splits, _, inits = real
    split = splits["train"]
    model = MoleculeModel(args)
    model.load_state_dict(inits[0])
    model = model.to(args.device)
    tolerance = {"tf32": 2e-2, "bf16": 1e-1}.get(options.get("attn_precision"), 1e-4)
    for rows in fast_train.batches_of(next(fast_train.epoch_orders(SEED, split.n, 1)), 16)[:5]:
        with torch.no_grad():
            for db in fast_train.Prefetcher(split, [rows], args.device):
                base = fast_train.forward(model, db, fast_train.Options(skip_pad=True))
                out = fast_train.forward(model, db, fast_train.Options(skip_pad=True, **options))
                assert torch.allclose(base, out, rtol=tolerance, atol=tolerance), (options, (base - out).abs().max().item())


@pytest.mark.parametrize("precision", ["ieee", "tf32", "bf16"])
def test_fused_attn_training_drift_against_noise_floor_on_gpu(real, precision):
    """200 steps from one init, as `bench_speed.py --mode drift`: exact against exact is the noise floor."""
    from emulator_bench.bench_speed import drift

    args, splits, _, inits = real
    split = splits["train"]
    rows = next(fast_train.epoch_orders(SEED, split.n, 1))
    result = drift(args, split, rows, inits[0], fast_train.Options(skip_pad=True, fused_attn=True, attn_precision=precision), 200)
    print(f"\n[gpu drift 200 steps, skip_pad + fused_attn {precision}] {result}")
    assert result["option_weights"] < 2e-2 and result["option_loss"] < 0.1
