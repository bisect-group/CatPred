"""CPU checks of the wrapper against the original CatPred code on the tiny fixture (64 train, 16 val, 16 test rows).

The reference is catpred.train.run_training. All comparisons are bit for bit unless a test says otherwise.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import FIXTURE_DATA
from emulator_bench import fast_train, train_run
from emulator_bench.common import SEQUENCE_COL, SMILES_COL, TARGET_COL, discover_split_jobs
from emulator_bench.esm_cache import EsmCache, EsmCacheWriter, sequence_key
from emulator_bench.reference_run import Recorder, run_original

SEED, EPOCHS, MEMBERS = 666, 3, 3
GROUP = "random_splits_grouped_sequence"


@pytest.fixture(autouse=True, scope="session")
def deterministic():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def run_wrapper(work, esm_dir, run_dir, workers=1, epochs=EPOCHS, members=MEMBERS, extra=()):
    argv = ["train_run.py", "--task", "kcat", "--split_group", GROUP, "--seed", str(SEED), "--data_root", str(FIXTURE_DATA),
            "--out_root", str(work), "--esm_dir", str(esm_dir), "--run_dir", str(run_dir), "--epochs", str(epochs),
            "--ensemble_size", str(members), "--cpu", "--deterministic", "--workers", str(workers), *extra]
    saved = sys.argv
    sys.argv = argv
    try:
        train_run.main()
    finally:
        sys.argv = saved


@pytest.fixture(scope="module")
def reference(tables, esm_cache, tmp_path_factory):
    return run_original(tables, esm_cache, tmp_path_factory.mktemp("reference"), SEED, EPOCHS, MEMBERS)


@pytest.fixture(scope="module")
def wrapper(work, esm_dir, reference):
    run_dir = work / "wrapper"
    with Recorder() as recorder:
        run_wrapper(work, esm_dir, run_dir)
    return recorder, run_dir


@pytest.fixture(scope="module")
def egnn_path(work, tables):
    """A ProGRES dict with random 128-d vectors for two thirds of the fixture sequences. The others get the mean vector."""
    sequences = sorted(set(pd.concat(tables.values())[SEQUENCE_COL]))
    gen = torch.Generator().manual_seed(0)
    vectors = {sequence_key(seq): torch.randn(128, generator=gen) for i, seq in enumerate(sequences) if i % 3}
    assert 0 < len(vectors) < len(sequences)
    path = work / "egnn" / "dict.pt"
    path.parent.mkdir()
    torch.save(vectors, path)
    return path


@pytest.fixture(scope="module")
def reference_egnn(tables, esm_cache, egnn_path, tmp_path_factory):
    return run_original(tables, esm_cache, tmp_path_factory.mktemp("reference_egnn"), SEED, EPOCHS, MEMBERS, egnn_path=egnn_path)


@pytest.fixture(scope="module")
def wrapper_egnn(work, esm_dir, egnn_path, reference_egnn):
    run_dir = work / "wrapper_egnn"
    with Recorder() as recorder:
        run_wrapper(work, esm_dir, run_dir, extra=["--egnn_path", str(egnn_path), "--min_hit_rate", "0"])
    return recorder, run_dir


def test_esm_cache_equals_original_esm_call(esm_cache, tables):
    from catpred.data.esm_utils import get_protein_embedder

    embed = get_protein_embedder("esm")["fn"]
    sequences = sorted(set(pd.concat(tables.values())[SEQUENCE_COL]))
    assert len(esm_cache) == len(sequences)
    for seq in sequences[:20]:
        original = embed(seq, name="x", device="cpu")[0][0].numpy()
        stored = esm_cache.get(seq)
        assert stored.dtype == np.float32 and stored.shape == (len(seq), 1280)
        assert np.array_equal(stored, original)


def test_esm_truncation_to_2047_rows(tmp_path):
    from catpred.data.esm_utils import get_single_esm_repr

    seq = ("ACDEFGHIKLMNPQRSTVWY" * 105)[:2100]
    vectors = get_single_esm_repr(seq).detach().numpy()
    assert vectors.shape == (2047, 1280)
    writer = EsmCacheWriter(tmp_path, 0)
    writer.add(seq, vectors)
    writer.close()
    assert np.array_equal(EsmCache(tmp_path).get(seq), vectors)


def test_esm_cache_resumes_after_crash(tmp_path):
    rows = np.random.default_rng(0).random((5, 1280), dtype=np.float32)
    writer = EsmCacheWriter(tmp_path, 0)
    writer.add("AAAAA", rows)
    writer.flush()
    writer.add("CCCCC", rows)  # written to the data file, but not indexed
    writer.file.close()
    resumed = EsmCacheWriter(tmp_path, 0)
    assert resumed.done == {train_run_key("AAAAA")}
    resumed.add("CCCCC", rows * 2)
    resumed.close()
    cache = EsmCache(tmp_path)
    assert np.array_equal(cache.get("AAAAA"), rows) and np.array_equal(cache.get("CCCCC"), rows * 2)


def train_run_key(seq):
    from emulator_bench.esm_cache import sequence_key

    return sequence_key(seq)


def test_split_discovery_finds_every_folder(tmp_path):
    for group in ("random_splits_grouped_sequence", "uniprot_time_splits"):
        (tmp_path / "kcat" / group).mkdir(parents=True)
        (tmp_path / "kcat" / group / "train.parquet").touch()
    for group, thresholds in (("enzyme_sequence_splits", ("threshold_0.5", "threshold_0.9")), ("substrate_splits", ("threshold_0.3",))):
        for t in thresholds:
            (tmp_path / "km" / group / t).mkdir(parents=True)
            (tmp_path / "km" / group / t / "train.parquet").touch()
    found = {(j.task, j.split_group, j.split_name) for j in discover_split_jobs(tmp_path)}
    assert found == {("kcat", "random_splits_grouped_sequence", "all"), ("kcat", "uniprot_time_splits", "all"),
                     ("km", "enzyme_sequence_splits", "threshold_0.5"), ("km", "enzyme_sequence_splits", "threshold_0.9"),
                     ("km", "substrate_splits", "threshold_0.3")}


def test_unknown_residues_and_missing_targets_are_dropped(tmp_path):
    df = pd.DataFrame({SMILES_COL: ["CCO"] * 5, SEQUENCE_COL: ["ACDE", "ACXE", "BBBB", "ACDU", "ACDF"],
                       TARGET_COL: [1.0, 2.0, 3.0, 4.0, float("nan")]})
    paths = {}
    for split in ("train", "val", "test"):
        paths[split] = tmp_path / f"{split}.parquet"
        df.to_parquet(paths[split])
    sequences, labels, dropped = train_run.prepare_inputs(paths, tmp_path / "run", None)
    assert sequences["train"] == ["ACDE"] and labels["train"].tolist() == [1.0]
    assert dropped == {"train": 4, "val": 4, "test": 4}


def test_summary_counts_invalid_smiles(wrapper):
    summary = pd.read_csv(wrapper[1] / "run_summary.csv").iloc[0]
    assert summary["dropped_invalid_smiles_train"] == 1 and summary["dropped_unknown_residue_train"] == 0


def test_init_weights_and_rng_order_equal_original(reference, wrapper):
    assert len(reference.inits) == 2 * MEMBERS
    for ours, theirs in zip(wrapper[0].inits[: 2 * MEMBERS], reference.inits):
        assert all(torch.equal(ours[k], theirs[k]) for k in theirs)


def test_batch_order_equals_original_sampler(reference, wrapper):
    n_train = len(pd.read_csv(wrapper[1] / "inputs" / "train.csv")) - 1  # one SMILES is invalid
    ours = list(fast_train.epoch_orders(SEED, n_train, MEMBERS * EPOCHS))
    assert ours == reference.orders


def test_per_step_losses_and_weights_equal_original(reference, wrapper):
    recorder = wrapper[0]
    assert len(reference.losses) == len(recorder.losses) == MEMBERS * EPOCHS * 4
    assert all(torch.equal(a, b) for a, b in zip(reference.losses, recorder.losses))
    assert all(torch.equal(a, b) for a, b in zip(reference.params, recorder.params))


def test_validation_scores_best_epoch_and_test_predictions_equal_original(reference, wrapper):
    for k in range(MEMBERS):
        history = pd.read_csv(wrapper[1] / f"model_{k}" / "history.csv", float_precision="round_trip")
        scores = reference.val_scores[k * EPOCHS : (k + 1) * EPOCHS]
        assert history["val_rmse"].tolist() == [float(x) for x in scores]
        member = np.load(wrapper[1] / "members" / f"member_{k}.npz")
        assert int(member["best_epoch"]) == int(np.argmin(scores))  # the original keeps the first minimum
        assert np.array_equal(member["pred_test"], reference.test_preds[k])


def test_ensemble_prediction_equals_original_average(reference, wrapper, tables):
    pred = pd.read_csv(wrapper[1] / "pred_label_test.csv", float_precision="round_trip")
    mean = sum(reference.test_preds) / MEMBERS
    assert np.array_equal(pred[f"{TARGET_COL}_pred"].to_numpy(), mean)
    assert np.array_equal(pred[f"{TARGET_COL}_label"].to_numpy(), tables["test"][TARGET_COL].to_numpy())


def test_structure_features_per_step_losses_weights_and_predictions_equal_original(reference_egnn, wrapper_egnn):
    recorder, run_dir = wrapper_egnn
    assert len(reference_egnn.losses) == len(recorder.losses) == MEMBERS * EPOCHS * 4
    assert all(torch.equal(a, b) for a, b in zip(reference_egnn.losses, recorder.losses))
    assert all(torch.equal(a, b) for a, b in zip(reference_egnn.params, recorder.params))
    for k in range(MEMBERS):
        assert np.array_equal(np.load(run_dir / "members" / f"member_{k}.npz")["pred_test"], reference_egnn.test_preds[k])


def test_structure_features_change_the_model(reference, reference_egnn):
    assert not torch.equal(reference.losses[0], reference_egnn.losses[0])


def test_members_in_separate_processes_give_the_same_result(work, esm_dir, wrapper):
    run_wrapper(work, esm_dir, work / "workers3", workers=3)
    for k in range(MEMBERS):
        a, b = np.load(wrapper[1] / "members" / f"member_{k}.npz"), np.load(work / "workers3" / "members" / f"member_{k}.npz")
        assert all(np.array_equal(a[f"pred_{s}"], b[f"pred_{s}"]) for s in ("train", "val", "test"))


def test_finished_run_is_skipped_and_changed_settings_restart(work, esm_dir, wrapper, capsys):
    run_wrapper(work, esm_dir, wrapper[1])
    assert "Skipping finished run" in capsys.readouterr().out
    run_wrapper(work, esm_dir, work / "restart", epochs=1, members=1)
    assert (work / "restart" / "run_summary.csv").exists()
    run_wrapper(work, esm_dir, work / "restart", epochs=2, members=1)
    assert len(pd.read_csv(work / "restart" / "model_0" / "history.csv")) == 2
    assert len(pd.read_csv(next(work.glob("restart.old-*")) / "model_0" / "history.csv")) == 1  # the old run is kept


def test_low_structure_hit_rate_fails_and_rates_are_reported(work, esm_dir, egnn_path, wrapper_egnn, tmp_path):
    with pytest.raises(ValueError, match="hit rate"):
        setup_split(esm_dir, tmp_path, egnn_path, min_hit_rate=0.98)
    _, splits, _ = setup_split(esm_dir, tmp_path, egnn_path)
    assert all(0.4 < splits[s].egnn_hit_rate < 0.9 for s in ("train", "val", "test"))
    summary = pd.read_csv(work / "wrapper_egnn" / "run_summary.csv")
    assert 0.4 < summary["egnn_hit_rate_train"][0] < 0.9


def original_batch(split, rows, esm_cache):
    """An original MoleculeDataset batch whose records carry the sequence and its ESM tensor."""
    from catpred.data.data import MoleculeDataset

    points = []
    for r in rows:
        d = split.dataset._data[r]
        seq = split.sequences[r]
        d.protein_record = {"name": sequence_key(seq), "seq": seq, "esm2_feats": torch.from_numpy(np.array(esm_cache.get(seq)))}
        points.append(d)
    return MoleculeDataset(points)


def setup_split(esm_dir, tmp_path, egnn_path=None, min_hit_rate=0):
    paths = {s: FIXTURE_DATA / "kcat" / GROUP / f"{s}.parquet" for s in ("train", "val", "test")}
    args, splits, scaler, *_ = train_run.prepare_run(paths, tmp_path, esm_dir, SEED, 1, 1, None, True, egnn_path=egnn_path, min_hit_rate=min_hit_rate)
    return args, splits, scaler


@pytest.mark.parametrize("structure", [False, True])
@pytest.mark.parametrize("train_mode", [True, False])
def test_forward_and_gradients_equal_original_model(esm_dir, esm_cache, tmp_path, train_mode, structure, egnn_path):
    from catpred.models import MoleculeModel
    from catpred.train.loss_functions import get_loss_func

    args, splits, _ = setup_split(esm_dir, tmp_path, egnn_path if structure else None)
    split = splits["train"]
    torch.manual_seed(1)
    model = MoleculeModel(args)
    model.train(train_mode)
    rows = list(range(40, 56))
    batch = original_batch(split, rows, esm_cache)
    out = model(batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_descriptors(),
                batch.bond_features(), None, None)
    loss_func = get_loss_func(args)
    targets = torch.tensor(batch.targets())
    loss_func(out, targets).sum().div(16).backward()
    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad()

    hb = fast_train.build_host_batch(split, rows)
    db = fast_train.assemble(hb, hb.tensors())
    fast_out = fast_train.forward(model, db)
    loss_func(fast_out, db.targets).sum().div(16).backward()
    assert torch.equal(out, fast_out)
    assert all(torch.equal(g, p.grad) for g, p in zip(grads, [p for p in model.parameters() if p.grad is not None]))


def test_batch_tensors_equal_batchmolgraph(esm_dir, esm_cache, tmp_path):
    args, splits, _ = setup_split(esm_dir, tmp_path)
    split = splits["train"]
    rng = np.random.default_rng(0)
    for rows in (list(rng.permutation(split.n)[:16]), list(rng.permutation(split.n)[:7]), [3]):
        rows = [int(r) for r in rows]
        graph = original_batch(split, rows, esm_cache).batch_graph()[0]
        hb = fast_train.build_host_batch(split, rows)
        db = fast_train.assemble(hb, hb.tensors())
        for ours, theirs in zip(db.graph.get_components()[:5], graph.get_components()[:5]):
            assert ours.dtype == theirs.dtype and torch.equal(ours, theirs)
        assert db.graph.get_components()[5:] == graph.get_components()[5:]


def test_tokens_and_esm_are_cut_to_the_esm_length(esm_dir, esm_cache, tmp_path):
    """When a sequence is longer than its ESM rows (cut at 2048 tokens), the original cuts tokens and ESM to the same length."""
    args, splits, _ = setup_split(esm_dir, tmp_path)
    split = splits["train"]

    split.esm = [e[:10] for e in split.esm]
    rows = list(range(16))
    hb = fast_train.build_host_batch(split, rows)
    assert hb.tokens.shape == (16, 10) and hb.dims[-1] == 10
    from catpred.models import MoleculeModel

    torch.manual_seed(1)
    model = MoleculeModel(args)
    batch = original_batch(split, rows, esm_cache)
    for d, e in zip(batch._data, split.esm):
        d.protein_record["esm2_feats"] = torch.from_numpy(np.array(e))
    out = model(batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_descriptors(),
                batch.bond_features(), None, None)
    db = fast_train.assemble(hb, hb.tensors())
    assert torch.equal(out, fast_train.forward(model, db))


def test_launcher_dry_run_lists_every_seed_and_split(work, esm_dir):
    import subprocess

    cmd = [sys.executable, str(FIXTURE_DATA.parents[2] / "emulator_bench" / "launch_parallel_retrain.py"), "--dry_run",
           "--data_root", str(FIXTURE_DATA), "--out_root", str(work), "--esm_dir", str(esm_dir)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    assert out.splitlines()[0].startswith("3 runs: 1 split folders x 3 seeds")
    assert [line.strip() for line in out.splitlines()[1:]] == [f"kcat/{GROUP}/all/seed_{s}" for s in (666, 777, 888)]


def test_launcher_rejects_partial_esm_cache(work, esm_dir, tmp_path):
    import json
    import subprocess

    index = json.loads((esm_dir / "shard_0.json").read_text())
    partial = {k: v[:-1] for k, v in index.items()}
    (tmp_path / "shard_0.json").write_text(json.dumps(partial))
    cmd = [sys.executable, str(FIXTURE_DATA.parents[2] / "emulator_bench" / "launch_parallel_retrain.py"), "--dry_run",
           "--data_root", str(FIXTURE_DATA), "--out_root", str(work), "--esm_dir", str(tmp_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0 and "lacks" in result.stderr


def test_touching_the_split_files_keeps_a_finished_run(work, esm_dir, tmp_path, capsys):
    import shutil

    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DATA, data)
    argv = ["train_run.py", "--task", "kcat", "--split_group", GROUP, "--seed", str(SEED), "--data_root", str(data),
            "--out_root", str(work), "--esm_dir", str(esm_dir), "--run_dir", str(tmp_path / "run"), "--epochs", "1",
            "--ensemble_size", "1", "--cpu", "--deterministic"]
    saved = sys.argv
    sys.argv = argv
    try:
        train_run.main()
        for f in data.rglob("*.parquet"):
            f.touch()
        capsys.readouterr()
        train_run.main()
    finally:
        sys.argv = saved
    assert "Skipping finished run" in capsys.readouterr().out


def test_workers_do_not_write_inputs(tmp_path):
    paths = next(j for j in discover_split_jobs(FIXTURE_DATA, ["kcat"]) if j.split_group == GROUP).paths()
    train_run.prepare_inputs(paths, tmp_path, None, write=False)
    assert not (tmp_path / "inputs").exists()


def fp64_batch(split, rows):
    """A fast-path batch with every float tensor in float64."""
    hb = fast_train.build_host_batch(split, rows)
    db = fast_train.assemble(hb, hb.tensors())
    f_atoms, f_bonds, *rest = db.graph.get_components()
    db.graph = fast_train.FastGraph((f_atoms.double(), f_bonds.double(), *rest))
    for name in ("esm", "esm_rows", "egnn"):
        setattr(db, name, getattr(db, name).double())
    return db


def forward_and_grads(model, db, opts):
    from catpred.train.loss_functions import get_loss_func

    model.zero_grad()
    out = fast_train.forward(model, db, opts)
    get_loss_func(model.args)(out, db.targets).sum().div(len(db.targets)).backward()
    return out.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]


@pytest.mark.parametrize("structure", [False, True])
def test_skip_pad_equals_exact_path_in_float64(esm_dir, tmp_path, structure, egnn_path):
    """In float64 the rounding is negligible, so a difference above 1e-12 would be a difference in the maths."""
    from catpred.models import MoleculeModel

    args, splits, _ = setup_split(esm_dir, tmp_path, egnn_path if structure else None)
    torch.manual_seed(1)
    model = MoleculeModel(args).double()
    for p in model.parameters():  # initialize_weights zeroes the biases, which would hide a bias error
        if p.dim() == 1:
            torch.nn.init.normal_(p)
    for rows in (list(range(40, 56)), list(range(5)), [7]):
        db = fp64_batch(splits["train"], rows)
        assert len(rows) == 1 or (db.tokens == fast_train.PAD_ID).any()  # padded positions are present
        exact_out, exact_grads = forward_and_grads(model, db, fast_train.Options())
        skip_out, skip_grads = forward_and_grads(model, db, fast_train.Options(skip_pad=True))
        out_diff = (exact_out - skip_out).abs().max().item()
        grad_diff = max((a - b).abs().max().item() for a, b in zip(exact_grads, skip_grads))
        print(f"\n[skip_pad fp64] batch of {len(rows)}: output diff {out_diff:.1e}, gradient diff {grad_diff:.1e} "
              f"(largest gradient {max(g.abs().max().item() for g in exact_grads):.1e}, padded share {(db.tokens == fast_train.PAD_ID).float().mean().item():.2f})")
        assert out_diff < 1e-12 and grad_diff < 1e-12
        assert len(exact_grads) == len(skip_grads)


def test_skip_pad_float32_differs_from_exact_only_by_rounding(esm_dir, tmp_path):
    from catpred.models import MoleculeModel

    args, splits, _ = setup_split(esm_dir, tmp_path)
    torch.manual_seed(1)
    model = MoleculeModel(args)
    hb = fast_train.build_host_batch(splits["train"], list(range(40, 56)))
    db = fast_train.assemble(hb, hb.tensors())
    exact_out, exact_grads = forward_and_grads(model, db, fast_train.Options())
    skip_out, skip_grads = forward_and_grads(model, db, fast_train.Options(skip_pad=True))
    assert torch.allclose(exact_out, skip_out, rtol=1e-4, atol=1e-5)
    assert all(torch.allclose(a, b, rtol=1e-3, atol=1e-5) for a, b in zip(exact_grads, skip_grads))


def test_options_and_structure_dict_are_part_of_the_run_settings(work, esm_dir, egnn_path):
    run_dir = work / "options"
    run_wrapper(work, esm_dir, run_dir, epochs=1, members=1, extra=["--amp", "--skip_pad", "--egnn_path", str(egnn_path), "--min_hit_rate", "0"])
    settings = json.loads((run_dir / "run_settings.json").read_text())
    assert settings["options"]["amp"] and settings["options"]["skip_pad"] and settings["egnn"]
    first_id = pd.read_csv(run_dir / "run_summary.csv", dtype={"settings_id": str})["settings_id"][0]
    assert np.isfinite(pd.read_csv(run_dir / "pred_label_test.csv").to_numpy()).all()
    run_wrapper(work, esm_dir, run_dir, epochs=1, members=1, extra=["--egnn_path", str(egnn_path), "--min_hit_rate", "0"])
    settings = json.loads((run_dir / "run_settings.json").read_text())
    assert not settings["options"]["amp"] and not settings["options"]["skip_pad"] and settings["egnn"]
    assert pd.read_csv(run_dir / "run_summary.csv", dtype={"settings_id": str})["settings_id"][0] != first_id
