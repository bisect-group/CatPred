"""Train and evaluate one CatPred run: one task, one split folder, one seed, an ensemble of 10 members.

The splits are read from the EMULaToR dataset tree. Rows with residues outside the original vocabulary are dropped and counted.
Members train in separate processes on the same GPU (--workers). Each process rebuilds the same inits and batch orders,
so the result does not depend on the number of workers.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from emulator_bench.common import (  # noqa: E402
    SEQUENCE_COL,
    SMILES_COL,
    TARGET_COL,
    SplitJob,
    default_data_root,
    default_out_root,
    ensure_repo_on_path,
    esm_cache_dir,
    stable_hash,
    unknown_residue_mask,
    write_json,
)
from emulator_bench.esm_cache import EsmCache  # noqa: E402
from emulator_bench.fast_train import (  # noqa: E402
    Options,
    SplitData,
    build_train_args,
    configure_runtime,
    epoch_orders,
    member_inits,
    predict_split,
    settings_options,
    train_member,
)
from emulator_bench.fused_attention import PRECISIONS  # noqa: E402

ensure_repo_on_path()
MIN_HIT_RATE = 0.98  # a lower share of rows with a structure vector means a wrong dict: the rest would get the mean vector
SPLITS = ("train", "val", "test")


def prepare_inputs(paths: dict, run_dir: Path, limit_rows: int | None, write: bool = True):
    """Keep the rows the original model can featurize and write one CSV per split (only if write). Returns sequences, labels and dropped counts."""
    sequences, labels, dropped = {}, {}, {}
    for split, path in paths.items():
        df = pd.read_parquet(path, columns=[SMILES_COL, SEQUENCE_COL, TARGET_COL])
        if limit_rows:
            df = df.head(limit_rows)
        bad = np.array(unknown_residue_mask(df[SEQUENCE_COL]), dtype=bool) | df[TARGET_COL].isna().to_numpy()
        dropped[split] = int(bad.sum())
        df = df[~bad].reset_index(drop=True)
        if write:
            (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
            df.assign(row_id=range(len(df)))[["row_id", SMILES_COL, TARGET_COL]].to_csv(run_dir / "inputs" / f"{split}.csv", index=False)
        sequences[split] = df[SEQUENCE_COL].tolist()
        labels[split] = df[TARGET_COL].to_numpy()
    return sequences, labels, dropped


def load_egnn(path):
    """The ProGRES dict and its mean vector, computed as in MoleculeModel.create_protein_model."""
    vectors = torch.load(path, weights_only=True)
    return vectors, torch.stack(list(vectors.values())).mean(dim=0)


def load_splits(args, run_dir: Path, sequences, labels, esm_cache, egnn=None):
    """The original get_data builds the datapoints and drops rows with invalid SMILES (they are counted).
    The train targets are scaled by the original StandardScaler."""
    from catpred.data import get_data, get_task_names, validate_dataset_type
    from catpred.features import (reset_featurization_parameters, set_adding_hs, set_explicit_h, set_keeping_atom_map)

    reset_featurization_parameters()
    set_explicit_h(args.explicit_h)
    set_adding_hs(args.adding_h)
    set_keeping_atom_map(args.keeping_atom_map)
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)
    data, invalid = {}, {}
    for split in SPLITS:
        data[split] = get_data(path=str(run_dir / "inputs" / f"{split}.csv"), protein_records_path=None, args=args,
                               smiles_columns=args.smiles_columns, loss_function=args.loss_function, skip_none_targets=True,
                               store_row=True)
        kept = [int(d.row["row_id"]) for d in data[split]._data]
        invalid[split] = len(sequences[split]) - len(kept)
        sequences[split] = [sequences[split][i] for i in kept]
        labels[split] = labels[split][kept]
    validate_dataset_type(data["train"], dataset_type=args.dataset_type)
    args.features_size = data["train"].features_size()
    args.train_data_size = len(data["train"])
    scaler = data["train"].normalize_targets()
    graph_by_smiles = {}
    splits = {s: SplitData(data[s], sequences[s], esm_cache, graph_by_smiles, egnn) for s in SPLITS}
    return splits, scaler, invalid


def prepare_run(paths: dict, run_dir: Path, esm_dir: Path, seed: int, epochs: int, ensemble_size: int,
                limit_rows: int | None = None, cpu: bool = False, write_inputs: bool = True, egnn_path=None,
                min_hit_rate: float = MIN_HIT_RATE):
    """Everything a run needs before the first step: args, split data, scaler, labels, dropped counts and member inits."""
    if egnn_path:  # the original model loads the dict itself and accepts only trusted folders
        os.environ["CATPRED_TRUSTED_DESERIALIZATION_ROOTS"] = os.pathsep.join(
            [str(Path(egnn_path).resolve().parent), os.environ.get("CATPRED_TRUSTED_DESERIALIZATION_ROOTS", "")])
    sequences, labels, dropped = prepare_inputs(paths, run_dir, limit_rows, write_inputs)
    args = build_train_args(*(run_dir / "inputs" / f"{s}.csv" for s in SPLITS), run_dir, seed, epochs, ensemble_size, cpu=cpu, egnn_path=egnn_path)
    splits, scaler, invalid = load_splits(args, run_dir, sequences, labels, EsmCache(esm_dir), load_egnn(egnn_path) if egnn_path else None)
    if egnn_path:
        rates = {s: splits[s].egnn_hit_rate for s in SPLITS}
        print(f"[run] rows with a structure vector: {rates}", flush=True)
        if min(rates.values()) < min_hit_rate:
            raise ValueError(f"Structure vector hit rate {rates} is below {min_hit_rate}. Is {egnn_path} the right dict?")
    with contextlib.redirect_stdout(io.StringIO()):
        inits = member_inits(args, ensemble_size, epochs)
    return args, splits, scaler, labels, dropped, invalid, inits


def metrics_of(y_true, y_pred) -> dict:
    from scipy.stats import spearmanr

    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "PCC": float(np.corrcoef(y_true, y_pred)[0, 1]), "SCC": float(spearmanr(y_true, y_pred)[0]),
        "R2": 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot, "RMSE": mse**0.5, "MSE": mse,
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
    }


def member_worker(member_ids, prepared, opts: Options, results_dir: Path, deterministic: bool = False) -> None:
    """Train the listed members one after another and save each result."""
    from catpred.utils import save_checkpoint

    args, splits, scaler, labels, dropped, invalid, inits = prepared
    torch.set_num_threads(2)
    configure_runtime(opts, deterministic)
    for k in member_ids:
        start = time.time()
        orders = islice(epoch_orders(args.seed, splits["train"].n, (k + 1) * args.epochs), k * args.epochs, None)
        model, best_epoch, best_score, history = train_member(args, k, inits[k], orders, splits["train"], splits["val"], scaler,
                                                              args.device, opts, log=lambda m: print(m, flush=True))
        preds = {s: predict_split(model, splits[s], scaler, args.batch_size, args.device, opts)[:, 0] for s in SPLITS}
        (Path(args.save_dir) / f"model_{k}").mkdir(parents=True, exist_ok=True)
        save_checkpoint(str(Path(args.save_dir) / f"model_{k}" / "model.pt"), model, scaler, None, None, None, None, args)
        pd.DataFrame(history).to_csv(Path(args.save_dir) / f"model_{k}" / "history.csv", index=False)
        extra = {**{f"label_{s}": labels[s] for s in SPLITS}, **{f"dropped_{s}": dropped[s] for s in SPLITS},
                 **{f"invalid_{s}": invalid[s] for s in SPLITS}, **{f"egnn_hit_rate_{s}": splits[s].egnn_hit_rate for s in SPLITS}}
        np.savez(results_dir / f"member_{k}.tmp.npz", best_epoch=best_epoch, best_score=best_score, seconds=time.time() - start,
                 **{f"pred_{s}": p for s, p in preds.items()}, **extra)
        (results_dir / f"member_{k}.tmp.npz").rename(results_dir / f"member_{k}.npz")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", required=True)
    p.add_argument("--split_group", required=True)
    p.add_argument("--split_name", default="all")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data_root", default=str(default_data_root()))
    p.add_argument("--out_root", default=str(default_out_root()))
    p.add_argument("--run_dir", default=None, help="Output folder. Default: <out_root>/<task>/<split_group>/<split_name>/seed_<seed>.")
    p.add_argument("--esm_dir", default=None, help="ESM cache folder. Default: the cache under --out_root.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--ensemble_size", type=int, default=10)
    p.add_argument("--workers", type=int, default=3, help="Member processes on the GPU. Does not change the result.")
    p.add_argument("--members", type=int, nargs="+", default=None, help="Internal: train only these members, then stop.")
    p.add_argument("--limit_rows", type=int, default=None, help="Smoke tests only: use the first N rows of each split.")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms(True). Slower. Used by the equivalence tests.")
    p.add_argument("--tf32", action="store_true", help="Deviation: TF32 matmuls.")
    p.add_argument("--amp", action="store_true", help="Deviation: bf16 autocast.")
    p.add_argument("--fused_adam", action="store_true", help="Deviation: fused Adam.")
    p.add_argument("--skip_pad", action="store_true", help="Deviation: skip padded positions in the first pooling layer (other rounding).")
    p.add_argument("--compile", action="store_true", help="Deviation: torch.compile(dynamic=True) of the attentive pooling (same maths, other rounding).")
    p.add_argument("--fused_attn", action="store_true", help="Deviation: Triton kernel for the attention core of the Seq-Attn block (same maths, other rounding, GPU only).")
    p.add_argument("--attn_precision", choices=PRECISIONS, default="ieee", help="Dot products of --fused_attn: ieee (fp32), tf32 or bf16. Only ieee keeps the exact maths.")
    p.add_argument("--egnn_path", default=None, help="ProGRES dict from cache_structures.py. Adds structure features. Default: none.")
    p.add_argument("--min_hit_rate", type=float, default=MIN_HIT_RATE, help="Fail if a split has a lower share of rows with a structure vector.")
    return p.parse_args()


def summarize(run_dir: Path, settings_id: str, a, start: float) -> dict:
    members = [np.load(run_dir / "members" / f"member_{k}.npz") for k in range(a.ensemble_size)]
    first = members[0]
    summary = {"task": a.task, "split_group": a.split_group, "split_name": a.split_name, "seed": a.seed, "settings_id": settings_id,
               **{f"dropped_unknown_residue_{s}": int(first[f"dropped_{s}"]) for s in SPLITS},
               **{f"dropped_invalid_smiles_{s}": int(first[f"invalid_{s}"]) for s in SPLITS},
               **{f"egnn_hit_rate_{s}": float(first[f"egnn_hit_rate_{s}"]) for s in SPLITS},
               "best_epochs": " ".join(str(int(m["best_epoch"])) for m in members),
               "member_seconds": float(sum(m["seconds"] for m in members))}
    for split in SPLITS:
        total = np.zeros(len(first[f"label_{split}"]))
        for m in members:
            total += m[f"pred_{split}"]
        pred, label = total / a.ensemble_size, first[f"label_{split}"]
        pd.DataFrame({f"{TARGET_COL}_pred": pred, f"{TARGET_COL}_label": label}).to_csv(run_dir / f"pred_label_{split}.csv", index=False)
        metrics = metrics_of(label, pred)
        pd.DataFrame([metrics]).to_csv(run_dir / f"final_results_{split}.csv", index=False)
        summary.update({f"{split}_{k}": v for k, v in metrics.items()})
    summary["elapsed_seconds"] = time.time() - start
    return summary


def main() -> None:
    a = parse_args()
    os.environ.setdefault("CATPRED_CACHE_PATH", str(Path(a.out_root).expanduser().resolve() / "embeddings" / "catpred_native_cache"))
    if a.attn_precision != "ieee" and not a.fused_attn:
        raise SystemExit("--attn_precision needs --fused_attn")
    if a.fused_attn and a.cpu:
        raise SystemExit("--fused_attn needs a CUDA device. Remove --cpu.")
    opts = Options(a.tf32, a.amp, a.fused_adam, a.skip_pad, fused_attn=a.fused_attn, attn_precision=a.attn_precision, compile=a.compile)
    out_root = Path(a.out_root).expanduser().resolve()
    if a.egnn_path:
        a.egnn_path = str(Path(a.egnn_path).expanduser().resolve())
    job_dir = Path(a.data_root) / a.task / a.split_group
    job = SplitJob(a.task, a.split_group, a.split_name, job_dir if a.split_name == "all" else job_dir / a.split_name)
    run_dir = Path(a.run_dir) if a.run_dir else job.run_dir(out_root, a.seed)
    esm_dir = Path(a.esm_dir) if a.esm_dir else esm_cache_dir(out_root)
    settings = {"epochs": a.epochs, "ensemble_size": a.ensemble_size, "seed": a.seed, "limit_rows": a.limit_rows,
                "options": settings_options(opts), "egnn": hashlib.sha256(Path(a.egnn_path).read_bytes()).hexdigest() if a.egnn_path else None,
                "splits": {s: hashlib.md5(p.read_bytes()).hexdigest() for s, p in job.paths().items()}}
    settings_id = stable_hash(settings)
    settings_path, summary_path = run_dir / "run_settings.json", run_dir / "run_summary.csv"
    if a.members is None:
        if summary_path.exists() and pd.read_csv(summary_path, dtype={"settings_id": str})["settings_id"][0] == settings_id:
            print(f"Skipping finished run {run_dir}")
            return
        if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
            old = run_dir.with_name(f"{run_dir.name}.old-{stable_hash(json.loads(settings_path.read_text()))}-{int(time.time())}")
            run_dir.rename(old)  # results of other settings must not be reused, and are never deleted
            print(f"[run] settings changed: moved {run_dir} to {old}", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(settings_path, settings)
        (run_dir / "members").mkdir(exist_ok=True)
    start = time.time()
    results_dir = run_dir / "members"
    todo = a.members if a.members is not None else [k for k in range(a.ensemble_size) if not (results_dir / f"member_{k}.npz").exists()]

    if a.members is None and todo:
        prepare_inputs(job.paths(), run_dir, a.limit_rows)  # written once here, workers only read

    def prepare():
        return prepare_run(job.paths(), run_dir, esm_dir, a.seed, a.epochs, a.ensemble_size, a.limit_rows, a.cpu, write_inputs=False, egnn_path=a.egnn_path,
                           min_hit_rate=a.min_hit_rate)

    if a.members is not None or (a.workers <= 1 and todo):
        prepared = prepare()
        print(f"[run] rows dropped (unknown residue or missing target, invalid SMILES): {prepared[4]} {prepared[5]}", flush=True)
        member_worker(todo, prepared, opts, results_dir, a.deterministic)
        if a.members is not None:
            return
    elif todo:
        # Workers are new processes: forking after the CPU thread pools started can deadlock.
        procs = [subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--members", *map(str, todo[w :: a.workers])])
                 for w in range(min(a.workers, len(todo)))]
        if any([proc.wait() for proc in procs]):  # wait for all workers before the check
            raise RuntimeError("a member worker failed")
    summary = summarize(run_dir, settings_id, a, start)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"[run] done {run_dir} test RMSE {summary['test_RMSE']:.4f} in {summary['elapsed_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
