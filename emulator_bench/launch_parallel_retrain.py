"""Launch every CatPred run: kcat and km, all split folders found on disk, seeds 666 777 888.

Each run is one train_run.py process with one GPU (CUDA_VISIBLE_DEVICES). A finished run is skipped when its settings match.
Run order: task, then seed, then split folder, so each seed finishes on all splits early.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import SEEDS, SPLIT_GROUPS, TASKS, default_data_root, default_out_root, discover_split_jobs, esm_cache_dir  # noqa: E402

from emulator_bench.cache_embeddings import unique_sequences  # noqa: E402
from emulator_bench.esm_cache import sequence_key  # noqa: E402

TRAIN_RUN = Path(__file__).with_name("train_run.py")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_root", default=str(default_data_root()))
    p.add_argument("--out_root", default=str(default_out_root()))
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--split_groups", nargs="+", default=list(SPLIT_GROUPS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--gpus", nargs="+", default=["0"], help="GPU ids. Each run gets one GPU.")
    p.add_argument("--runs_per_gpu", type=int, default=1, help="Concurrent runs on each GPU.")
    p.add_argument("--workers", type=int, default=3, help="Member processes inside each run. Does not change the result.")
    p.add_argument("--amp", action="store_true", help="Deviation: bf16 autocast in the forward pass.")
    p.add_argument("--tf32", action="store_true", help="Deviation: TF32 matmuls.")
    p.add_argument("--fused_adam", action="store_true", help="Deviation: fused Adam.")
    p.add_argument("--skip_pad", action="store_true", help="Deviation: skip padded positions in the first pooling layer (other rounding).")
    p.add_argument("--compile", action="store_true", help="Deviation: torch.compile(dynamic=True) of the attentive pooling (same maths, other rounding).")
    p.add_argument("--fused_attn", action="store_true", help="Deviation: Triton kernel for the attention core of the Seq-Attn block (same maths, other rounding, GPU only).")
    p.add_argument("--attn_precision", choices=("ieee", "tf32", "bf16"), default="ieee", help="Dot products of --fused_attn. Only ieee keeps the exact maths.")
    p.add_argument("--egnn_path", default=None, help="ProGRES dict from cache_structures.py. Adds structure features. Default: none.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--ensemble_size", type=int, default=10)
    p.add_argument("--limit_rows", type=int, default=None, help="Smoke tests only: first N rows of each split.")
    p.add_argument("--esm_dir", default=None, help="ESM cache folder. Default: the cache under --out_root.")
    p.add_argument("--cpu", action="store_true", help="Smoke tests only.")
    p.add_argument("--dry_run", action="store_true", help="List the runs and stop.")
    a = p.parse_args()

    out_root = Path(a.out_root).expanduser().resolve()
    jobs = discover_split_jobs(Path(a.data_root), a.tasks, a.split_groups)
    runs = [(job, seed) for task in a.tasks for seed in a.seeds for job in jobs if job.task == task]
    flags = [f"--{f}" for f in ("amp", "tf32", "fused_adam", "skip_pad", "fused_attn", "compile", "cpu") if getattr(a, f)]
    flags += ["--attn_precision", a.attn_precision] * (a.attn_precision != "ieee")
    print(f"{len(runs)} runs: {len(jobs)} split folders x {len(a.seeds)} seeds, {a.ensemble_size} members x {a.epochs} epochs each")
    for job, seed in runs:
        print(f"  {job.run_dir(out_root, seed).relative_to(out_root)}")
    esm_dir = Path(a.esm_dir) if a.esm_dir else esm_cache_dir(out_root)
    cached = {k for f in esm_dir.glob("shard_*.json") for k in json.loads(f.read_text())["keys"]}
    missing = sum(sequence_key(s) not in cached for s in unique_sequences(Path(a.data_root), a.tasks)[0])
    if missing:
        sys.exit(f"ESM cache in {esm_dir} lacks {missing} sequences. Run emulator_bench/cache_embeddings.py first.")
    if a.egnn_path:
        if not Path(a.egnn_path).expanduser().exists():
            sys.exit(f"ProGRES dict {a.egnn_path} not found. Run emulator_bench/cache_structures.py first.")
        flags += ["--egnn_path", str(Path(a.egnn_path).expanduser().resolve())]
        print(f"structure features: {a.egnn_path}")
    if a.dry_run:
        return

    queue, failed = Queue(), []
    for run in runs:
        queue.put(run)

    def slot(gpu: str) -> None:
        while True:
            try:
                job, seed = queue.get_nowait()
            except Empty:
                return
            run_dir = job.run_dir(out_root, seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(TRAIN_RUN), "--task", job.task, "--split_group", job.split_group, "--split_name", job.split_name,
                   "--seed", str(seed), "--data_root", a.data_root, "--out_root", str(out_root), "--esm_dir", str(esm_dir),
                   "--epochs", str(a.epochs), "--ensemble_size", str(a.ensemble_size), "--workers", str(a.workers), *flags]
            if a.limit_rows:
                cmd += ["--limit_rows", str(a.limit_rows)]
            start = time.time()
            with open(run_dir.parent / f"seed_{seed}.log", "a") as log:
                code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}).returncode
            print(f"[{time.strftime('%H:%M:%S')}] gpu {gpu} {run_dir.relative_to(out_root)} exit {code} after {time.time() - start:.0f}s", flush=True)
            if code:
                failed.append(run_dir)

    threads = [threading.Thread(target=slot, args=(gpu,)) for gpu in a.gpus for _ in range(a.runs_per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        sys.exit(f"{len(failed)} runs failed: {[str(f) for f in failed]}")


if __name__ == "__main__":
    main()
