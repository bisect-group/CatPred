import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

from src.utils.rich_progress import progress  # noqa: E402

from common import (  # noqa: E402
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_VALUE_TYPES,
    ENZYME_FEATURES,
    REPO_ROOT,
    default_base_dir,
    default_data_root,
    discover_split_jobs,
)

ALIGN_SCRIPT = HERE / "align_structures.py"
CACHE_SCRIPT = HERE / "cache_embeddings.py"
TRAIN_SCRIPT = HERE / "train_single_target_tvt.py"
_ACTIVE_PROCESSES: dict[int, tuple[subprocess.Popen, str]] = {}
_ACTIVE_LOCK = threading.Lock()
_STOP_REQUESTED = threading.Event()


def _parse_gpus(items) -> list[str]:
    out = [part.strip() for item in items for part in str(item).split(",") if part.strip()]
    return out or ["0"]


def _handle_stop_signal(signum, _frame) -> None:
    _STOP_REQUESTED.set()
    raise KeyboardInterrupt(f"received signal {signum}")


def _register_process(process: subprocess.Popen, label: str) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES[process.pid] = (process, label)


def _unregister_process(process: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES.pop(process.pid, None)


def _terminate_process(process: subprocess.Popen, label: str, grace_seconds: float = 15.0) -> None:
    if process.poll() is not None:
        return
    print(f"[stop] terminating {label} pid={process.pid}", flush=True)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass

    if process.poll() is None:
        print(f"[stop] killing {label} pid={process.pid}", flush=True)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            process.kill()
        process.wait(timeout=5.0)


def _terminate_active_processes() -> None:
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE_PROCESSES.values())
    for process, label in processes:
        _terminate_process(process, label)


def _run_subprocess(cmd: list[str], env: dict, label: str) -> None:
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        start_new_session=True,
    )
    _register_process(process, label)
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        _STOP_REQUESTED.set()
        _terminate_process(process, label)
        raise
    finally:
        _unregister_process(process)

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def _run_step(cmd: list[str], env: dict, dry_run: bool, label: str) -> None:
    print(f"[{label}] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    _run_subprocess(cmd, env, label)


def _run_step(cmd: list[str], env: dict, dry_run: bool, label: str) -> None:
    print(f"[{label}] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    _run_subprocess(cmd, env, label)


def _scope_args(args) -> list[str]:
    out = ["--base_dir", args.base_dir, "--value_types", *args.value_types, "--split_groups", *args.split_groups]
    if args.thresholds:
        out.extend(["--thresholds", *args.thresholds])
    return out


def _run_prepare_steps(args, gpus: list[str]) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus[0]
    if not args.skip_align:
        cmd = [
            sys.executable, str(ALIGN_SCRIPT), *_scope_args(args),
            "--data_root", args.data_root,
            "--workers", str(args.align_workers),
            "--identity_threshold", str(args.identity_threshold),
        ]
        if args.limit_rows:
            cmd.extend(["--limit_rows", str(args.limit_rows)])
        if args.overwrite_align:
            cmd.append("--overwrite")
        _run_step(cmd, env, args.dry_run, "align")
    if not args.skip_embeddings:
        cmd = [
            sys.executable, str(CACHE_SCRIPT), *_scope_args(args),
            "--cache_dir", args.cache_dir,
            "--progres_device", "cuda:0",
            "--esm_gpus", "0",
        ]
        if args.overwrite_embeddings:
            cmd.append("--overwrite")
        _run_step(cmd, env, args.dry_run, "embeddings")


def _train_command(job: dict, seed: int, args) -> tuple[list[str], Path]:
    out_dir = Path(job["threshold_dir"]) / "catpred_results" / args.run_name / f"seed_{seed}"
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--train_csv", job["train_path"],
        "--val_csv", job["val_path"],
        "--test_csv", job["test_path"],
        "--out_dir", str(out_dir),
        "--task_name", job["value_type"],
        "--seed", str(seed),
        "--num_workers", str(args.num_workers),
        "--device", "cuda:0",
        "--cache_dir", args.cache_dir,
        "--mixed_precision", args.mixed_precision,
        "--resume_if_complete",
        "--require_cached_esm",
    ]
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.ensemble_size is not None:
        cmd.extend(["--ensemble_size", str(args.ensemble_size)])
    if ENZYME_FEATURES[job["value_type"]] == "esm":
        cmd.append("--add_esm_feats")
    else:
        cmd.extend(["--add_pretrained_egnn_feats", "--pretrained_egnn_feats_path", str(Path(args.cache_dir) / "progres" / "progres_egnn_by_structure_id.pt")])
    return cmd, out_dir


def _run_train_job(item) -> dict:
    cmd, out_dir, gpu, env, gpu_semaphore = item
    if _STOP_REQUESTED.is_set():
        return {"out_dir": str(out_dir), "gpu": gpu, "status": "stopped_before_start"}
    marker = out_dir / "final_results_test.csv"
    if marker.exists():
        return {"out_dir": str(out_dir), "gpu": gpu, "status": "skipped_complete"}
    env = dict(env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    label = f"train gpu={gpu} out={out_dir}"
    gpu_semaphore.acquire()
    try:
        _run_subprocess(cmd, env, label)
    except subprocess.CalledProcessError as exc:
        returncode = int(exc.returncode)
        result = {
            "out_dir": str(out_dir),
            "gpu": gpu,
            "status": "failed",
            "returncode": returncode,
            "cmd": " ".join(str(part) for part in cmd),
        }
        if returncode < 0:
            result["signal"] = -returncode
            try:
                result["signal_name"] = signal.Signals(-returncode).name
            except ValueError:
                result["signal_name"] = f"SIG{returncode * -1}"
        return result
    finally:
        gpu_semaphore.release()
    return {"out_dir": str(out_dir), "gpu": gpu, "status": "completed"}


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_stop_signal)

    parser = argparse.ArgumentParser(description="Run CatPred with its original settings for every seed, split and kinetic task.")
    parser.add_argument("--data_root", default=str(default_data_root()), type=str, help="Read-only kinetic_dataset folder.")
    parser.add_argument("--base_dir", default=str(default_base_dir()), type=str, help="Work tree for staged splits, caches and results.")
    parser.add_argument("--value_types", nargs="+", default=list(DEFAULT_VALUE_TYPES))
    parser.add_argument("--split_groups", nargs="+", default=list(DEFAULT_SPLIT_GROUPS))
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--gpus", nargs="+", default=["0"], help="Physical GPU ids, for example --gpus 0 1 or --gpus 0,1")
    parser.add_argument("--max_parallel_per_gpu", default=1, type=int)
    parser.add_argument("--seeds", nargs="+", default=[666, 777, 888], type=int)
    parser.add_argument("--cache_dir", default=None, type=str, help="Default: <base_dir>/embeddings.")
    parser.add_argument("--run_name", default="original_defaults", type=str)
    parser.add_argument("--epochs", default=None, type=int, help="Smoke tests only. Default is the original 30.")
    parser.add_argument("--ensemble_size", default=None, type=int, help="Smoke tests only. Default is the original 10.")
    parser.add_argument("--limit_rows", default=None, type=int, help="Smoke tests only. Stage the first N rows of each split.")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--mixed_precision", choices=["auto", "none", "bf16", "fp16"], default="none")
    parser.add_argument("--skip_align", action="store_true")
    parser.add_argument("--skip_embeddings", action="store_true")
    parser.add_argument("--overwrite_align", action="store_true")
    parser.add_argument("--overwrite_embeddings", action="store_true")
    parser.add_argument("--align_workers", default=max(1, (os.cpu_count() or 8) // 2), type=int)
    parser.add_argument("--identity_threshold", default=90.0, type=float)
    parser.add_argument("--fail_fast", action="store_true", help="Stop all active jobs after the first failed training subprocess.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    gpus = _parse_gpus(args.gpus)
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    args.base_dir = str(Path(args.base_dir).expanduser().resolve())
    args.cache_dir = str(Path(args.cache_dir or Path(args.base_dir) / "embeddings").expanduser().resolve())

    _run_prepare_steps(args, gpus)

    jobs = discover_split_jobs(
        base_dir=args.base_dir,
        value_types=args.value_types,
        split_groups=args.split_groups,
        thresholds=args.thresholds,
    )
    base_env = os.environ.copy()
    per_gpu_limit = max(1, args.max_parallel_per_gpu)
    gpu_semaphores = {gpu: threading.BoundedSemaphore(per_gpu_limit) for gpu in gpus}
    slots = gpus * per_gpu_limit
    train_items = []
    pending_jobs = 0
    for job in jobs:
        for seed in args.seeds:
            cmd, out_dir = _train_command(job, seed, args)
            done = (out_dir / "final_results_test.csv").exists()
            gpu = slots[(len(train_items) if done else pending_jobs) % len(slots)]
            train_items.append((cmd, out_dir, gpu, base_env, gpu_semaphores[gpu]))
            pending_jobs += not done

    print(f"[launch] train_jobs={len(train_items)} pending_jobs={pending_jobs} gpu_slots={slots}", flush=True)
    if args.dry_run:
        for cmd, _out_dir, gpu, _env, _sem in train_items:
            print(f"[gpu {gpu}] {' '.join(cmd)}")
        return

    results = []
    stopped = False
    max_workers = len(slots)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = []
    try:
        futures = [executor.submit(_run_train_job, item) for item in train_items]
        for future in progress(concurrent.futures.as_completed(futures), total=len(futures), desc="Retrain", unit="job"):
            result = future.result()
            results.append(result)
            if result.get("status") == "failed":
                print(
                    "[launch] job failed "
                    f"gpu={result.get('gpu')} out={result.get('out_dir')} "
                    f"returncode={result.get('returncode')} "
                    f"signal={result.get('signal_name', '')}",
                    flush=True,
                )
                if args.fail_fast:
                    stopped = True
                    _STOP_REQUESTED.set()
                    for pending in futures:
                        pending.cancel()
                    _terminate_active_processes()
                    break
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        stopped = True
        _STOP_REQUESTED.set()
        print("\n[launch] stop requested; terminating active training subprocesses...", flush=True)
        for future in futures:
            future.cancel()
        _terminate_active_processes()
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        _STOP_REQUESTED.set()
        _terminate_active_processes()
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    manifest_path = Path(args.base_dir) / "catpred_parallel_retrain_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"interrupted": stopped, "jobs": results}, handle, indent=2, sort_keys=True)
    print(f"[launch] wrote manifest: {manifest_path}", flush=True)
    failed_jobs = sum(1 for result in results if result.get("status") == "failed")
    if failed_jobs:
        print(f"[launch] failed_jobs={failed_jobs}; completed jobs remain resumable", flush=True)
        raise SystemExit(1)
    if stopped:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
