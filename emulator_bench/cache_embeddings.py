"""Build the ESM2 cache for every unique kcat and km sequence the original model can featurize.

Each sequence goes through the original get_single_esm_repr, one sequence per call, as the original does.
Run one process per GPU with --shard i --num_shards n. The build resumes from the saved shard index.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from emulator_bench.common import (  # noqa: E402
    SEQUENCE_COL,
    TASKS,
    default_data_root,
    default_out_root,
    discover_split_jobs,
    ensure_repo_on_path,
    esm_cache_dir,
    unknown_residue_mask,
    write_json,
)
from emulator_bench.esm_cache import EsmCacheWriter, sequence_key  # noqa: E402


def unique_sequences(data_root: Path, tasks) -> tuple[list[str], int]:
    """Sequences of all discovered split files that pass the vocabulary filter, longest first. Also the count removed."""
    seqs = set()
    for job in discover_split_jobs(data_root, tasks):
        for path in job.paths().values():
            seqs.update(pd.read_parquet(path, columns=[SEQUENCE_COL])[SEQUENCE_COL])
    ordered = sorted(seqs)
    bad = unknown_residue_mask(ordered)
    keep = [s for s, b in zip(ordered, bad) if not b]
    return sorted(keep, key=lambda s: (-len(s), s)), len(ordered) - len(keep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", default=str(default_data_root()))
    parser.add_argument("--out_root", default=str(default_out_root()))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Embed only this many sequences (tests).")
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    os.environ["CATPRED_BENCH_OUT_ROOT"] = str(out_root)
    os.environ.setdefault("CATPRED_CACHE_PATH", str(out_root / "embeddings" / "catpred_native_cache"))
    ensure_repo_on_path()
    from emulator_bench.fair_esm_shim import ensure_fair_esm

    ensure_fair_esm(download=True)
    import torch

    from catpred.data.esm_utils import get_single_esm_repr

    sequences, n_unknown = unique_sequences(Path(args.data_root), args.tasks)
    mine = sequences[args.shard :: args.num_shards][: args.limit]
    writer = EsmCacheWriter(esm_cache_dir(out_root), args.shard)
    todo = [s for s in mine if sequence_key(s) not in writer.done]
    print(f"[esm] shard {args.shard}/{args.num_shards}: {len(mine)} sequences, {len(todo)} to embed, {n_unknown} skipped (unknown residues)", flush=True)
    start = time.time()
    for i, seq in enumerate(todo, 1):
        vectors = get_single_esm_repr(seq).detach().cpu().numpy().astype(np.float32, copy=False)
        writer.add(seq, vectors)
        if i % 500 == 0:
            torch.cuda.empty_cache()
            print(f"[esm] shard {args.shard}: {i}/{len(todo)} {time.time() - start:.0f}s", flush=True)
    writer.close()
    write_json(
        esm_cache_dir(out_root) / "manifest.json",
        {
            "model": "esm2_t33_650M_UR50D",
            "layer": 33,
            "pooling": "none (per residue, cls and eos rows removed)",
            "max_tokens": 2048,
            "dtype": "float32",
            "call": "one sequence per call, catpred.data.esm_utils.get_single_esm_repr",
            "key": "sha1 of the sequence",
        },
    )
    print(f"[esm] done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
