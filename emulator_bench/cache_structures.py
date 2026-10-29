"""Build the ProGRES vector dict (`--pretrained_egnn_feats_path` of the original code) for every kcat and km sequence.

One structure is chosen per sequence (`structures.py`) and embedded with progres 0.2.7 (128-d, `progres_shim.py`).
The dict key is the sequence key of `esm_cache.py`. A sequence without a structure is left out, so the original model
gives it the mean vector. Vectors are cached in a JSONL file keyed by sequence key and structure file fingerprint, and the build resumes.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from emulator_bench.cache_embeddings import unique_sequences  # noqa: E402
from emulator_bench.common import EMULATOR_ROOT, SEQUENCE_COL, TASKS, default_data_root, default_out_root, write_json  # noqa: E402
from emulator_bench.esm_cache import sequence_key  # noqa: E402
from emulator_bench.structures import SOURCES, StructureResolver, resolve_sequences, sequence_candidates  # noqa: E402


def progres_dir(out_root: Path) -> Path:
    return Path(out_root) / "embeddings" / "progres_embeds"


def dict_path(out_root: Path) -> Path:
    return progres_dir(out_root) / "catpred_progres_embeds_dict.pt"


def file_fingerprint(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def embed_structures(table: pd.DataFrame, cache_path: Path, device: str = "cpu", log=print) -> dict:
    """Vectors (float32 lists) by sequence key for the rows of `table` that have a structure. New ones are appended to the JSONL cache."""
    import progres
    import torch

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if cache_path.exists():
        lines = cache_path.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                if i < len(lines) - 1:
                    raise
                log("[progres] skipped a truncated last cache line")
                break
            cached[(entry["key"], entry["fp"])] = entry["vec"]
    model, vectors, start = None, {}, time.time()
    with open(cache_path, "a") as out:
        for i, row in enumerate(table[table["structure_path"] != ""].itertuples(), 1):
            key, fp = sequence_key(row.sequence), file_fingerprint(row.structure_path)
            if (key, fp) not in cached:
                model = model or progres.load_trained_model(device)
                try:
                    vec = progres.embed_structure(row.structure_path, model=model, device=device).float().cpu().tolist()
                except progres.NoCoordinatesError:
                    log(f"[progres] no CA atoms in {row.structure_path}")
                    continue
                cached[(key, fp)] = vec
                out.write(json.dumps({"key": key, "fp": fp, "vec": vec}) + "\n")
                if i % 500 == 0:
                    out.flush()
                    log(f"[progres] {i}/{len(table)} {time.time() - start:.0f}s")
            vectors[key] = cached[(key, fp)]
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in sorted(vectors.items())}


def coverage_table(table: pd.DataFrame, embedded: dict, sequences_by_task: dict) -> pd.DataFrame:
    """Unique sequences per task by structure source. `none` counts sequences without a vector."""
    source = table.assign(source=[s if sequence_key(q) in embedded else "none" for q, s in zip(table[SEQUENCE_COL], table["source"])])
    source = source.set_index(SEQUENCE_COL)["source"]
    rows = []
    for task, seqs in sequences_by_task.items():
        counts = source.loc[sorted(seqs)].value_counts()
        rows.append({"task": task, "sequences": len(seqs), **{s: int(counts.get(s, 0)) for s in (*SOURCES, "none")}})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_root", default=str(default_data_root()))
    p.add_argument("--out_root", default=str(default_out_root()))
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--structure_root", default=str(EMULATOR_ROOT / "data" / "intermediate"))
    for name, sub in (("mmcif_dir", "mmcif"), ("alphafold_dir", "alphafold"), ("esmfold2_dir", "esmfold2_pdb"), ("esm3_dir", "esm")):
        p.add_argument(f"--{name}", default=None, help=f"Default: <structure_root>/{sub}")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=8, help="Processes that extract chains from mmCIF files.")
    p.add_argument("--limit", type=int, default=None, help="Tests: only the first N sequences.")
    a = p.parse_args()

    out_root = Path(a.out_root).expanduser().resolve()
    os.environ["CATPRED_BENCH_OUT_ROOT"] = str(out_root)
    from emulator_bench.progres_shim import PROGRES_VERSION, WEIGHTS_SHA256, WHEEL_SHA256, ensure_progres

    ensure_progres(download=True)
    root = Path(a.structure_root)
    dirs = {n: Path(getattr(a, n) or root / sub) for n, sub in (("mmcif_dir", "mmcif"), ("alphafold_dir", "alphafold"),
                                                                  ("esmfold2_dir", "esmfold2_pdb"), ("esm3_dir", "esm"))}
    resolver = StructureResolver.from_dirs(*dirs.values(), chains_dir=out_root / "structures" / "chains")
    by_task = {task: set(unique_sequences(Path(a.data_root), [task])[0]) for task in a.tasks}
    sequences = sorted(set().union(*by_task.values()))[: a.limit]
    table = resolve_sequences(resolver, sequence_candidates(Path(a.data_root), a.tasks, sequences), sequences, a.workers)
    out_dir = progres_dir(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors = embed_structures(table, out_dir / "vectors.jsonl", a.device)

    import torch

    tmp = dict_path(out_root).with_name(f".dict.tmp.{os.getpid()}")
    torch.save(vectors, tmp)
    tmp.replace(dict_path(out_root))
    table.assign(key=[sequence_key(s) for s in table[SEQUENCE_COL]]).drop(columns=SEQUENCE_COL).to_csv(out_dir / "structure_table.csv", index=False)
    coverage = coverage_table(table, vectors, {t: s & set(sequences) for t, s in by_task.items()})
    coverage.to_csv(out_dir / "structure_coverage.csv", index=False)
    write_json(out_dir / "manifest.json", {
        "model": f"progres {PROGRES_VERSION} trained_model.pt (Zenodo 13365312)", "wheel_sha256": WHEEL_SHA256, "weights_sha256": WEIGHTS_SHA256,
        "dim": 128, "key": "sha1 of the sequence", "call": "progres.embed_structure(path, model=model)",
        "structure_rule": "experimental mmCIF chain (dataset pdb_chain, best coverage), AlphaFold, ESMFold2, ESM3",
        "vectors": len(vectors), "dirs": {k: str(v) for k, v in dirs.items()}})
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
