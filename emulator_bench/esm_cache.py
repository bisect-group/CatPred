"""Packed cache of per-residue ESM2 vectors, one entry per unique sequence.

Each vector is what catpred/data/esm_utils.py get_single_esm_repr returns: esm2_t33_650M_UR50D,
layer 33, no pooling, no cls or eos row, fp32, tokens cut to 2048 (so at most 2047 rows).
Rows are stored in raw fp32 shard files. Each shard has a JSON index that holds sequence hash, row offset and row count.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ESM_DIM = 1280
SHARD_FLUSH_EVERY = 200


def sequence_key(sequence: str) -> str:
    return hashlib.sha1(sequence.encode()).hexdigest()


def _read_index(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"keys": [], "offsets": [], "rows": []}


def _write_index(path: Path, index: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(index))
    os.replace(tmp, path)


class EsmCacheWriter:
    """Appends vectors to one shard. Resumes from the last saved index and drops rows after it."""

    def __init__(self, cache_dir: Path, shard: int):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.dir / f"shard_{shard}.f32"
        self.index_path = self.dir / f"shard_{shard}.json"
        self.index = _read_index(self.index_path)
        self.done = set(self.index["keys"])
        self.n_rows = sum(self.index["rows"])
        with open(self.data_path, "ab"):
            pass
        os.truncate(self.data_path, self.n_rows * ESM_DIM * 4)
        self.file = open(self.data_path, "ab")

    def add(self, sequence: str, vectors: np.ndarray) -> None:
        assert vectors.dtype == np.float32 and vectors.shape[1] == ESM_DIM
        self.file.write(np.ascontiguousarray(vectors).tobytes())
        self.index["keys"].append(sequence_key(sequence))
        self.index["offsets"].append(self.n_rows)
        self.index["rows"].append(int(vectors.shape[0]))
        self.n_rows += vectors.shape[0]
        if len(self.index["keys"]) % SHARD_FLUSH_EVERY == 0:
            self.flush()

    def flush(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())
        _write_index(self.index_path, self.index)

    def close(self) -> None:
        self.flush()
        self.file.close()


class EsmCache:
    """Read-only view of all shards. get() returns a memory-mapped (rows, 1280) fp32 array."""

    def __init__(self, cache_dir: Path):
        self.entries = {}
        self.maps = {}
        for index_path in sorted(Path(cache_dir).glob("shard_*.json")):
            index = _read_index(index_path)
            data_path = index_path.with_suffix(".f32")
            n_rows = sum(index["rows"])
            if n_rows:
                self.maps[data_path.name] = np.memmap(data_path, dtype=np.float32, mode="r", shape=(n_rows, ESM_DIM))
            for key, offset, rows in zip(index["keys"], index["offsets"], index["rows"]):
                self.entries[key] = (data_path.name, offset, rows)

    def __contains__(self, sequence: str) -> bool:
        return sequence_key(sequence) in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, sequence: str) -> np.ndarray:
        name, offset, rows = self.entries[sequence_key(sequence)]
        return self.maps[name][offset : offset + rows]
