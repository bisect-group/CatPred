"""Paths, split discovery and the amino-acid filter shared by all emulator_bench scripts."""

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EMULATOR_ROOT = REPO_ROOT.parents[1]
MODEL_NAME = "CatPred"
CACHE_VERSION = 4
TASKS = ("kcat", "km")
SEEDS = (666, 777, 888)
SPLIT_GROUPS = (
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "enzyme_structure_splits",
    "substrate_splits",
    "conformer_cosine_splits",
)
SEQUENCE_COL, SMILES_COL, TARGET_COL = "sequence", "smiles", "log10_value"
# Residues that seq_to_tensor in catpred/models/model.py accepts. Any other letter raises KeyError there.
ORIGINAL_AA_VOCAB = frozenset("ACDQKIPTFGHELRWVNYMS")


def ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def default_data_root() -> Path:
    return EMULATOR_ROOT / "data" / "processed" / "datasets" / "kinetic_dataset"


def default_out_root() -> Path:
    return Path(os.getenv("CATPRED_BENCH_OUT_ROOT", EMULATOR_ROOT / "outputs" / "baselines" / MODEL_NAME)).expanduser().resolve()


def esm_cache_dir(out_root: Path) -> Path:
    return Path(out_root) / "embeddings" / f"esm2_t33_650M_UR50D_v{CACHE_VERSION}"


def molgraph_cache_path(out_root: Path, task: str) -> Path:
    return Path(out_root) / "embeddings" / f"molgraph_v{CACHE_VERSION}" / f"{task}.pkl"


@dataclass(frozen=True)
class SplitJob:
    task: str
    split_group: str
    split_name: str  # "all" for a group without threshold folders
    split_dir: Path

    def paths(self) -> dict[str, Path]:
        return {name: self.split_dir / f"{name}.parquet" for name in ("train", "val", "test")}

    def run_dir(self, out_root: Path, seed: int) -> Path:
        return Path(out_root) / self.task / self.split_group / self.split_name / f"seed_{seed}"


def discover_split_jobs(data_root: Path | None = None, tasks=TASKS, split_groups=SPLIT_GROUPS) -> list[SplitJob]:
    """Find every split folder on disk. Thresholds are never hardcoded."""
    root = Path(data_root or default_data_root())
    jobs = []
    for task in tasks:
        for group in split_groups:
            group_dir = root / task / group
            folders = [("all", group_dir)] if (group_dir / "train.parquet").exists() else []
            folders += [(d.name, d) for d in sorted(group_dir.glob("threshold_*")) if (d / "train.parquet").exists()]
            jobs += [SplitJob(task, group, name, folder) for name, folder in folders]
    return jobs


def unknown_residue_mask(sequences) -> "list[bool]":
    """True for sequences with a residue outside the original 20-letter vocabulary."""
    return [not set(str(seq)) <= ORIGINAL_AA_VOCAB for seq in sequences]


def stable_hash(payload, length: int = 16) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:length]


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)
