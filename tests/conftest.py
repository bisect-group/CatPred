import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DATA = Path(__file__).parent / "fixtures" / "kinetic_dataset"
sys.path.insert(0, str(ROOT))

from emulator_bench.common import default_out_root  # noqa: E402

REAL_OUT_ROOT = default_out_root()  # before the work fixture points the environment at a scratch folder


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device, the real caches and the EMULaToR data (run with --gpu)")


@pytest.fixture(scope="session", autouse=True)
def work(tmp_path_factory):
    """Scratch output root. The catpred native ESM cache and the fair-esm folder stay inside it."""
    root = tmp_path_factory.mktemp("catpred")
    os.environ["CATPRED_BENCH_OUT_ROOT"] = str(root)
    os.environ["CATPRED_CACHE_PATH"] = str(root / "native_cache")
    from emulator_bench.fair_esm_shim import ensure_fair_esm

    ensure_fair_esm(download=True)
    return root


@pytest.fixture(scope="session")
def tables():
    base = FIXTURE_DATA / "kcat" / "random_splits_grouped_sequence"
    return {s: pd.read_parquet(base / f"{s}.parquet") for s in ("train", "val", "test")}


@pytest.fixture(scope="session")
def esm_dir(work):
    """ESM cache of the fixture sequences, built by the real cache_embeddings.py on the CPU."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, str(ROOT / "emulator_bench" / "cache_embeddings.py"), "--data_root", str(FIXTURE_DATA), "--out_root", str(work)]
    subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    return work / "embeddings" / "esm2_t33_650M_UR50D_v4"


@pytest.fixture(scope="session")
def esm_cache(esm_dir):
    from emulator_bench.esm_cache import EsmCache

    return EsmCache(esm_dir)
