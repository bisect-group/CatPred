"""Make the Meta fair-esm package importable when the environment ships another `esm` package.

The mldb environment holds the Biohub `esm` package. CatPred needs fair-esm 2.0.0.
`setup_fair_esm` unpacks the pinned wheel into a folder under the output root. It never touches the environment.
`CATPRED_FAIR_ESM_PATH` selects another folder.
"""

import hashlib
import importlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FAIR_ESM_VERSION = "2.0.0"
FAIR_ESM_SHA256 = "7dcb58828a401155d1457d071a03800bd1a50331ddf24ac79ad1dd0f2ea682a2"
FAIR_ESM_INFO_URL = f"https://pypi.org/pypi/fair-esm/{FAIR_ESM_VERSION}/json"


def _vendor_path() -> Path:
    explicit = os.getenv("CATPRED_FAIR_ESM_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()
    out_root = os.getenv("CATPRED_BENCH_OUT_ROOT") or REPO_ROOT.parents[1] / "outputs" / "baselines" / "CatPred"
    return Path(out_root).expanduser().resolve() / "embeddings" / "fair_esm"


def _is_fair_esm(module) -> bool:
    return hasattr(getattr(module, "pretrained", None), "esm2_t33_650M_UR50D")


def _download_wheel(vendor: Path) -> None:
    with urllib.request.urlopen(FAIR_ESM_INFO_URL) as response:
        info = json.load(response)
    url = next(f["url"] for f in info["urls"] if f["filename"].endswith(".whl"))
    with urllib.request.urlopen(url) as response:
        wheel = response.read()
    if hashlib.sha256(wheel).hexdigest() != FAIR_ESM_SHA256:
        raise RuntimeError(f"fair-esm wheel from {url} has an unexpected sha256")
    vendor.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(wheel)).extractall(vendor)


def ensure_fair_esm(download: bool = False):
    """Import fair-esm as `esm`. With download=True, fetch the pinned wheel when the folder is empty."""
    existing = sys.modules.get("esm")
    if existing is not None and _is_fair_esm(existing):
        return existing
    vendor = _vendor_path()
    if not (vendor / "esm" / "pretrained.py").exists():
        if not download:
            raise ImportError(
                f"fair-esm {FAIR_ESM_VERSION} not found in {vendor}. "
                "Run emulator_bench/cache_embeddings.py once; it downloads the pinned wheel there."
            )
        _download_wheel(vendor)
    for name in [m for m in sys.modules if m == "esm" or m.startswith("esm.")]:
        del sys.modules[name]
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    module = importlib.import_module("esm")
    if not _is_fair_esm(module):
        raise ImportError(f"The package in {vendor} does not expose esm.pretrained.esm2_t33_650M_UR50D.")
    return module
