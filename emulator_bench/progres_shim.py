"""Make the progres package (ProGRES structure embeddings) importable at the version CatPred pins.

The mldb environment holds progres 1.1.0. CatPred pins progres 0.2.7 (`modal_app.py`).
`ensure_progres` unpacks the pinned wheel and downloads the pinned trained weights into a folder under the output root.
It never touches the environment. `CATPRED_PROGRES_PATH` selects another folder.
The trained model of progres 0.2.7 (`trained_model.pt`, Zenodo record 13365312) gives 128-d embeddings.
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
PROGRES_VERSION = "0.2.7"
WHEEL_SHA256 = "5037ce4bf8fb093380b4090daafec1109ca2387ca461488abff9e7cf2fd7cfe2"
WEIGHTS_URL = "https://zenodo.org/record/13365312/files/trained_model.pt"
WEIGHTS_SHA256 = "3fa3de9af77527da3efb8f2ee33ad05e678303d4e9cbe1f25a3916a106e56be3"
WEIGHTS_SUBDIR = Path("data") / "trained_models" / "v_0_2_0"


def vendor_path() -> Path:
    explicit = os.getenv("CATPRED_PROGRES_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()
    out_root = os.getenv("CATPRED_BENCH_OUT_ROOT") or REPO_ROOT.parents[1] / "outputs" / "baselines" / "CatPred"
    return Path(out_root).expanduser().resolve() / "embeddings" / "progres"


def _fetch(url: str, sha256: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f"{url} has an unexpected sha256")
    return data


def _download(vendor: Path) -> None:
    with urllib.request.urlopen(f"https://pypi.org/pypi/progres/{PROGRES_VERSION}/json") as response:
        url = next(f["url"] for f in json.load(response)["urls"] if f["filename"].endswith(".whl"))
    vendor.mkdir(parents=True, exist_ok=True)
    if not (vendor / "progres" / "progres.py").exists():
        zipfile.ZipFile(io.BytesIO(_fetch(url, WHEEL_SHA256))).extractall(vendor)
    weights = vendor / WEIGHTS_SUBDIR / "trained_model.pt"
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        tmp = weights.with_name(f".trained_model.pt.tmp.{os.getpid()}")
        tmp.write_bytes(_fetch(WEIGHTS_URL, WEIGHTS_SHA256))
        tmp.replace(weights)


def ensure_progres(download: bool = False):
    """Import progres 0.2.7 from the vendor folder. With download=True, fetch the pinned wheel and weights when missing."""
    vendor = vendor_path()
    if not ((vendor / "progres" / "progres.py").exists() and (vendor / WEIGHTS_SUBDIR / "trained_model.pt").exists()):
        if not download:
            raise ImportError(f"progres {PROGRES_VERSION} not found in {vendor}. Run emulator_bench/cache_structures.py once.")
        _download(vendor)
    existing = sys.modules.get("progres")
    if existing is not None and Path(existing.__file__).resolve().parent == vendor / "progres":
        return existing
    for name in [m for m in sys.modules if m == "progres" or m.startswith("progres.")]:
        del sys.modules[name]
    os.environ["PROGRES_DATA_DIR"] = str(vendor / "data")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    return importlib.import_module("progres")
