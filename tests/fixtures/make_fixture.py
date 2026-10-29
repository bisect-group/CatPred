"""Write the tiny test fixture: 64 train, 16 val and 16 test rows of the kcat random_splits_grouped_sequence split.

Sequences are cut to 12-40 residues so the original ESM2 path runs on a CPU in seconds.
Usage: python make_fixture.py <kinetic_dataset/kcat/random_splits_grouped_sequence>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

split_dir = Path(sys.argv[1])
out = Path(__file__).parent / "kinetic_dataset" / "kcat" / "random_splits_grouped_sequence"
rng = np.random.default_rng(0)
for name, n in (("train", 64), ("val", 16), ("test", 16)):
    df = pd.read_parquet(split_dir / f"{name}.parquet", columns=["smiles", "sequence", "log10_value"])
    df = df[df.sequence.map(lambda s: set(s) <= set("ACDQKIPTFGHELRWVNYMS")) & (df.smiles.str.len() < 60)]
    df = df.sample(n, random_state=0).reset_index(drop=True)
    df["sequence"] = [s[: int(rng.integers(12, 41))] for s in df.sequence]
    df.to_parquet(out / f"{name}.parquet", index=False)
