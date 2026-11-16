"""Rebuild the CatPred result tables from the saved predictions of every run.

Reads <out_root>/<task>/<split_group>/<split_name>/seed_<seed>/pred_label_{train,val,test}.csv for every split folder
found in the dataset tree. Writes, inside <out_root>:
  catpred_seed_results.csv      one row per task, split, seed and train/val/test part
  aggregate_catpred_results.csv mean and std over seeds
  catpred_missing_results.csv   runs and parts that are not finished
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from emulator_bench.common import SEEDS, SPLIT_GROUPS, TARGET_COL, TASKS, default_data_root, default_out_root, discover_split_jobs  # noqa: E402

PARTS = ("train", "val", "test")
METRICS = ("rmse", "mae", "mse", "r2", "pcc", "scc", "p1mag")


def metrics_from_predictions(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    pred, label = df[f"{TARGET_COL}_pred"], df[f"{TARGET_COL}_label"]
    err = pred - label
    mse = float(np.mean(err**2))
    ss_tot = float(np.sum((label - label.mean()) ** 2))
    return {
        "rmse": mse**0.5, "mae": float(np.mean(np.abs(err))), "mse": mse, "r2": 1.0 - float(np.sum(err**2)) / ss_tot,
        "pcc": float(pred.corr(label, method="pearson")), "scc": float(pred.corr(label, method="spearman")),
        "p1mag": float(np.mean(np.abs(err) <= 1.0)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_root", default=str(default_data_root()))
    p.add_argument("--out_root", default=str(default_out_root()))
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--split_groups", nargs="+", default=list(SPLIT_GROUPS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    a = p.parse_args()

    out_root = Path(a.out_root).expanduser().resolve()
    rows, missing = [], []
    for job in discover_split_jobs(Path(a.data_root), a.tasks, a.split_groups):
        for seed in a.seeds:
            run_dir = job.run_dir(out_root, seed)
            ident = {"task": job.task, "split_group": job.split_group, "split_name": job.split_name, "seed": seed}
            summary = pd.read_csv(run_dir / "run_summary.csv", dtype={"settings_id": str}).iloc[0].to_dict() if (run_dir / "run_summary.csv").exists() else None
            if summary is None:
                missing.append({**ident, "missing": "run_summary.csv"})
                continue
            dropped = {k: v for k, v in summary.items() if k.startswith("dropped_")}
            for part in PARTS:
                path = run_dir / f"pred_label_{part}.csv"
                if not path.exists():
                    missing.append({**ident, "missing": path.name})
                    continue
                rows.append({**ident, "part": part, **metrics_from_predictions(path), **dropped})

    seeds_df = pd.DataFrame(rows)
    seeds_df.to_csv(out_root / "catpred_seed_results.csv", index=False)
    pd.DataFrame(missing).to_csv(out_root / "catpred_missing_results.csv", index=False)
    if seeds_df.empty:
        print(f"No finished runs under {out_root}; {len(missing)} missing")
        return
    keys = ["task", "split_group", "split_name", "part"]
    agg = seeds_df.groupby(keys)[list(METRICS)].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = seeds_df.groupby(keys)["seed"].nunique()
    agg.reset_index().to_csv(out_root / "aggregate_catpred_results.csv", index=False)
    print(f"{len(seeds_df)} seed rows, {len(agg)} aggregate rows, {len(missing)} missing, written to {out_root}")


if __name__ == "__main__":
    main()
