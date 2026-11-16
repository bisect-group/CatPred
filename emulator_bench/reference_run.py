"""Run the original catpred training code (catpred.train.run_training) on EMULaToR split tables.

This is the reference for the equivalence tests and the speed benchmarks. Nothing in catpred is edited.
ESM2 vectors come from an EsmCache instead of the original per-row ESM call: `_load_protein_records` is
replaced in memory so that every protein record already holds its `esm2_feats` tensor.
`Recorder` captures init weights, batch order, per-step losses and weights, validation scores and test predictions.
"""

import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from emulator_bench.common import SEQUENCE_COL, SMILES_COL, TARGET_COL, ensure_repo_on_path
from emulator_bench.esm_cache import sequence_key

ensure_repo_on_path()
try:
    import tensorboardX  # noqa: F401
except ImportError:  # the original training code imports tensorboardX; torch ships an equivalent writer
    from torch.utils.tensorboard import SummaryWriter

    sys.modules["tensorboardX"] = types.SimpleNamespace(SummaryWriter=SummaryWriter)


class Recorder:
    """Patches catpred so that one run_training call records what happens. Use as a context manager."""

    def __init__(self, params_after_step: bool = True):
        self.params_after_step = params_after_step
        self.inits, self.orders, self.losses, self.params = [], [], [], []
        self.val_scores, self.checkpoints, self.test_preds = [], [], []

    def get_loss_func(self, original):
        def make(args):
            func = original(args)

            def wrapped(*a, **k):
                out = func(*a, **k)
                self.losses.append(out.detach().clone())
                return out

            return wrapped

        return make

    def __enter__(self):
        import catpred.data.data as data_module
        import catpred.models.model as model_module
        import catpred.train.loss_functions as loss_module
        import catpred.train.run_training as rt

        self._saved = []

        def patch(obj, name, value):
            self._saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        recorder = self
        init = model_module.MoleculeModel.__init__

        def recording_init(model, *a, **k):
            init(model, *a, **k)
            recorder.inits.append({n: v.detach().clone() for n, v in model.state_dict().items()})

        patch(model_module.MoleculeModel, "__init__", recording_init)
        sampler_iter = data_module.MoleculeSampler.__iter__

        def recording_iter(sampler):
            indices = list(sampler_iter(sampler))
            if sampler.shuffle:
                recorder.orders.append(indices)
            return iter(indices)

        patch(data_module.MoleculeSampler, "__iter__", recording_iter)
        patch(loss_module, "get_loss_func", self.get_loss_func(loss_module.get_loss_func))
        patch(rt, "get_loss_func", loss_module.get_loss_func)
        evaluate, predict, save = rt.evaluate, rt.predict, rt.save_checkpoint

        def rec_evaluate(*a, **k):
            scores = evaluate(*a, **k)
            recorder.val_scores.append(scores["rmse"][0])
            return scores

        def rec_predict(*a, **k):
            preds = predict(*a, **k)
            recorder.test_preds.append(np.array(preds)[:, 0])
            return preds

        def rec_save(*a, **k):
            recorder.checkpoints.append(len(recorder.val_scores))
            return save(*a, **k)

        patch(rt, "evaluate", rec_evaluate)
        patch(rt, "predict", rec_predict)
        patch(rt, "save_checkpoint", rec_save)
        if self.params_after_step:
            self._hook = register_optimizer_step_post_hook(self._after_step)
        return self

    def _after_step(self, optimizer, args, kwargs):
        self.params.append(torch.cat([p.detach().flatten().clone() for g in optimizer.param_groups for p in g["params"]]))

    def __exit__(self, *exc):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)
        if self.params_after_step:
            self._hook.remove()


def protein_name(sequence: str) -> str:
    """The protein record name. It is the key of the ProGRES dict that the model looks up."""
    return sequence_key(sequence)


def write_original_inputs(tables: dict, out_dir: Path) -> dict:
    """CSVs in the layout the original scripts read: smiles, target, pdbpath and sequence columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, df in tables.items():
        frame = pd.DataFrame({SMILES_COL: df[SMILES_COL], TARGET_COL: df[TARGET_COL], SEQUENCE_COL: df[SEQUENCE_COL],
                              "pdbpath": [f"/records/{protein_name(s)}.pdb" for s in df[SEQUENCE_COL]]})
        paths[split] = out_dir / f"{split}.csv"
        frame.to_csv(paths[split], index=False)
    return paths


def run_original(tables: dict, esm_cache, out_dir: Path, seed: int, epochs: int, ensemble_size: int, cpu: bool = True,
                 record_params: bool = True, egnn_path=None) -> Recorder:
    """Train with catpred's own get_data and run_training. Returns the Recorder of the run.
    With egnn_path, the original `--add_pretrained_egnn_feats` adds the ProGRES vectors of that dict."""
    import catpred.data.utils as data_utils
    from catpred.args import TrainArgs
    from catpred.data import get_data, validate_dataset_type
    from catpred.features import reset_featurization_parameters, set_adding_hs, set_explicit_h, set_keeping_atom_map
    from catpred.train.run_training import run_training

    # run_training reloads its own checkpoints from out_dir with torch.load.
    trusted = [str(out_dir.resolve()), os.environ.get("CATPRED_TRUSTED_DESERIALIZATION_ROOTS", "")]
    if egnn_path:
        trusted.append(str(Path(egnn_path).resolve().parent))
    os.environ["CATPRED_TRUSTED_DESERIALIZATION_ROOTS"] = os.pathsep.join(trusted)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    paths = write_original_inputs(tables, out_dir / "original_inputs")
    records = {}
    for df in tables.values():
        for seq in df[SEQUENCE_COL]:
            records[protein_name(seq) + ".pdb"] = {"name": protein_name(seq), "seq": seq, "esm2_feats": torch.from_numpy(np.array(esm_cache.get(seq)))}
    argv = [
        "--protein_records_path", "records.json.gz", "--data_path", str(paths["train"]), "--separate_val_path", str(paths["val"]),
        "--separate_test_path", str(paths["test"]), "--dataset_type", "regression", "--smiles_columns", SMILES_COL,
        "--target_columns", TARGET_COL, "--extra_metrics", "mae", "mse", "r2", "--ensemble_size", str(ensemble_size),
        "--seq_embed_dim", "36", "--seq_self_attn_nheads", "6", "--loss_function", "mve", "--batch_size", "16",
        "--save_dir", str(out_dir / "original"), "--epochs", str(epochs), "--add_esm_feats", "--seed", str(seed),
        "--pytorch_seed", str(seed),
    ]
    if egnn_path:
        argv += ["--add_pretrained_egnn_feats", "--pretrained_egnn_feats_path", str(egnn_path)]
    args = TrainArgs().parse_args(argv + (["--no_cuda"] if cpu else ["--gpu", "0"]))
    args.num_folds = 1
    saved = data_utils._load_protein_records
    data_utils._load_protein_records = lambda path: records
    try:
        reset_featurization_parameters()
        set_explicit_h(args.explicit_h)
        set_adding_hs(args.adding_h)
        set_keeping_atom_map(args.keeping_atom_map)
        args.task_names = [TARGET_COL]
        data = get_data(path=args.data_path, protein_records_path=args.protein_records_path, args=args, skip_none_targets=True)
        validate_dataset_type(data, dataset_type=args.dataset_type)
        args.features_size = data.features_size()
        with Recorder(record_params) as recorder:
            run_training(args=args, data=data, logger=None)
    finally:
        data_utils._load_protein_records = saved
    return recorder
