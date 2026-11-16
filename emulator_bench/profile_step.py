"""Share of GPU kernel time per model part in one fast-path training step (torch.profiler). Not used by the pipeline.

Backward kernels are assigned to the part of their forward op through the autograd sequence number.
Prints one JSON line with the share of each part and the wall time per step.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from emulator_bench import fast_train, reference_run  # noqa: E402,F401
from emulator_bench.common import default_data_root, default_out_root, esm_cache_dir  # noqa: E402
from emulator_bench.train_run import prepare_run  # noqa: E402


def label_module(module, name):
    inner = module.forward

    def wrapped(*a, **k):
        with record_function(name):
            return inner(*a, **k)

    module.forward = wrapped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="kcat")
    p.add_argument("--split_group", default="random_splits_grouped_sequence")
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--egnn_path", default=None)
    p.add_argument("--skip_pad", action="store_true")
    p.add_argument("--fused_attn", action="store_true")
    p.add_argument("--attn_precision", choices=fast_train.PRECISIONS, default="ieee")
    a = p.parse_args()
    opts = fast_train.Options(skip_pad=a.skip_pad, fused_attn=a.fused_attn, attn_precision=a.attn_precision)

    from catpred.models import MoleculeModel
    from catpred.train.loss_functions import get_loss_func
    from catpred.utils import build_lr_scheduler

    torch.set_num_threads(2)
    split_dir = default_data_root() / a.task / a.split_group
    paths = {s: split_dir / f"{s}.parquet" for s in ("train", "val", "test")}
    args, splits, _, _, _, _, inits = prepare_run(paths, Path(a.run_dir), esm_cache_dir(default_out_root()), a.seed, 1, 1, egnn_path=a.egnn_path)
    split = splits["train"]
    rows = next(fast_train.epoch_orders(a.seed, split.n, 1))
    device = args.device
    model = MoleculeModel(args)
    model.load_state_dict(inits[0])
    model = model.to(device)
    for name, mod in (("encoder", model.encoder.encoder[0]), ("seq_embed", model.seq_embedder), ("rotary", model.rotary_embedder),
                      ("attention", model.multihead_attn), ("readout", model.readout)):
        label_module(mod, name)
    label_module(model.attentive_pooler, "pooling")
    pool_fn = fast_train.pool_skip_pad  # does not call the pooler's forward

    def labelled_pool(*x):
        with record_function("pooling"):
            return pool_fn(*x)

    fast_train.pool_skip_pad = labelled_pool
    fused = fast_train.fused_mha

    def labelled_fused(*x):
        with record_function("attention"):
            return fused(*x)

    fast_train.fused_mha = labelled_fused
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}])
    scheduler, counts, loss_func = build_lr_scheduler(optimizer, args), {}, get_loss_func(args)
    batches = fast_train.batches_of(rows[: (a.steps + 15) * args.batch_size], args.batch_size)
    it = iter(fast_train.Prefetcher(split, batches, device))
    for _ in range(15):
        fast_train.train_step(model, next(it), loss_func, optimizer, scheduler, counts, opts)
    torch.cuda.synchronize()

    start = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(a.steps):
            db = next(it)
            optimizer.zero_grad()
            preds = fast_train.forward(model, db, opts)
            with record_function("loss"):
                loss = loss_func(preds.float(), db.targets).sum() / db.n
            loss.backward()
            with record_function("optimizer"):
                optimizer.step()
            scheduler.step()
        torch.cuda.synchronize()
    wall = (time.time() - start) / a.steps

    events = prof.events()
    parts = {"encoder", "seq_embed", "rotary", "attention", "pooling", "readout", "loss", "optimizer"}

    def part_of(e):
        while e is not None:
            if e.name in parts:
                return e.name
            e = e.cpu_parent
        return None

    by_seq = {}
    for e in events:
        if e.sequence_nr >= 0 and not e.name.endswith("Backward0") and "Backward" not in e.name and part_of(e):
            by_seq.setdefault(e.sequence_nr, part_of(e))
    total, share = 0.0, defaultdict(float)
    for e in events:
        k = sum(x.duration for x in e.kernels)
        if not k:
            continue
        label = part_of(e)
        if label is None:
            parent = e
            while parent is not None and label is None:
                label = by_seq.get(parent.sequence_nr) if parent.sequence_nr >= 0 else None
                parent = parent.cpu_parent
        label = label or "other(data, copies, backward without label)"
        share[label] += k
        total += k
    result = {k: round(100 * v / total, 1) for k, v in sorted(share.items(), key=lambda kv: -kv[1])}
    print(json.dumps({"kernel_ms_per_step": round(total / 1e3 / a.steps, 2), "wall_ms_per_step": round(wall * 1e3, 2),
                      "skip_pad": a.skip_pad, "fused_attn": a.fused_attn, "attn_precision": a.attn_precision, "percent_of_kernel_time": result}))


if __name__ == "__main__":
    main()
