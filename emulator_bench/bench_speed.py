"""Step rate of the original training loop and of the fast trainer on a real split. Not used by the pipeline.

`--mode plain` runs catpred.train.train.train (original data loader, forward and logging) on the first batches of epoch 0.
`--mode fast` runs the same batches through fast_train for --duration seconds (start together with --start_at).
`--mode drift` trains --steps steps twice with exact settings and once with the chosen options, from the same init,
and reports the largest loss and weight differences (the exact-vs-exact pair is the GPU noise floor).
Each mode prints one JSON line.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402

from emulator_bench import fast_train, reference_run  # noqa: E402,F401  (reference_run supplies the tensorboardX stand-in)
from emulator_bench.common import default_data_root, default_out_root, esm_cache_dir  # noqa: E402
from emulator_bench.train_run import prepare_run  # noqa: E402


def plain_rate(args, split, rows, steps, warmup):
    """Original loop: protein records hold CPU ESM tensors, graphs are rebuilt for every batch."""
    from catpred.data.data import MoleculeDataLoader, MoleculeDataset, set_cache_graph
    from catpred.models import MoleculeModel
    from catpred.train.loss_functions import get_loss_func
    from catpred.train.train import train
    from catpred.utils import build_lr_scheduler, build_optimizer

    set_cache_graph(False)
    points = []
    for r in rows[: (steps + warmup) * args.batch_size]:
        d = split.dataset._data[r]
        d.protein_record = {"seq": split.sequences[r], "esm2_feats": torch.from_numpy(np.array(split.esm[r]))}
        points.append(d)
    model = MoleculeModel(args).to(args.device)
    optimizer = build_optimizer(model, args)
    scheduler = build_lr_scheduler(optimizer, args)
    loss_func = get_loss_func(args)

    def run(chunk):
        loader = MoleculeDataLoader(MoleculeDataset(chunk), batch_size=args.batch_size, num_workers=0)
        train(model, loader, loss_func, optimizer, scheduler, args, logger=None, writer=None)

    run(points[: warmup * args.batch_size])
    torch.cuda.synchronize()
    start = time.time()
    run(points[warmup * args.batch_size :])
    torch.cuda.synchronize()
    return steps / (time.time() - start)


def fast_rate(args, split, rows, opts, duration, start_at, threads):
    from catpred.models import MoleculeModel
    from catpred.train.loss_functions import get_loss_func
    from catpred.utils import build_lr_scheduler

    device = args.device
    fast_train.configure_runtime(opts)
    model = MoleculeModel(args).to(device)
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}], fused=opts.fused_adam)
    scheduler = build_lr_scheduler(optimizer, args)
    loss_func, counts = get_loss_func(args), {}
    batches = fast_train.batches_of(rows, args.batch_size)
    steps, timing, first_step = 0, False, None
    begin = time.time()
    for db in fast_train.Prefetcher(split, batches, device, threads=threads):
        fast_train.train_step(model, db, loss_func, optimizer, scheduler, counts, opts)
        steps += 1
        if first_step is None:
            torch.cuda.synchronize()
            first_step = time.time() - begin  # includes the compile time with --compile
        if steps == 15:
            torch.cuda.synchronize()
            time.sleep(max(0.0, start_at - time.time()))
            start, steps, timing = time.time(), 0, True
        if timing and time.time() - start > duration:
            break
    torch.cuda.synchronize()
    return steps / (time.time() - start), first_step


def drift(args, split, rows, init, opts, steps):
    from catpred.models import MoleculeModel
    from catpred.train.loss_functions import get_loss_func
    from catpred.utils import build_lr_scheduler

    def run(o):
        fast_train.configure_runtime(o)
        model = MoleculeModel(args)
        model.load_state_dict(init)
        model = model.to(args.device)
        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}], fused=o.fused_adam)
        scheduler, counts, losses = build_lr_scheduler(optimizer, args), {}, []
        loss_func = get_loss_func(args)
        for db in fast_train.Prefetcher(split, fast_train.batches_of(rows[: steps * args.batch_size], args.batch_size), args.device):
            losses.append(fast_train.train_step(model, db, loss_func, optimizer, scheduler, counts, o))
        return torch.stack(losses), torch.cat([p.detach().flatten() for p in model.parameters()])

    (la, wa), (lb, wb), (lc, wc) = run(fast_train.Options()), run(fast_train.Options()), run(opts)
    return {"noise_loss": (la - lb).abs().max().item(), "noise_weights": (wa - wb).abs().max().item(),
            "option_loss": (la - lc).abs().max().item(), "option_weights": (wa - wc).abs().max().item()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["plain", "fast", "drift"], required=True)
    p.add_argument("--task", default="kcat")
    p.add_argument("--split_group", default="random_splits_grouped_sequence")
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--duration", type=float, default=40)
    p.add_argument("--start_at", type=float, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--run_dir", required=True)
    for flag in ("tf32", "amp", "fused_adam", "skip_pad", "fused_attn", "compile"):
        p.add_argument(f"--{flag}", action="store_true")
    p.add_argument("--attn_precision", choices=fast_train.PRECISIONS, default="ieee")
    p.add_argument("--egnn_path", default=None, help="ProGRES dict: adds structure features.")
    a = p.parse_args()
    flags = ("tf32", "amp", "fused_adam", "skip_pad", "fused_attn", "attn_precision", "compile")
    opts = fast_train.Options(a.tf32, a.amp, a.fused_adam, a.skip_pad, fused_attn=a.fused_attn, attn_precision=a.attn_precision, compile=a.compile)

    torch.set_num_threads(2)
    split_dir = default_data_root() / a.task / a.split_group
    paths = {s: split_dir / f"{s}.parquet" for s in ("train", "val", "test")}
    args, splits, _, _, _, _, inits = prepare_run(paths, Path(a.run_dir), esm_cache_dir(default_out_root()), a.seed, 1, 1, egnn_path=a.egnn_path)
    split = splits["train"]
    rows = next(fast_train.epoch_orders(a.seed, split.n, 1))
    torch.cuda.reset_peak_memory_stats()
    if a.mode == "drift":
        print(json.dumps({"mode": "drift", "steps": a.steps, **{k: getattr(a, k) for k in flags},
                          **drift(args, split, rows, inits[0], opts, a.steps)}))
        return
    if a.mode == "plain":
        rate = plain_rate(args, split, rows, a.steps, 10)
    else:
        rate, first_step = fast_rate(args, split, rows, opts, a.duration, a.start_at, a.threads)
        compiles = {"first_step_s": round(first_step, 1), **({"compiles": dict(counters["stats"]), "recompiles": dict(counters["recompiles"])} if a.compile else {})}
    print(json.dumps({"mode": a.mode, "steps_per_s": round(rate, 2), **(compiles if a.mode == "fast" else {}), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                      "reserved_gpu_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
                      **{k: v for k, v in vars(a).items() if k in (*flags, "threads")}}), flush=True)


if __name__ == "__main__":
    main()
