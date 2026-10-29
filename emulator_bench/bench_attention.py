"""Forward plus backward time of the Seq-Attn attention block: original `nn.MultiheadAttention` against the fused kernel. Not used by the pipeline.

Variants run in turn, round after round, so a shared GPU affects them alike. Prints one JSON line per length with the ratios to the original.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from emulator_bench import fused_attention  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lengths", type=int, nargs="+", default=[300, 800, 2047])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--iters", type=int, default=10)
    a = p.parse_args()
    mha = torch.nn.MultiheadAttention(36, 6, batch_first=True).cuda()
    for length in a.lengths:
        x = [torch.randn(a.batch, length, 36, device="cuda", requires_grad=True) for _ in range(3)]
        grad = torch.randn(a.batch, length, 36, device="cuda")
        variants = {"original": lambda: mha(*x)[0], **{f: (lambda f=f: fused_attention.fused_mha(mha, *x, f)) for f in fused_attention.PRECISIONS}}
        for fn in variants.values():
            fn().backward(grad)
        ms = {k: [] for k in variants}
        for _ in range(a.rounds):
            for name, fn in variants.items():
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(a.iters):
                    fn().backward(grad)
                end.record()
                end.synchronize()
                ms[name].append(start.elapsed_time(end) / a.iters)
        med = {k: sorted(v)[len(v) // 2] for k, v in ms.items()}
        print(json.dumps({"length": length, "batch": a.batch, "ms": {k: round(v, 2) for k, v in med.items()},
                          "speedup_vs_original": {k: round(med["original"] / v, 2) for k, v in med.items() if k != "original"},
                          "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}), flush=True)


if __name__ == "__main__":
    main()
