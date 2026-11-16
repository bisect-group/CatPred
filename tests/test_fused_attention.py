"""The fused attention kernel against the original `nn.MultiheadAttention(36, 6)` (called as `mha(q, k, v)`, no mask).

The reference runs in float64 on the GPU. The ieee kernel (fp32) must be as close to it as the float32 original is, up to a small factor.
The tf32 and bf16 kernels must stay inside a precision-appropriate bound. Weights and biases are random, so the biases count.
"""

import pytest
import torch

from emulator_bench import fast_train
from emulator_bench.fused_attention import fused_mha

LENGTHS = [1, 2, 7, 33, 64, 129, 500, 2047]
BOUND = {"ieee": 2e-5, "tf32": 1e-2, "bf16": 5e-2}  # max error relative to the largest reference value, outputs and every gradient


def make_case(length, seed=0, batch=4):
    g = torch.Generator().manual_seed(seed)
    mha = torch.nn.MultiheadAttention(36, 6, batch_first=True)
    with torch.no_grad():
        for p in mha.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.4)
    x = [torch.randn(batch, length, 36, generator=g) for _ in range(3)]  # q, k and v differ
    return mha, x


def outputs_and_grads(mha, x, dtype, precision=None, grad_out=None):
    """Forward output and gradients of the three inputs and all parameters. precision=None runs the original module."""
    model = type(mha)(36, 6, batch_first=True)
    model.load_state_dict(mha.state_dict())
    model = model.to("cuda", dtype)
    inputs = [t.to("cuda", dtype).requires_grad_(True) for t in x]
    out = model(*inputs)[0] if precision is None else fused_mha(model, *inputs, precision)
    out.backward(grad_out.to("cuda", dtype))
    return [out.detach(), *(t.grad for t in inputs), *(p.grad for p in model.parameters())]


def relative_errors(got, ref):
    """Max error of each tensor over its largest reference value. The floor skips tensors that are zero in exact maths (L=1: no gradient into q and k)."""
    floor = 1e-3 * max(b.abs().max().item() for b in ref)
    return [((a.double() - b).abs().max() / b.abs().max().clamp_min(floor)).item() for a, b in zip(got, ref)]


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["ieee", "tf32", "bf16"])
@pytest.mark.parametrize("length", LENGTHS)
def test_kernel_matches_original_forward_and_gradients(length, precision):
    mha, x = make_case(length)
    grad_out = torch.randn(4, length, 36, generator=torch.Generator().manual_seed(1))
    ref = outputs_and_grads(mha, x, torch.float64, None, grad_out)
    got = relative_errors(outputs_and_grads(mha, x, torch.float32, precision, grad_out), ref)
    base = relative_errors(outputs_and_grads(mha, x, torch.float32, None, grad_out), ref)
    if length == 1:  # the gradient into q and k is zero in exact maths. The kernel leaves rounding noise (4e-7 of the largest gradient for ieee)
        got = got[:1] + got[3:]
    print(f"\n[fused attn {precision} L={length}] max error {max(got):.1e}, float32 original {max(base):.1e}")
    assert max(got) < BOUND[precision]
    if precision == "ieee":
        assert max(got) < 10 * max(base) + 1e-7


@pytest.mark.gpu
def test_kernel_is_deterministic():
    mha, x = make_case(300)
    grad_out = torch.randn(4, 300, 36)
    a, b = (outputs_and_grads(mha, x, torch.float32, "ieee", grad_out) for _ in range(2))
    assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_fused_attn_on_cpu_raises():
    mha, x = make_case(5)
    with pytest.raises(RuntimeError, match="CUDA"):
        fused_mha(mha, *x)


def test_attention_keys_enter_the_settings_only_with_the_flag():
    assert "fused_attn" not in fast_train.settings_options(fast_train.Options())
    assert fast_train.settings_options(fast_train.Options(fused_attn=True, attn_precision="bf16"))["attn_precision"] == "bf16"


@pytest.mark.gpu
def test_compiled_pooling_matches_eager_for_changing_shapes():
    from torch._dynamo.utils import counters

    graphs = counters["stats"]["unique_graphs"]
    g = torch.Generator().manual_seed(0)
    pooler = torch.nn.Sequential()  # holds the weights of linear1 (1316 -> 1316) and linear2 (1316 -> 1)
    pooler.linear1, pooler.linear2 = torch.nn.Linear(1316, 1316), torch.nn.Linear(1316, 1)
    pooler = pooler.cuda()
    for batch, length in [(4, 30), (3, 77), (5, 128)]:
        lengths = torch.randint(1, length + 1, (batch,), generator=g)
        mask = torch.arange(length)[None] < lengths[:, None]
        esm = (torch.randn(batch, length, 1280, generator=g) * mask[..., None]).cuda()
        seq = torch.randn(batch, length, 36, generator=g).cuda()
        db = type("DB", (), {"esm": esm, "esm_rows": esm[mask.cuda()], "esm_index": mask.flatten().nonzero().flatten().cuda()})
        with torch.no_grad():
            eager = fast_train.pool_skip_pad(pooler, db, seq)
            compiled = fast_train.pool_skip_pad(pooler, db, seq, use_compile=True)
            from catpred.models.model import AttentivePooling

            original = AttentivePooling(1316, 1316).cuda()
            original.load_state_dict({k: v for k, v in pooler.state_dict().items()})
            reference = original(torch.cat([esm, seq], dim=-1))[0]
        assert torch.allclose(eager, reference, rtol=1e-4, atol=1e-5)
        assert torch.allclose(compiled, reference, rtol=1e-4, atol=1e-5)
    assert counters["stats"]["unique_graphs"] - graphs <= 1, "dynamic=True must not compile once per shape"
