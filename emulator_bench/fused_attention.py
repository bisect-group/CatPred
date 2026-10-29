"""Fused attention core (FlashAttention style, Triton) for the Seq-Attn block. GPU only.

Same maths as `nn.MultiheadAttention` called as `mha(q, k, v)` with no mask, no dropout and the default `need_weights=True`
(the returned weights are not used): in-projection, softmax(q k^T / sqrt(head_dim)) over every position (padded ones too), out-projection.
Only the attention core is a kernel. The L x L matrix is never stored. The forward keeps the row log-sum-exp. The backward recomputes the
probabilities (dK and dV in one kernel, dQ in another, so there are no atomic adds and the result is deterministic).
The head dimension is zero-padded to 16, the smallest size of `tl.dot`. Zero columns add nothing to the dot products, so the maths is exact.

`precision` sets the dot products: `ieee` (fp32, no TF32), `tf32`, or `bf16` (tiles cast to bf16, fp32 accumulation).
Softmax, running max and sum, exp and all outputs stay fp32.
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HEAD_DIM = 16  # padded head dimension
HD = tl.constexpr(HEAD_DIM)
PRECISIONS = ("ieee", "tf32", "bf16")
BLOCK, WARPS = 64, 4


@triton.jit
def _dot(a, b, PREC: tl.constexpr):
    if PREC == 2:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    elif PREC == 1:
        return tl.dot(a, b, input_precision="tf32")
    else:
        return tl.dot(a, b, input_precision="ieee")


@triton.jit(do_not_specialize=["L"])
def _fwd(Q, K, V, OUT, LSE, L, PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid, bh = tl.program_id(0), tl.program_id(1)
    base = bh.to(tl.int64) * L * HD
    rm = pid * BM + tl.arange(0, BM)
    rd = tl.arange(0, HD)
    q = tl.load(Q + base + rm[:, None] * HD + rd[None, :], mask=rm[:, None] < L, other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)
    for start in range(0, L, BN):
        rn = start + tl.arange(0, BN)
        k = tl.load(K + base + rn[:, None] * HD + rd[None, :], mask=rn[:, None] < L, other=0.0)
        v = tl.load(V + base + rn[:, None] * HD + rd[None, :], mask=rn[:, None] < L, other=0.0)
        s = tl.where(rn[None, :] < L, _dot(q, tl.trans(k), PREC), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + _dot(p, v, PREC)
        m_i = m_new
    tl.store(OUT + base + rm[:, None] * HD + rd[None, :], acc / l_i[:, None], mask=rm[:, None] < L)
    tl.store(LSE + bh.to(tl.int64) * L + rm, m_i + tl.log(l_i), mask=rm < L)


@triton.jit(do_not_specialize=["L"])
def _bwd_dkdv(Q, K, V, DO, LSE, DELTA, DK, DV, L, PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid, bh = tl.program_id(0), tl.program_id(1)
    base = bh.to(tl.int64) * L * HD
    rn = pid * BN + tl.arange(0, BN)
    rd = tl.arange(0, HD)
    k = tl.load(K + base + rn[:, None] * HD + rd[None, :], mask=rn[:, None] < L, other=0.0)
    v = tl.load(V + base + rn[:, None] * HD + rd[None, :], mask=rn[:, None] < L, other=0.0)
    dk = tl.zeros([BN, HD], tl.float32)
    dv = tl.zeros([BN, HD], tl.float32)
    for start in range(0, L, BM):
        rm = start + tl.arange(0, BM)
        row = rm < L
        q = tl.load(Q + base + rm[:, None] * HD + rd[None, :], mask=row[:, None], other=0.0)
        do = tl.load(DO + base + rm[:, None] * HD + rd[None, :], mask=row[:, None], other=0.0)
        lse = tl.load(LSE + bh.to(tl.int64) * L + rm, mask=row, other=0.0)
        delta = tl.load(DELTA + bh.to(tl.int64) * L + rm, mask=row, other=0.0)
        p = tl.where(row[:, None], tl.exp(_dot(q, tl.trans(k), PREC) - lse[:, None]), 0.0)
        dv += _dot(tl.trans(p), do, PREC)
        ds = p * (_dot(do, tl.trans(v), PREC) - delta[:, None])
        dk += _dot(tl.trans(ds), q, PREC)
    tl.store(DK + base + rn[:, None] * HD + rd[None, :], dk, mask=rn[:, None] < L)
    tl.store(DV + base + rn[:, None] * HD + rd[None, :], dv, mask=rn[:, None] < L)


@triton.jit(do_not_specialize=["L"])
def _bwd_dq(Q, K, V, DO, LSE, DELTA, DQ, L, PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid, bh = tl.program_id(0), tl.program_id(1)
    base = bh.to(tl.int64) * L * HD
    rm = pid * BM + tl.arange(0, BM)
    rd = tl.arange(0, HD)
    row = rm < L
    q = tl.load(Q + base + rm[:, None] * HD + rd[None, :], mask=row[:, None], other=0.0)
    do = tl.load(DO + base + rm[:, None] * HD + rd[None, :], mask=row[:, None], other=0.0)
    lse = tl.load(LSE + bh.to(tl.int64) * L + rm, mask=row, other=0.0)
    delta = tl.load(DELTA + bh.to(tl.int64) * L + rm, mask=row, other=0.0)
    dq = tl.zeros([BM, HD], tl.float32)
    for start in range(0, L, BN):
        rn = start + tl.arange(0, BN)
        col = rn < L
        k = tl.load(K + base + rn[:, None] * HD + rd[None, :], mask=col[:, None], other=0.0)
        v = tl.load(V + base + rn[:, None] * HD + rd[None, :], mask=col[:, None], other=0.0)
        p = tl.where(col[None, :], tl.exp(_dot(q, tl.trans(k), PREC) - lse[:, None]), 0.0)
        ds = p * (_dot(do, tl.trans(v), PREC) - delta[:, None])
        dq += _dot(ds, k, PREC)
    tl.store(DQ + base + rm[:, None] * HD + rd[None, :], dq, mask=row[:, None])


class _Attention(torch.autograd.Function):
    """softmax(q k^T) v for q, k, v of shape (B * H, L, HD), fp32. q is already scaled."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, q, k, v, prec):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        bh, length, _ = q.shape
        out, lse = torch.empty_like(q), torch.empty(bh, length, device=q.device)
        _fwd[(triton.cdiv(length, BLOCK), bh)](q, k, v, out, lse, length, prec, BLOCK, BLOCK, num_warps=WARPS)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.prec = prec
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        do = do.contiguous()
        bh, length, _ = q.shape
        delta = (do * out).sum(-1)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        grid = (triton.cdiv(length, BLOCK), bh)
        args = (q, k, v, do, lse, delta)
        _bwd_dkdv[grid](*args, dk, dv, length, ctx.prec, BLOCK, BLOCK, num_warps=WARPS)
        _bwd_dq[grid](*args, dq, length, ctx.prec, BLOCK, BLOCK, num_warps=WARPS)
        return dq, dk, dv, None


def attention_core(q, k, v, precision: str = "ieee"):
    """q, k, v: (batch, heads, length, head_dim) with q scaled already. Returns (batch, heads, length, head_dim)."""
    if not q.is_cuda:
        raise RuntimeError("--fused_attn needs a CUDA device. The Triton kernel does not run on the CPU.")
    batch, heads, length, dim = q.shape
    pad = lambda x: F.pad(x, (0, HEAD_DIM - dim)).reshape(batch * heads, length, HEAD_DIM)  # noqa: E731
    out = _Attention.apply(pad(q), pad(k), pad(v), PRECISIONS.index(precision))
    return out.view(batch, heads, length, HEAD_DIM)[..., :dim]


def fused_mha(mha: torch.nn.MultiheadAttention, q, k, v, precision: str = "ieee"):
    """`mha(q, k, v)[0]` for a batch-first self-attention block, with the kernel as attention core."""
    batch, length, embed = q.shape
    heads = mha.num_heads
    dim = embed // heads
    w_q, w_k, w_v = mha.in_proj_weight.chunk(3)
    b_q, b_k, b_v = mha.in_proj_bias.chunk(3)

    def split(x):
        return x.view(batch, length, heads, dim).transpose(1, 2)

    scale = math.sqrt(1.0 / dim)  # as in F.multi_head_attention_forward: q is scaled before the product
    out = attention_core(split(F.linear(q, w_q, b_q)) * scale, split(F.linear(k, w_k, b_k)), split(F.linear(v, w_v, b_v)), precision)
    return mha.out_proj(out.transpose(1, 2).reshape(batch, length, embed))
