import torch
import triton
import triton.language as tl
import time

N = 2_479_833
V = 24
D = 32
XNUMEL = N * D
XBLOCK = 64


# Original kernel: one atomic_add per (token, dim) element → ~1.24M CTAs each
# writing to 768-element output → massive contention (~600K writers per location).
@triton.jit
def triton_poi_fused__to_copy_embedding_dense_backward_mm_slice_91(
    in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 32
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (184 + x0 + 224 * x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([1], 23, tl.int64)
    tmp3 = tl.minimum(tmp2, tmp1, tl.PropagateNan.ALL)
    tmp4 = tl.full([1], -24, tl.int64)
    tmp5 = tl.maximum(tmp4, tmp3, tl.PropagateNan.ALL)
    tmp6 = (tl.full([XBLOCK], 24, tl.int32)).to(tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp1 >= tmp10
    tmp12 = tl.full([1], 24, tl.int64)
    tmp13 = tmp1 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.full([1], -1, tl.int64)
    tmp16 = tmp1 != tmp15
    tmp17 = tmp14 & tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.full([1], 0.0, tl.float32)
    tmp21 = tl.where(tmp17, tmp19, tmp20)
    tl.atomic_add(out_ptr0 + (x0 + 32 * tmp9), tmp21, xmask, sem='relaxed')


# Optimized kernel: each CTA processes CHUNK tokens, reduces per vocab row in
# registers via tl.static_range + tl.sum, then issues only V*D = 768 atomic_adds
# per CTA.  Atomic contention drops from ~600K writers/location to ~N/CHUNK.
#
# CHUNK=512: N/CHUNK ≈ 4844 CTAs  →  4844 writers/location  (~120× fewer)
# CHUNK=1024: N/CHUNK ≈ 2422 CTAs  →  2422 writers/location  (~250× fewer)
@triton.jit
def triton_poi_fused__to_copy_embedding_dense_backward_mm_slice_91_opt(
    in_ptr0, in_ptr1, out_ptr0, N,
    CHUNK: tl.constexpr,
    D: tl.constexpr = 32,
    V: tl.constexpr = 24,
):
    pid = tl.program_id(0)
    chunk_start = pid * CHUNK

    # Token indices for this CTA: [CHUNK]
    tok_idx = chunk_start + tl.arange(0, CHUNK)
    tok_mask = tok_idx < N

    # Load raw vocab indices (int32): [CHUNK]
    raw = tl.load(in_ptr0 + tok_idx, tok_mask, other=0).to(tl.int64)

    # Validity: index must be in [0, V)  (-1 padding is already excluded by >= 0)
    valid = tok_mask & (raw >= 0) & (raw < V)

    # Clamp to [-V, V-1] then wrap negatives  (matches original clamping logic)
    clamped = tl.minimum(tl.full([CHUNK], V - 1, tl.int64),
                         tl.maximum(tl.full([CHUNK], -V, tl.int64), raw))
    norm_idx = tl.where(clamped < 0, clamped + V, clamped)  # [CHUNK], values in [0, V)

    # Load gradient slice in_ptr1[tok_idx, 184 + d]: [CHUNK, D]
    d_idx = tl.arange(0, D)[None, :]       # [1, D]
    t_idx = tok_idx[:, None]               # [CHUNK, 1]
    grad = tl.load(in_ptr1 + (184 + d_idx + 224 * t_idx),
                   tok_mask[:, None]).to(tl.float32)
    # Zero-out invalid / padding tokens
    grad = tl.where(valid[:, None], grad, tl.zeros([CHUNK, D], tl.float32))

    # Intra-CTA reduction: for each vocab row, sum contributions then atomic_add once.
    # tl.static_range unrolls the V=24 iterations at compile time.
    for v in tl.static_range(V):
        row_mask = (norm_idx == v)[:, None]                                        # [CHUNK, 1]
        row_sum = tl.sum(tl.where(row_mask, grad, tl.zeros([CHUNK, D], tl.float32)), axis=0)  # [D]
        tl.atomic_add(out_ptr0 + (v * D + tl.arange(0, D)), row_sum, sem='relaxed')


def make_hot50(device):
    tail = torch.randint(1, V, (N,), device=device, dtype=torch.int32)
    hot_mask = torch.rand(N, device=device) < 0.50
    return torch.where(hot_mask, 0, tail)


def run():
    device = 'cuda'
    N_local = 2_479_833
    xnumel = N_local * D

    in_ptr0 = make_hot50(device)
    in_ptr1 = torch.randn(N_local, 224, dtype=torch.bfloat16, device=device)

    # ── correctness check ────────────────────────────────────────────────────
    out_ref = torch.zeros(V, D, dtype=torch.float32, device=device)
    out_opt = torch.zeros(V, D, dtype=torch.float32, device=device)

    orig_grid = ((xnumel + XBLOCK - 1) // XBLOCK,)
    triton_poi_fused__to_copy_embedding_dense_backward_mm_slice_91[orig_grid](
        in_ptr0, in_ptr1, out_ref, xnumel, XBLOCK=XBLOCK,
        num_warps=1, num_stages=1,
    )

    CHUNK = 512
    opt_grid = ((N_local + CHUNK - 1) // CHUNK,)
    triton_poi_fused__to_copy_embedding_dense_backward_mm_slice_91_opt[opt_grid](
        in_ptr0, in_ptr1, out_opt, N_local, CHUNK=CHUNK,
        num_warps=4, num_stages=1,
    )
    torch.cuda.synchronize()

    max_err = (out_ref - out_opt).abs().max().item()
    rel_err = max_err / out_ref.abs().max().item()
    print(f"Max abs error: {max_err:.6f}  rel: {rel_err:.2e}")
    assert rel_err < 1e-3, f"Correctness check failed: rel_err={rel_err:.2e}"
    print("Correctness OK")

    # ── benchmark ─────────────────────────────────────────────────────────────
    N_RUNS = 20

    def bench(fn, grid, *args, **kwargs):
        for _ in range(5):       # warmup
            fn[grid](*args, **kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_RUNS):
            fn[grid](*args, **kwargs)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / N_RUNS * 1000

    out_ref.zero_()
    t_orig = bench(
        triton_poi_fused__to_copy_embedding_dense_backward_mm_slice_91,
        orig_grid,
        in_ptr0, in_ptr1, out_ref, xnumel, XBLOCK=XBLOCK,
        num_warps=1, num_stages=1,
    )

    out_opt.zero_()
    t_opt = bench(
        triton_poi_fused__to_copy_embedding_dense_backward_mm_slice_91_opt,
        opt_grid,
        in_ptr0, in_ptr1, out_opt, N_local, CHUNK=CHUNK,
        num_warps=4, num_stages=1,
    )

    print(f"Original : {t_orig:.3f} ms")
    print(f"Optimized: {t_opt:.3f} ms  ({t_orig / t_opt:.1f}x speedup)")


if __name__ == '__main__':
    run()
