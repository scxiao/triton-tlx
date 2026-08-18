"""
Correctness tests for the optimized embedding-backward scatter-add kernel.

Both GPU kernels are compared against a CPU reference (torch.scatter_add) that
is exact up to float32 precision, giving an independent ground truth.

Test matrix
-----------
- Index distributions: uniform, all-same-row, padding (-1), out-of-range high/low
- Boundary sizes: N < CHUNK, N == CHUNK, N not a multiple of CHUNK
- CHUNK values: 256, 512, 1024
- Large realistic workload (make_hot50 distribution)
"""

import pytest
import torch

from test_torch_atomic import (
    triton_poi_fused__to_copy_embedding_dense_backward_mm_slice_91 as orig_kernel,
    triton_poi_fused__to_copy_embedding_dense_backward_mm_slice_91_opt as opt_kernel,
    make_hot50,
)

V = 24
D = 32
COL_OFFSET = 184
IN_STRIDE = 224
XBLOCK = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def cpu_reference(indices: torch.Tensor, grad_in: torch.Tensor) -> torch.Tensor:
    """Exact CPU reference: torch.scatter_add on valid tokens only.

    Matches original kernel semantics:
      valid  = 0 <= idx < V   (the != -1 check is redundant since -1 < 0)
      output[valid_idx] += grad_in[token, COL_OFFSET : COL_OFFSET+D]
    """
    idx = indices.cpu().to(torch.int64)          # [N]
    grad = grad_in.cpu()[:, COL_OFFSET:COL_OFFSET + D].to(torch.float32)  # [N, D]

    valid = (idx >= 0) & (idx < V)               # [N]
    valid_idx = idx.clamp(0, V - 1)              # safe index for scatter (invalid → row 0, zeroed below)

    # scatter_add: out[valid_idx[i], d] += grad[i, d] for valid tokens
    out = torch.zeros(V, D, dtype=torch.float32)
    out.scatter_add_(0,
                     valid_idx[:, None].expand(-1, D) * valid[:, None].long(),
                     grad * valid[:, None].float())
    # Tokens that scattered to row 0 but were invalid added to row 0; subtract them.
    # It's simpler to just use a masked loop for correctness clarity:
    out = torch.zeros(V, D, dtype=torch.float32)
    for i in range(len(idx)):
        if valid[i]:
            out[valid_idx[i]] += grad[i]
    return out


def run_orig(in_ptr0, in_ptr1, N):
    xnumel = N * D
    out = torch.zeros(V, D, dtype=torch.float32, device=in_ptr0.device)
    grid = ((xnumel + XBLOCK - 1) // XBLOCK,)
    orig_kernel[grid](in_ptr0, in_ptr1, out, xnumel,
                      XBLOCK=XBLOCK, num_warps=1, num_stages=1)
    torch.cuda.synchronize()
    return out


def run_opt(in_ptr0, in_ptr1, N, chunk):
    out = torch.zeros(V, D, dtype=torch.float32, device=in_ptr0.device)
    grid = ((N + chunk - 1) // chunk,)
    opt_kernel[grid](in_ptr0, in_ptr1, out, N,
                     CHUNK=chunk, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    return out


def make_inputs(indices, N, device="cuda"):
    """Build in_ptr0 (int32) and in_ptr1 (bf16 [N, IN_STRIDE]) on device."""
    in_ptr0 = torch.tensor(indices, dtype=torch.int32, device=device)
    in_ptr1 = torch.randn(N, IN_STRIDE, dtype=torch.bfloat16, device=device)
    return in_ptr0, in_ptr1


def check(out_opt, ref_cpu, *, atol, rtol, label=""):
    err = (out_opt.cpu() - ref_cpu).abs()
    scale = ref_cpu.abs().clamp(min=1.0)
    assert (err <= atol + rtol * scale).all(), (
        f"{label} max_abs={err.max():.4e}  "
        f"max_rel={(err / scale).max():.4e}  "
        f"atol={atol}  rtol={rtol}"
    )


# ── index-distribution tests (small N for fast, exact CPU reference) ─────────

@pytest.mark.parametrize("chunk", [256, 512, 1024])
def test_uniform_random(chunk):
    """Uniformly distributed valid indices: every vocab row gets roughly N/V hits."""
    N = 1000
    indices = torch.randint(0, V, (N,)).tolist()
    in_ptr0, in_ptr1 = make_inputs(indices, N)
    ref = cpu_reference(in_ptr0, in_ptr1)
    out = run_opt(in_ptr0, in_ptr1, N, chunk)
    check(out, ref, atol=1e-3, rtol=1e-4, label="uniform_random")


@pytest.mark.parametrize("chunk", [256, 512, 1024])
def test_all_same_row(chunk):
    """All tokens map to row 0 — maximum atomic contention in the original."""
    N = 800
    indices = [0] * N
    in_ptr0, in_ptr1 = make_inputs(indices, N)
    ref = cpu_reference(in_ptr0, in_ptr1)
    out = run_opt(in_ptr0, in_ptr1, N, chunk)
    check(out, ref, atol=1e-2, rtol=1e-4, label="all_same_row")


def test_all_padding():
    """All indices are -1 (padding sentinel) → output must be all zeros."""
    N = 500
    indices = [-1] * N
    in_ptr0, in_ptr1 = make_inputs(indices, N)
    out_orig = run_orig(in_ptr0, in_ptr1, N)
    out_opt  = run_opt(in_ptr0, in_ptr1, N, chunk=512)
    assert out_orig.abs().max() == 0.0, "original: padding should produce all-zero output"
    assert out_opt.abs().max()  == 0.0, "optimized: padding should produce all-zero output"


def test_out_of_range_high():
    """Indices >= V are invalid and must contribute zero."""
    N = 300
    indices = list(range(V, V + N))   # all >= 24, all invalid
    in_ptr0, in_ptr1 = make_inputs(indices, N)
    out_orig = run_orig(in_ptr0, in_ptr1, N)
    out_opt  = run_opt(in_ptr0, in_ptr1, N, chunk=512)
    assert out_orig.abs().max() == 0.0
    assert out_opt.abs().max()  == 0.0


def test_out_of_range_negative():
    """Indices < -V (e.g. -100) are invalid and must contribute zero."""
    N = 300
    indices = [-100] * N
    in_ptr0, in_ptr1 = make_inputs(indices, N)
    out_orig = run_orig(in_ptr0, in_ptr1, N)
    out_opt  = run_opt(in_ptr0, in_ptr1, N, chunk=512)
    assert out_orig.abs().max() == 0.0
    assert out_opt.abs().max()  == 0.0


def test_mixed_valid_and_invalid():
    """Mix of valid [0,V), padding (-1), and out-of-range indices."""
    N = 900
    indices = (
        list(range(V)) * 10           # 240 valid, each row once
        + [-1] * 300                  # padding
        + [V + 5] * 200               # out-of-range high
        + [-50] * 160                 # out-of-range low
    )
    assert len(indices) == N
    in_ptr0, in_ptr1 = make_inputs(indices, N)
    ref = cpu_reference(in_ptr0, in_ptr1)
    out = run_opt(in_ptr0, in_ptr1, N, chunk=512)
    check(out, ref, atol=1e-3, rtol=1e-4, label="mixed_valid_invalid")


# ── boundary size tests ───────────────────────────────────────────────────────

@pytest.mark.parametrize("chunk", [256, 512, 1024])
def test_n_less_than_chunk(chunk):
    """N < CHUNK: the single CTA must handle the tail mask correctly."""
    N = chunk // 4
    indices = torch.randint(0, V, (N,)).tolist()
    in_ptr0, in_ptr1 = make_inputs(indices, N)
    ref = cpu_reference(in_ptr0, in_ptr1)
    out = run_opt(in_ptr0, in_ptr1, N, chunk)
    check(out, ref, atol=1e-3, rtol=1e-4, label=f"n_less_than_chunk(chunk={chunk})")


@pytest.mark.parametrize("chunk", [256, 512, 1024])
def test_n_equals_chunk(chunk):
    """N == CHUNK: exactly one CTA, no tail."""
    N = chunk
    indices = torch.randint(0, V, (N,)).tolist()
    in_ptr0, in_ptr1 = make_inputs(indices, N)
    ref = cpu_reference(in_ptr0, in_ptr1)
    out = run_opt(in_ptr0, in_ptr1, N, chunk)
    check(out, ref, atol=1e-3, rtol=1e-4, label=f"n_equals_chunk(chunk={chunk})")


@pytest.mark.parametrize("chunk", [256, 512, 1024])
def test_n_not_multiple_of_chunk(chunk):
    """N is not a multiple of CHUNK: last CTA uses the tail mask."""
    N = chunk * 3 + chunk // 3 + 7   # deliberately unaligned
    indices = torch.randint(0, V, (N,)).tolist()
    in_ptr0, in_ptr1 = make_inputs(indices, N)
    ref = cpu_reference(in_ptr0, in_ptr1)
    out = run_opt(in_ptr0, in_ptr1, N, chunk)
    check(out, ref, atol=1e-3, rtol=1e-4, label=f"n_not_multiple_of_chunk(chunk={chunk})")


# ── agreement with original kernel (large realistic workload) ─────────────────

@pytest.mark.parametrize("chunk", [256, 512, 1024])
def test_agrees_with_original_large(chunk):
    """Both GPU kernels must agree on the full-scale hot-50 distribution.

    Uses relative tolerance to absorb fp32 accumulation-order differences
    inherent in non-deterministic atomic_add.
    """
    N = 2_479_833
    in_ptr0 = make_hot50("cuda")
    in_ptr1 = torch.randn(N, IN_STRIDE, dtype=torch.bfloat16, device="cuda")

    out_orig = run_orig(in_ptr0, in_ptr1, N)
    out_opt  = run_opt(in_ptr0, in_ptr1, N, chunk)

    # Scale tolerance to the magnitude of the accumulated sum.
    # With ~1.25M tokens hitting a single row the sum magnitude is O(1000),
    # and fp32 accumulation error over that many terms is O(1e-4 * magnitude).
    check(out_opt, out_orig.cpu(), atol=1.0, rtol=1e-3,
          label=f"large_hot50(chunk={chunk})")
