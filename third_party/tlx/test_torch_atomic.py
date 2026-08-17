import torch
import triton
import triton.language as tl
import time
import statistics

N = 2_479_833
V = 24
D = 32
XNUMEL = N * D
XBLOCK = 64

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
    # tl.store(out_ptr0 + (x0 + 32 * tmp9), tmp21, xmask)


def make_hot50(device):
      tail = torch.randint(
          1, V, (N,), device=device, dtype=torch.int32
      )
      hot_mask = torch.rand(N, device=device) < 0.50
      return torch.where(hot_mask, 0, tail)
  
  
def run():
    device = 'cuda'

    # Input shapes from dump:
    # in_ptr0: [2479833], int32
    # in_ptr1: [2479833, 224], bfloat16
    # out_ptr0: [24, 32], float32
    # xnumel: 79354656  (= 2479833 * 32)
    # XBLOCK: 64

    N = 2479833
    xnumel = N * 32  # 79354656

    # in_ptr0 = torch.randint(0, 24, (N,), dtype=torch.int32, device=device)
    in_ptr0 = make_hot50(device)
    in_ptr1 = torch.randn(N, 224, dtype=torch.bfloat16, device=device)
    out_ptr0 = torch.zeros(24, 32, dtype=torch.float32, device=device)

    XBLOCK = 64
    grid = ((xnumel + XBLOCK - 1) // XBLOCK,)

    # Warmup
    for _ in range(3):
        out_ptr0.zero_()
        triton_poi_fused__to_copy_embedding_dense_backward_mm_slice_91[grid](
            in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK=XBLOCK,
            num_warps=1, num_stages=1
        )
    torch.cuda.synchronize()

    # Timed runs
    N_RUNS = 10
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        # out_ptr0.zero_()
        triton_poi_fused__to_copy_embedding_dense_backward_mm_slice_91[grid](
            in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK=XBLOCK,
            num_warps=1, num_stages=1
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / N_RUNS * 1000
    print(f"Kernel time: {avg_ms:.3f} ms")


if __name__ == '__main__':
    run()
