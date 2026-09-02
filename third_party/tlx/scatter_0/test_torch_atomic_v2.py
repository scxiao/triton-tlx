# KERNEL CALLS: 1

import triton
import triton.language as tl

# from torch._inductor.runtime import triton_helpers, triton_heuristics
# from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
# from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
# triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
# from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton.jit
def triton_poi_fused__to_copy_index_add_new_zeros_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 104986560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 96
    x2 = xindex
    x0 = (xindex % 96)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 501, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 501)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 501")
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp8 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp8)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 501)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 501")
    tmp14 = tl.full([XBLOCK], 6048, tl.int32)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp13 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp13)
    tl.device_assert(((0 <= tmp17) & (tmp17 < 6048)) | ~(xmask), "index out of bounds: 0 <= tmp17 < 6048")
    tl.atomic_add(out_ptr0 + (x0 + 96*tmp4), tmp7, xmask, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x0 + 96*tmp11), tmp7, xmask, sem='relaxed')
    tl.atomic_add(out_ptr2 + (x0 + 96*tmp17), tmp7, xmask, sem='relaxed')


@triton.jit
def triton_poi_fused__to_copy_index_add_new_zeros_4_v1(
    in_ptr0, 
    in_ptr1, 
    in_ptr2, 
    in_ptr3, 
    out_ptr0, 
    out_ptr1, 
    out_ptr2, 
    xnumel, 
    CHUNK : tl.constexpr,
    D: tl.constexpr,
    high_bound : tl.constexpr):

    pid = tl.program_id(0)
    chunk_start = pid * CHUNK
    tok_idx = chunk_start + tl.arange(0, CHUNK)
    tok_mask = tok_idx < xnumel

    tmp0 = tl.load(in_ptr0 + tok_idx, tok_mask, other=0.0)
    tmp1 = tl.full([CHUNK], 501, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 501)) | ~(tok_mask), "index out of bounds: 0 <= tmp4 < 501")

    d_offs = tl.arange(0, D)
    data_offs = tok_idx[:, None] * D + d_offs[None,:]
    tmp6 = tl.load(in_ptr1 + data_offs, tok_mask[:, None], other=0.0)
    tmp7 = tmp6.to(tl.float32)
    
    for v in tl.static_range(high_bound):
        row_mask = (tmp4 == v)[:, None]
        row_sum = tl.sum(tl.where(row_mask, tmp7, tl.zeros([CHUNK, D], tl.float32)), axis=0)
        tl.atomic_add(out_ptr0 + (v * D + tl.arange(0, D)), row_sum, sem='relaxed')
    
 
def rand_int_strided(shape, strides, *, low: int, high: int, device="cuda:0", dtype=torch.int64):
    """
    Like torch._dynamo.testing.rand_strided but for integer tensors:
    - creates a flat storage big enough for the given strided view
    - fills it with randint(low, high)
    - returns an as_strided view
    """
    # compute how big the backing storage must be
    storage_size = 1
    for dim, stride in zip(shape, strides):
        if dim == 0:
            continue
        storage_size = max(storage_size, (dim - 1) * stride + 1)

    storage = torch.randint(low, high, (storage_size,), device=device, dtype=dtype)
    return torch.as_strided(storage, size=shape, stride=strides)

high_bound = 32
def get_args():
    torch.manual_seed(1)
    arg_0 = rand_int_strided((1093610,), (1,), low=0, high=high_bound, device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((1093610, 96), (96, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_int_strided((1093610,), (1,), low=0, high=high_bound, device='cuda:0', dtype=torch.int64)
    arg_3 = rand_int_strided((1093610,), (1,), low=0, high=high_bound, device='cuda:0', dtype=torch.int64)
    arg_4 = rand_strided((501, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((501, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((6048, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 104986560




if __name__ == '__main__':
    from triton.testing import do_bench
    import torch
    import os

    # Configuration
    xnumel = 104986560
    XBLOCK = 64
    grid_x = (xnumel + XBLOCK - 1) // XBLOCK
    grid = (grid_x, 1, 1)
    kwargs = {'num_warps': 1, 'num_stages': 1}
    num_gb = 0.23892696

    # Get kernel arguments (convert to list for mutability in restore operations)
    kernel_args = list(get_args())
    added_args = {'XBLOCK': 128}

    # Initialize CUDA device
    cuda_device = 0

    # Initialize CUDA context
    torch.cuda.set_device(cuda_device)
    torch.cuda.init()  # Explicitly initialize CUDA

    # Create launch lambda
    def kernel_launch():
        with torch.cuda._DeviceGuard(cuda_device):
            torch.cuda.set_device(cuda_device)
            # For kernels with constexpr params in signature, append them as positional args
            constexpr_positional_args = []
            for param_name in added_args:
                constexpr_positional_args.append(added_args[param_name])

            return triton_poi_fused__to_copy_index_add_new_zeros_4[grid](*kernel_args, *constexpr_positional_args, **kwargs)

    # Run benchmark
    print(f"Kernel: triton_poi_fused__to_copy_index_add_new_zeros_4")
    print(f"Grid: {grid}")
    print()

    # Measure performance
    ms = do_bench(kernel_launch, warmup=25, rep=100)
    print(f"Median time: {ms:.3f}ms")
    gb_per_s = 0.23892696 / (ms / 1000.0)
    print(f"Throughput: {gb_per_s:.1f}GB/s")

    added_args_v1 = {'CHUNK': 64, 'D': 96, 'high_bound' : high_bound}
    kernel_args_v1 = kernel_args
    # kernel_args_v1.append(1093610)
    grid_v1 = (501,)
    kwargs_v1 = {'num_warps': 8, 'num_stages': 1}
    num_gb = 0.23892696
    def kernel_launch_v1():
        with torch.cuda._DeviceGuard(cuda_device):
            torch.cuda.set_device(cuda_device)
            # For kernels with constexpr params in signature, append them as positional args
            constexpr_positional_args_v1 = []
            for param_name in added_args_v1:
                constexpr_positional_args_v1.append(added_args_v1[param_name])

            return triton_poi_fused__to_copy_index_add_new_zeros_4_v1[grid_v1](*kernel_args_v1, *constexpr_positional_args_v1, **kwargs_v1)

    ms_v1 = do_bench(kernel_launch_v1, warmup=25, rep=100)
    print(f"\nImproved Version_1")
    print(f"Median time: {ms_v1:.3f}ms")
    gb_per_s_v1 = 0.23892696 / (ms_v1 / 1000.0)
    print(f"Throughput: {gb_per_s_v1:.1f}GB/s")
