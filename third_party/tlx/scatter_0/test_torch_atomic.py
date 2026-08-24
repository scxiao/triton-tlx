# KERNEL CALLS: 1

import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

from triton.testing import do_bench
import os

def torch_naive_ref(in_ptr0, in_values, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, val_elem_num):
    row_out0 = out_ptr0.shape[0]
    for i in range(row_out0):
        out_ptr0[i] = in_values[in_ptr0 == i].to(torch.float32).sum(dim=0)
    
    row_out1 = out_ptr1.shape[0]
    for i in range(row_out1):
        out_ptr1[i] = in_values[in_ptr2 == i].to(torch.float32).sum(dim=0)

    row_out2 = out_ptr2.shape[0]
    for i in range(row_out2):
        out_ptr2[i] = in_values[in_ptr3 == i].to(torch.float32).sum(dim=0)


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
def triton_poi_fused__to_copy_index_add_new_zeros_4_v1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, 
                                                       x, y, BLOCK_M : tl.constexpr, BLOCK_N : tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_M
    xindex = xoffset + tl.arange(0, BLOCK_M)
    xmask = xindex < x

    yindex = tl.arange(0, BLOCK_N)
    ymask = yindex < y

    xy_index = xindex[:, None] * 96 + yindex[None, :]
    xy_mask = xmask[:, None] & ymask[None,:]

    tmp0 = tl.load(in_ptr0 + xindex, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + xy_index, xy_mask)

    tmp8 = tl.load(in_ptr2 + xindex, xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + xindex, xmask, eviction_policy='evict_last')
    tmp1 = tl.full([BLOCK_M], 501, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 501)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 501")
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp8 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp8)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 501)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 501")
    tmp14 = tl.full([BLOCK_M], 6048, tl.int32)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp13 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp13)
    tl.device_assert(((0 <= tmp17) & (tmp17 < 6048)) | ~(xmask), "index out of bounds: 0 <= tmp17 < 6048")
    tl.atomic_add(out_ptr0 + (yindex[None,:] + 96*tmp4[:,None]), tmp7, xy_mask, sem='relaxed')
    tl.atomic_add(out_ptr1 + (yindex[None,:] + 96*tmp11[:,None]), tmp7, xy_mask, sem='relaxed')
    tl.atomic_add(out_ptr2 + (yindex[None,:] + 96*tmp17[:,None]), tmp7, xy_mask, sem='relaxed')
    # tl.store(out_ptr0 + (yindex[None,:] + 96*tmp4[:,None]), tmp7, xy_mask)
    # tl.store(out_ptr1 + (yindex[None,:] + 96*tmp11[:,None]), tmp7, xy_mask)
    # tl.store(out_ptr2 + (yindex[None,:] + 96*tmp17[:,None]), tmp7, xy_mask)


@gluon.jit
def gluon_poi_fused__to_copy_index_add_new_zeros_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, 
                                                       x, y, BLOCK_M : tl.constexpr, BLOCK_N : tl.constexpr):
    
    blocked: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[4, 1], order=[1, 0])
    blocked1: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 16], warps_per_cta=[4, 1], order=[1, 0])

    xoffset = ttgl.program_id(axis=0) * BLOCK_M
    xindex_1_blk = xoffset + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, blocked))
    xindex_0_blk = xoffset + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(0, blocked))
    xindex_1_blk1 = xoffset + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, blocked1))
    xindex_0_blk1 = xoffset + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(0, blocked1))

    yindex_1_blk = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, blocked))
    yindex_0_blk = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, blocked))
    yindex_1_blk1 = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, blocked1))
    yindex_0_blk1 = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, blocked1))

    xmask_blk = xindex_1_blk < x
    xmask_blk1 = xindex_1_blk1 < x

    ymask_blk = yindex_0_blk < y
    ymask_blk1 = yindex_0_blk1 < y

    xy_offsets = xindex_1_blk1[:, None] * 96 + yindex_0_blk1[None, :]
    xy_mask_blk1 = xmask_blk1[:, None] & ymask_blk1[None,:]
    xy_mask_blk = xmask_blk[:, None] & ymask_blk[None,:]

    tmp0 = ttgl.amd.cdna3.buffer_load(ptr=in_ptr0, offsets=xindex_1_blk, mask=xmask_blk)
    #tmp0 = tl.load(in_ptr0 + xindex, xmask, eviction_policy='evict_last')
    tmp6_blk1 = ttgl.amd.cdna3.buffer_load(ptr=in_ptr1, offsets=xy_offsets, mask=xy_mask_blk1)
    #tmp6 = tl.load(in_ptr1 + xy_index, xy_mask)
    tmp8 = ttgl.amd.cdna3.buffer_load(ptr=in_ptr2, offsets=xindex_1_blk, mask=xmask_blk)
    tmp13 = ttgl.amd.cdna3.buffer_load(ptr=in_ptr3, offsets=xindex_1_blk, mask=xmask_blk)
    #tmp8 = tl.load(in_ptr2 + xindex, xmask, eviction_policy='evict_last')
    #tmp13 = tl.load(in_ptr3 + xindex, xmask, eviction_policy='evict_last')

    # tmp1 = ttgl.full([BLOCK_M], 501, tl.int32)
    tmp2 = tmp0 + 501
    tmp3 = tmp0 < 0
    tmp4 = ttgl.where(tmp3, tmp2, tmp0).to(tl.int32)
    # tl.device_assert(((0 <= tmp4) & (tmp4 < 501)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 501")
    tmp6_blk = ttgl.convert_layout(tmp6_blk1, blocked)
    tmp7 = tmp6_blk.to(tl.float32)

    tmp9 = tmp8 + 501
    tmp10 = tmp8 < 0
    tmp11 = ttgl.where(tmp10, tmp9, tmp8).to(tl.int32)
    # tl.device_assert(((0 <= tmp11) & (tmp11 < 501)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 501")
    # tmp14 = tl.full([BLOCK_M], 6048, tl.int32)
    tmp15 = tmp13 + 6048
    tmp16 = tmp13 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp13).to(tl.int32)
    # tl.device_assert(((0 <= tmp17) & (tmp17 < 6048)) | ~(xmask), "index out of bounds: 0 <= tmp17 < 6048")

    # ttgl.amd.cdna4.buffer_atomic_add(ptr=out_ptr0, 
    #                                  offsets=(yindex_0_blk[None,:] + 96*tmp4[:,None]), 
    #                                  value=tmp7, 
    #                                  mask=xy_mask_blk, 
    #                                  sem='relaxed')
    # ttgl.amd.cdna4.buffer_atomic_add(ptr=out_ptr1, 
    #                                  offsets=(yindex_0_blk[None,:] + 96*tmp11[:,None]), 
    #                                  value=tmp7, 
    #                                  mask=xy_mask_blk, 
    #                                  sem='relaxed')
    # ttgl.amd.cdna4.buffer_atomic_add(ptr=out_ptr2, 
    #                                  offsets=(yindex_0_blk[None,:] + 96*tmp17[:,None]), 
    #                                  value=tmp7, 
    #                                  mask=xy_mask_blk, 
    #                                  sem='relaxed')
    ttgl.atomic_add(out_ptr0 + (yindex_0_blk[None,:] + 96*tmp4[:,None]), tmp7, xy_mask_blk, sem='relaxed')
    ttgl.atomic_add(out_ptr1 + (yindex_0_blk[None,:] + 96*tmp11[:,None]), tmp7, xy_mask_blk, sem='relaxed')
    ttgl.atomic_add(out_ptr2 + (yindex_0_blk[None,:] + 96*tmp17[:,None]), tmp7, xy_mask_blk, sem='relaxed')


@gluon.jit
def gluon_poi_fused__to_copy_index_add_new_zeros_v1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, 
                                                       x, y, BLOCK_M : tl.constexpr, BLOCK_N : tl.constexpr):
    
    blocked: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[4, 1], order=[1, 0])
    blocked1: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 16], warps_per_cta=[4, 1], order=[1, 0])

    blocked_out: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[64, 1], warps_per_cta=[1, 4], order=[1, 0])

    xoffset = ttgl.program_id(axis=0) * BLOCK_M
    xindex_1_blk = xoffset + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, blocked))
    xindex_0_blk = xoffset + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(0, blocked))
    xindex_1_blk1 = xoffset + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, blocked1))
    xindex_0_blk1 = xoffset + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(0, blocked1))

    yindex_1_blk = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, blocked))
    yindex_0_blk = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, blocked))
    yindex_1_blk1 = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, blocked1))
    yindex_0_blk1 = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, blocked1))

    xmask_blk = xindex_1_blk < x
    xmask_blk1 = xindex_1_blk1 < x

    ymask_blk = yindex_0_blk < y
    ymask_blk1 = yindex_0_blk1 < y

    xy_offsets = xindex_1_blk1[:, None] * 96 + yindex_0_blk1[None, :]
    xy_mask_blk1 = xmask_blk1[:, None] & ymask_blk1[None,:]
    xy_mask_blk = xmask_blk[:, None] & ymask_blk[None,:]

    # wrap up output blocked layout
    xindex_1_blk_out = xoffset + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, blocked_out))
    yindex_0_blk_out = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, blocked_out))
    xmask_blk_out = xindex_1_blk_out < x
    ymask_blk_out = yindex_0_blk_out < y
    xy_mask_out = xmask_blk_out[:, None] & ymask_blk_out[None,:]


    tmp0 = ttgl.amd.cdna3.buffer_load(ptr=in_ptr0, offsets=xindex_1_blk, mask=xmask_blk)
    #tmp0 = tl.load(in_ptr0 + xindex, xmask, eviction_policy='evict_last')
    tmp6_blk1 = ttgl.amd.cdna3.buffer_load(ptr=in_ptr1, offsets=xy_offsets, mask=xy_mask_blk1)
    #tmp6 = tl.load(in_ptr1 + xy_index, xy_mask)
    tmp8 = ttgl.amd.cdna3.buffer_load(ptr=in_ptr2, offsets=xindex_1_blk, mask=xmask_blk)
    tmp13 = ttgl.amd.cdna3.buffer_load(ptr=in_ptr3, offsets=xindex_1_blk, mask=xmask_blk)
    #tmp8 = tl.load(in_ptr2 + xindex, xmask, eviction_policy='evict_last')
    #tmp13 = tl.load(in_ptr3 + xindex, xmask, eviction_policy='evict_last')

    # tmp1 = ttgl.full([BLOCK_M], 501, tl.int32)
    tmp2 = tmp0 + 501
    tmp3 = tmp0 < 0
    tmp4 = ttgl.where(tmp3, tmp2, tmp0).to(tl.int32)
    out0_offset = (yindex_0_blk[None,:] + 96*tmp4[:,None])
    out0_offset_blk_out = ttgl.convert_layout(out0_offset, blocked_out)
    # tmp4_blk_out = ttgl.convert_layout(tmp4, blocked_out)
    # tl.device_assert(((0 <= tmp4) & (tmp4 < 501)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 501")
    tmp6_blk_out = ttgl.convert_layout(tmp6_blk1, blocked_out)
    tmp7 = tmp6_blk_out.to(tl.float32)

    tmp9 = tmp8 + 501
    tmp10 = tmp8 < 0
    tmp11 = ttgl.where(tmp10, tmp9, tmp8).to(tl.int32)
    out1_offset = (yindex_0_blk[None,:] + 96*tmp11[:,None])
    out1_offset_blk_out = ttgl.convert_layout(out1_offset, blocked_out)

    # tmp11_blk_out = ttgl.convert_layout(tmp11, blocked_out)
    # tl.device_assert(((0 <= tmp11) & (tmp11 < 501)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 501")
    # tmp14 = tl.full([BLOCK_M], 6048, tl.int32)
    tmp15 = tmp13 + 6048
    tmp16 = tmp13 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp13).to(tl.int32)
    out2_offset = (yindex_0_blk[None,:] + 96*tmp17[:,None])
    out2_offset_blk_out = ttgl.convert_layout(out2_offset, blocked_out)
    # tmp17_blk_out = ttgl.convert_layout(tmp17, blocked_out)
    # tl.device_assert(((0 <= tmp17) & (tmp17 < 6048)) | ~(xmask), "index out of bounds: 0 <= tmp17 < 6048")

    # ttgl.amd.cdna4.buffer_atomic_add(ptr=out_ptr0, 
    #                                  offsets=(yindex_0_blk[None,:] + 96*tmp4[:,None]), 
    #                                  value=tmp7, 
    #                                  mask=xy_mask_blk, 
    #                                  sem='relaxed')
    # ttgl.amd.cdna4.buffer_atomic_add(ptr=out_ptr1, 
    #                                  offsets=(yindex_0_blk[None,:] + 96*tmp11[:,None]), 
    #                                  value=tmp7, 
    #                                  mask=xy_mask_blk, 
    #                                  sem='relaxed')
    # ttgl.amd.cdna4.buffer_atomic_add(ptr=out_ptr2, 
    #                                  offsets=(yindex_0_blk[None,:] + 96*tmp17[:,None]), 
    #                                  value=tmp7, 
    #                                  mask=xy_mask_blk, 
    #                                  sem='relaxed')
    # ttgl.store(out_ptr0 + out0_offset_blk_out, tmp7, xy_mask_out)
    # ttgl.store(out_ptr1 + out1_offset_blk_out, tmp7, xy_mask_out)
    # ttgl.store(out_ptr2 + out2_offset_blk_out, tmp7, xy_mask_out)
    ttgl.atomic_add(out_ptr0 + out0_offset_blk_out, tmp7, xy_mask_out, sem='relaxed')
    ttgl.atomic_add(out_ptr1 + out1_offset_blk_out, tmp7, xy_mask_out, sem='relaxed')
    ttgl.atomic_add(out_ptr2 + out2_offset_blk_out, tmp7, xy_mask_out, sem='relaxed')


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

def get_args():
    torch.manual_seed(10)
    high_bound = 32
    arg_0 = rand_int_strided((1093610,), (1,), low=0, high=high_bound, device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((1093610, 96), (96, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_int_strided((1093610,), (1,), low=0, high=high_bound, device='cuda:0', dtype=torch.int64)
    arg_3 = rand_int_strided((1093610,), (1,), low=0, high=high_bound, device='cuda:0', dtype=torch.int64)

    arg_4 = rand_strided((501, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((501, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((6048, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg_4.zero_()
    arg_5.zero_()
    arg_6.zero_()
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 104986560


def run_torch_naive_ref():
    # torch naive run
    kernel_args_ref = list(get_args())
    torch_naive_ref(*kernel_args_ref)
    ref_out0 = kernel_args_ref[4]
    ref_out1 = kernel_args_ref[5]
    ref_out2 = kernel_args_ref[6]

    return ref_out0, ref_out1, ref_out2

def test_original_kernel_correctness():
    # torch reference impl
    ref_out0, ref_out1, ref_out2 = run_torch_naive_ref()

    # triton run
    kwargs = {'num_warps': 1, 'num_stages': 1}
    kernel_args = list(get_args())
    added_args = {'XBLOCK': 64}
    constexpr_positional_args = []
    for param_name in added_args:
        constexpr_positional_args.append(added_args[param_name])
    xnumel = 104986560
    XBLOCK = 64
    grid_x = (xnumel + XBLOCK - 1) // XBLOCK
    grid = (grid_x, 1, 1)
    triton_poi_fused__to_copy_index_add_new_zeros_4[grid](*kernel_args, *constexpr_positional_args, **kwargs)
    tri_out0 = kernel_args[4]
    tri_out1 = kernel_args[5]
    tri_out2 = kernel_args[6]

    # check torch and triton output correctness
    rtol = 2e-4
    atol = 2e-4
    torch.testing.assert_close(ref_out0, tri_out0, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_out1, tri_out1, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_out2, tri_out2, rtol=rtol, atol=atol)


def run_original_kernel_perf():
    # Configuration
    xnumel = 104986560
    XBLOCK = 64
    grid_x = (xnumel + XBLOCK - 1) // XBLOCK
    grid = (grid_x, 1, 1)
    kwargs = {'num_warps': 1, 'num_stages': 1}
    num_gb = 0.23892696

    # Get kernel arguments (convert to list for mutability in restore operations)
    kernel_args = list(get_args())
    added_args = {'XBLOCK': 64}

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
    print()
    print()
    print(f"-----------------------------------------------------------------------")
    print(f">>>>>> Original Kernel: triton_poi_fused__to_copy_index_add_new_zeros_4")
    print(f"Grid: {grid}")
    print()

    # Measure performance
    ms = do_bench(kernel_launch, warmup=25, rep=100)
    print(f"Median time: {ms:.3f}ms")
    gb_per_s = 0.23892696 / (ms / 1000.0)
    print(f"Throughput: {gb_per_s:.1f}GB/s")


def test_v1_correctness():
    # torch reference impl
    ref_out0, ref_out1, ref_out2 = run_torch_naive_ref()

    # triton run
    row_num = 1093610
    kwargs = {'num_warps': 1, 'num_stages': 1}
    kernel_args = list(get_args())
    kernel_args[-1] = row_num
    kernel_args.append(96)
    BLOCK_M, BLOCK_N = 64, 128
    added_args = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N}
    constexpr_positional_args = []
    for param_name in added_args:
        constexpr_positional_args.append(added_args[param_name])
    grid_x = (row_num + BLOCK_M - 1) // BLOCK_M
    grid = (grid_x, 1, 1)
    triton_poi_fused__to_copy_index_add_new_zeros_4_v1[grid](*kernel_args, *constexpr_positional_args, **kwargs)
    tri_out0 = kernel_args[4]
    tri_out1 = kernel_args[5]
    tri_out2 = kernel_args[6]

    # check torch and triton output correctness
    rtol = 2e-4
    atol = 2e-4
    torch.testing.assert_close(ref_out0, tri_out0, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_out1, tri_out1, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_out2, tri_out2, rtol=rtol, atol=atol)


def run_v1_perf():
    row_num = 1093610
    kwargs = {'num_warps': 4, 'num_stages': 1}
    kernel_args = list(get_args())
    kernel_args[-1] = row_num
    kernel_args.append(96)
    BLOCK_M, BLOCK_N = 128, 128
    added_args = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N}

    grid_x = (row_num + BLOCK_M - 1) // BLOCK_M
    grid = (grid_x, 1, 1)

    # Initialize CUDA device
    cuda_device = 0

    # Initialize CUDA context
    torch.cuda.set_device(cuda_device)
    torch.cuda.init()  # Explicitly initialize CUDA

    # Create launch lambda
    def v1_kernel_launch():
        with torch.cuda._DeviceGuard(cuda_device):
            torch.cuda.set_device(cuda_device)
            # For kernels with constexpr params in signature, append them as positional args
            constexpr_positional_args = []
            for param_name in added_args:
                constexpr_positional_args.append(added_args[param_name])
            return triton_poi_fused__to_copy_index_add_new_zeros_4_v1[grid](*kernel_args, *constexpr_positional_args, **kwargs)

    # Run benchmark
    print()
    print()
    print(f"-----------------------------------------------------------------")
    print(f">>>>>> Kernel: triton_poi_fused__to_copy_index_add_new_zeros_4_v1")
    print(f"Grid: {grid}")
    print()

    # Measure performance
    ms = do_bench(v1_kernel_launch, warmup=25, rep=100)
    print(f"Median time: {ms:.3f}ms")
    gb_per_s = 0.23892696 / (ms / 1000.0)
    print(f"Throughput: {gb_per_s:.1f}GB/s")


def test_gluon_correctness():
    # torch reference impl
    ref_out0, ref_out1, ref_out2 = run_torch_naive_ref()

    # triton run
    row_num = 1093610
    kwargs = {'num_warps': 4, 'num_stages': 1}
    kernel_args = list(get_args())
    kernel_args[-1] = row_num
    kernel_args.append(96)
    BLOCK_M, BLOCK_N = 128, 128
    added_args = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N}
    constexpr_positional_args = []
    for param_name in added_args:
        constexpr_positional_args.append(added_args[param_name])
    grid_x = (row_num + BLOCK_M - 1) // BLOCK_M
    grid = (grid_x, 1, 1)
    gluon_poi_fused__to_copy_index_add_new_zeros_v1[grid](*kernel_args, *constexpr_positional_args, **kwargs)
    tri_out0 = kernel_args[4]
    tri_out1 = kernel_args[5]
    tri_out2 = kernel_args[6]

    # check torch and triton output correctness
    rtol = 2e-4
    atol = 2e-4
    torch.testing.assert_close(ref_out0, tri_out0, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_out1, tri_out1, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_out2, tri_out2, rtol=rtol, atol=atol)


def run_gluon_perf():
    row_num = 1093610
    kwargs = {'num_warps': 4, 'num_stages': 1}
    kernel_args = list(get_args())
    kernel_args[-1] = row_num
    kernel_args.append(96)
    BLOCK_M, BLOCK_N = 128, 128
    added_args = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N}

    grid_x = (row_num + BLOCK_M - 1) // BLOCK_M
    grid = (grid_x, 1, 1)

    # Initialize CUDA device
    cuda_device = 0

    # Initialize CUDA context
    torch.cuda.set_device(cuda_device)
    torch.cuda.init()  # Explicitly initialize CUDA

    # Create launch lambda
    def gluon_kernel_launch():
        with torch.cuda._DeviceGuard(cuda_device):
            torch.cuda.set_device(cuda_device)
            # For kernels with constexpr params in signature, append them as positional args
            constexpr_positional_args = []
            for param_name in added_args:
                constexpr_positional_args.append(added_args[param_name])
            return gluon_poi_fused__to_copy_index_add_new_zeros_v1[grid](*kernel_args, *constexpr_positional_args, **kwargs)

    # Run benchmark
    print()
    print()
    print(f"-------------------------------------------------------------")
    print(f">>>>>> Kernel: gluon_poi_fused__to_copy_index_add_new_zeros_v1")
    print(f"Grid: {grid}")
    print()

    # Measure performance
    ms = do_bench(gluon_kernel_launch, warmup=25, rep=100)
    print(f"Median time: {ms:.3f}ms")
    gb_per_s = 0.23892696 / (ms / 1000.0)
    print(f"Throughput: {gb_per_s:.1f}GB/s")


if __name__ == '__main__':
    test_original_kernel_correctness()
    test_v1_correctness()
    test_gluon_correctness()
    run_original_kernel_perf()
    run_v1_perf()
    run_gluon_perf()
