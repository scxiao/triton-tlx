# KERNEL CALLS: 1

import triton
import triton.language as tl

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
def triton_poi_fused__to_copy_index_add_new_zeros_4buffers(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 104986560
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    buffer_idx = pid % 4
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
    buffer_size_1 = 501 * 96
    buffer_offset_1 = buffer_idx * buffer_size_1
    tl.atomic_add(out_ptr0 + (x0 + 96*tmp4 + buffer_offset_1), tmp7, xmask, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x0 + 96*tmp11 + buffer_offset_1), tmp7, xmask, sem='relaxed')
    buffer_size_2 = 6048 * 96
    buffer_offset_2 = buffer_idx * buffer_size_2
    tl.atomic_add(out_ptr2 + (x0 + 96*tmp17 + buffer_offset_2), tmp7, xmask, sem='relaxed')


@triton.jit
def triton_reduce_4_buffers_1(out_ptr2, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_num : tl.constexpr = 1093610
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]

    row_num = 6048
    xnumel = row_num * 96
    buffer_1_offset = xnumel
    buffer_2_offset = 2 * xnumel
    buffer_3_offset = 3 * xnumel 
    xmask = xindex < xnumel
    out2_tmp0 = tl.load(out_ptr2 + xindex, xmask)
    out2_tmp1 = tl.load(out_ptr2 + xindex + buffer_1_offset, xmask)
    out2_tmp2 = tl.load(out_ptr2 + xindex + buffer_2_offset, xmask)
    out2_tmp3 = tl.load(out_ptr2 + xindex + buffer_3_offset, xmask)
    out2_tmp = out2_tmp0 + out2_tmp1 + out2_tmp2 + out2_tmp3
    tl.store(out_ptr2 + xindex, out2_tmp, xmask)


@triton.jit
def triton_reduce_4_buffers_2(out_ptr0, out_ptr1, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]

    xnumel = 501 * 96
    buffer_1_offset = xnumel
    buffer_2_offset = 2 * xnumel
    buffer_3_offset = 3 * xnumel 
    xmask = xindex < xnumel
    # read from the same location from 4 buffers
    out0_tmp0 = tl.load(out_ptr0 + xindex, xmask)
    out0_tmp1 = tl.load(out_ptr0 + xindex + buffer_1_offset, xmask)
    out0_tmp2 = tl.load(out_ptr0 + xindex + buffer_2_offset, xmask)
    out0_tmp3 = tl.load(out_ptr0 + xindex + buffer_3_offset, xmask)
    out0_tmp = out0_tmp0 + out0_tmp1 + out0_tmp2 + out0_tmp3
    tl.store(out_ptr0 + xindex, out0_tmp, xmask)

    out1_tmp0 = tl.load(out_ptr1 + xindex, xmask)
    out1_tmp1 = tl.load(out_ptr1 + xindex + buffer_1_offset, xmask)
    out1_tmp2 = tl.load(out_ptr1 + xindex + buffer_2_offset, xmask)
    out1_tmp3 = tl.load(out_ptr1 + xindex + buffer_3_offset, xmask)
    out1_tmp = out1_tmp0 + out1_tmp1 + out1_tmp2 + out1_tmp3
    tl.store(out_ptr1 + xindex, out1_tmp, xmask)


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
    torch.manual_seed(10)
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


def get_args_4_out_buffers():
    torch.manual_seed(10)
    arg_0 = rand_int_strided((1093610,), (1,), low=0, high=high_bound, device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((1093610, 96), (96, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_int_strided((1093610,), (1,), low=0, high=high_bound, device='cuda:0', dtype=torch.int64)
    arg_3 = rand_int_strided((1093610,), (1,), low=0, high=high_bound, device='cuda:0', dtype=torch.int64)

    arg_4 = rand_strided((501 * 4, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((501 * 4, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((6048 * 4, 96), (96, 1), device='cuda:0', dtype=torch.float32)
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
    
    print(f"original_kernel_correctness: passed")


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


def test_4_buffers_correctness():
    # torch reference impl
    ref_out0, ref_out1, ref_out2 = run_torch_naive_ref()

    # triton run
    kwargs = {'num_warps': 1, 'num_stages': 1}
    kernel_args = list(get_args_4_out_buffers())
    added_args = {'XBLOCK': 64}
    constexpr_positional_args = []
    for param_name in added_args:
        constexpr_positional_args.append(added_args[param_name])
    xnumel = 104986560
    XBLOCK = 64
    grid_x = (xnumel + XBLOCK - 1) // XBLOCK
    grid = (grid_x, 1, 1)
    triton_poi_fused__to_copy_index_add_new_zeros_4buffers[grid](*kernel_args, *constexpr_positional_args, **kwargs)

    xnumel = 6048 * 96
    grid_x = (xnumel + XBLOCK - 1) // XBLOCK
    grid_1 = (grid_x, 1, 1)
    triton_reduce_4_buffers_1[grid_1](kernel_args[6], XBLOCK=XBLOCK)

    xnumel = 501 * 96
    grid_x = (xnumel + XBLOCK - 1) // XBLOCK
    grid_2 = (grid_x, 1, 1)
    triton_reduce_4_buffers_2[grid_2](kernel_args[4], kernel_args[5], XBLOCK=XBLOCK)
    tri_out0 = kernel_args[4][0:501]
    tri_out1 = kernel_args[5][0:501]
    tri_out2 = kernel_args[6][0:6048]

    # check torch and triton output correctness
    rtol = 2e-4
    atol = 2e-4
    torch.testing.assert_close(ref_out0, tri_out0, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_out1, tri_out1, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_out2, tri_out2, rtol=rtol, atol=atol)

    print(f"4_buffer_kernel_correctness: passed")

def run_4_buffers_perf():
    # Configuration
    xnumel = 104986560
    XBLOCK = 64
    grid_x = (xnumel + XBLOCK - 1) // XBLOCK
    grid = (grid_x, 1, 1)
    kwargs = {'num_warps': 1, 'num_stages': 1}
    num_gb = 0.23892696

    # Get kernel arguments (convert to list for mutability in restore operations)
    kernel_args = list(get_args_4_out_buffers())
    added_args = {'XBLOCK': 64}

    xnumel = 6048 * 96
    grid_x = (xnumel + XBLOCK - 1) // XBLOCK
    grid_1 = (grid_x, 1, 1)

    xnumel = 501 * 96
    grid_x = (xnumel + XBLOCK - 1) // XBLOCK
    grid_2 = (grid_x, 1, 1)

    # Initialize CUDA device
    cuda_device = 0

    # Initialize CUDA context
    torch.cuda.set_device(cuda_device)
    torch.cuda.init()  # Explicitly initialize CUDA

    # Create launch lambda
    def kernel_launch_4_buffers():
        with torch.cuda._DeviceGuard(cuda_device):
            torch.cuda.set_device(cuda_device)
            # For kernels with constexpr params in signature, append them as positional args
            constexpr_positional_args = []
            for param_name in added_args:
                constexpr_positional_args.append(added_args[param_name])
            triton_poi_fused__to_copy_index_add_new_zeros_4buffers[grid](*kernel_args, *constexpr_positional_args, **kwargs)
            triton_reduce_4_buffers_1[grid_1](kernel_args[6], XBLOCK=XBLOCK)
            triton_reduce_4_buffers_2[grid_2](kernel_args[4], kernel_args[5], XBLOCK=XBLOCK)

    # Run benchmark
    print()
    print()
    print(f"-----------------------------------------------------------------------")
    print(f">>>>>> first optimized version: triton_poi_fused__to_copy_index_add_new_zeros_4_buffers")
    print(f"Grid: {grid}")
    print()

    # Measure performance
    ms = do_bench(kernel_launch_4_buffers, warmup=25, rep=100)
    print(f"Median time: {ms:.3f}ms")
    gb_per_s = 0.23892696 / (ms / 1000.0)
    print(f"Throughput: {gb_per_s:.1f}GB/s")



if __name__ == '__main__':
    test_original_kernel_correctness()
    test_4_buffers_correctness()
    run_original_kernel_perf()
    run_4_buffers_perf()