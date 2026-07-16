## Jul 13 2026
1. Run the command `python python glocal_full.py` generates the following outputs:
```
repro M=32768 N=262144 K=4096  x=(32768, 4096) w=(1, 262144, 4096)
x = tensor([0.3984, 0.5156, 0.0249, 0.9414, 0.9453, 0.7969, 0.4141, 0.8203, 0.2295,
        0.9102], device='cuda:0', dtype=torch.bfloat16)
w = tensor([[ 0.7344, -0.2832,  0.4551,  ..., -1.7656, -0.5391, -1.0703],
        [-1.2578, -1.3047, -0.9141,  ...,  1.3984, -1.7266,  0.9102],
        [-0.1465, -0.1650,  0.4434,  ..., -0.4824,  0.2295,  0.1494],
        ...,
        [ 0.2432, -0.8477, -2.2500,  ...,  0.9805, -0.3242, -0.2090],
        [ 0.1973,  1.6641, -0.8789,  ..., -1.0000, -2.6562,  0.9336],
        [-0.5430, -0.0356, -0.0187,  ..., -0.8516,  0.2656,  0.1035]],
       device='cuda:0', dtype=torch.bfloat16)
gcc: warning: ‘-x c’ after last input file has no effect
  run 0: nhuge=265984
  run 1: nhuge=268352
  run 2: nhuge=262256
  run 3: nhuge=264368
```
The # of `nhuage` indicates the number of corrupted data in the output, corresponding to the following check in the reproducer. 
```
        invalid = (yf.abs() > 1e30) | ~torch.isfinite(yf)
        nhuge = int(invalid.sum())
        print(f"  run {r}: nhuge={nhuge}")
```

Another observation, different executions genereate different number of `nhuag` as:
```
root@smci355-ccs-aus-m01-21:/workspace/projects/triton-tlx# python glocal_full.py 
repro M=32768 N=262144 K=4096  x=(32768, 4096) w=(1, 262144, 4096)
x = tensor([0.3984, 0.5156, 0.0249, 0.9414, 0.9453, 0.7969, 0.4141, 0.8203, 0.2295,
        0.9102], device='cuda:0', dtype=torch.bfloat16)
w = tensor([[ 0.7344, -0.2832,  0.4551,  ..., -1.7656, -0.5391, -1.0703],
        [-1.2578, -1.3047, -0.9141,  ...,  1.3984, -1.7266,  0.9102],
        [-0.1465, -0.1650,  0.4434,  ..., -0.4824,  0.2295,  0.1494],
        ...,
        [ 0.2432, -0.8477, -2.2500,  ...,  0.9805, -0.3242, -0.2090],
        [ 0.1973,  1.6641, -0.8789,  ..., -1.0000, -2.6562,  0.9336],
        [-0.5430, -0.0356, -0.0187,  ..., -0.8516,  0.2656,  0.1035]],
       device='cuda:0', dtype=torch.bfloat16)
  run 0: nhuge=269216
  run 1: nhuge=280512
  run 2: nhuge=265488
  run 3: nhuge=259648
```
Note that the input data are exactly the same in different execution (call `torch.manual_seed()`).


2. some weird changes can make the corrupted output gone
There are two lines to write outputs
```
        tl.store(y_ptr + (row_t[:, None] * stride_ym + offs_n[None, :] * stride_yn),
                    yt, mask=smask_mt & mask_n_col)
        tl.store(y_ptr + (row_b[:, None] * stride_ym + offs_n[None, :] * stride_yn),
                    yb, mask=smask_mb & mask_n_col)
```
- If we change either one (or both) of them to buffer_store as:
```
        offs_t = row_t[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tlx.buffer_store(yt, y_ptr, offs_t.to(tl.uint32), mask=smask_mt & mask_n_col)
        offs_b = row_b[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tlx.buffer_store(yb, y_ptr, offs_b.to(tl.uint32), mask=smask_mb & mask_n_col)
```
The outputs are correct as:
```
repro M=32768 N=262144 K=4096  x=(32768, 4096) w=(1, 262144, 4096)
  run 0: nhuge=0
  run 1: nhuge=0
  run 2: nhuge=0
  run 3: nhuge=0
```
- If we add a `.to(tl.uint32)` in the `tl.store()` call for the offset calculation before adding to `y_ptr`, i.e., we change the two lines to
```
        tl.store(y_ptr + (row_t[:, None] * stride_ym + offs_n[None, :] * stride_yn).to(tl.uint32),
                    yt, mask=smask_mt & mask_n_col)
        tl.store(y_ptr + (row_b[:, None] * stride_ym + offs_n[None, :] * stride_yn).to(tl.uint32),
                    yb, mask=smask_mb & mask_n_col)
```
the results are also correct. This behavior corresponds to the buffer_store changes that require the offsets to be
`tl.uint32` or `tl.int32` data types.

- If we remove the mask in the `tl.store()`, i.e., the program is changed to:
```
        tl.store(y_ptr + (row_t[:, None] * stride_ym + offs_n[None, :] * stride_yn), yt)
        tl.store(y_ptr + (row_b[:, None] * stride_ym + offs_n[None, :] * stride_yn), yb)
```
there is no corrupted data.


3. Checking the indices of the corrupted values by dividing the output into 16 sections, all corrumpted values
are in the last two sections. Specifically, the output tensor has 32768 lines, dividing to 16 sections, which means
each section contains 2048 rows, so all corrupted data are in the last 4096 rows in the output tensor.

Seems like something wrong with regard to the offset calculation when `tl.int64` is used.

4. A lot of other changes that are supposed not to change the output behavior make an impact. For example
 - BLOCK_SIZE_M=256 -> 128
 - BLOCK_SIZE_N=256 -> 12
 - NUM_STAGES=2->1
 - num_warps=8->4

Tried the fix in the issue https://github.com/AMD-Triton/triton-tickets/issues/1350#issuecomment-4468473379, and include the mi350, the error outputs are still there, so it seems like
the problem is not related to issue: https://github.com/AMD-Triton/triton-tickets/issues/1350.


Jul 15 2026

1. Removing the second set of dot op does not help
```
        _workgroup_barrier()
        b_cur = tlx.local_load(tlx.local_view(buffers_B, oe), token=None)
        at_cur = tlx.local_load(tlx.local_view(buffers_A_top, oe), token=None)
        ab_cur = tlx.local_load(tlx.local_view(buffers_A_bot, oe), token=None)
        acc_top = tl.dot(at_cur, b_cur, acc_top)
        acc_bot = tl.dot(ab_cur, b_cur, acc_bot)
```

2. Switch the call of the two tl.store(), the problem is also gone

3. remove the mask of the `tl.store()` can also make the issue gone


Jul 16 2026
1. Disable the llvm optimization, problem is gone
`DISABLE_LLVM_OPT=1 python glocal_full.py`

