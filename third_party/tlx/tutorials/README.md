- Run the script: third_party/tlx/tutorials/amd_bmm_shared_perf.py and got the following perf numbers:
```
  Script run: amd_bmm_shared_perf.py completed successfully. Results:

  M x N x K (B)         path     TLX    rocBLAS   ratio
  1024x256x256 (320)    direct    88u      76u    0.86x  OK
  395x256x320 (1024)    direct   146u     121u    0.83x  OK
  40x256x1956 (1024)    reg      192u     182u    0.95x  OK
  262x256x294 (1024)    reg      124u      81u    0.65x  OK
  1195x256x2309 (1024)  reg     2181u    1705u    0.78x  OK
```

The hipblasLT kernels called for each input shape
```
  ┌──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ 
  │ Shape (B×M×N×K)      │ hipBLASLt kernel                                                                                                         │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
  │ 320×1024×256×256     │ Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x32_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_    │ 
  │                      │ DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA4096_LBSPPB512_LBSPPM0_     │ 
  │                      │ LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_   │ 
  │                      │ SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_  │ 
  │                      │ WG32_8_1                                                                                                                 │ 
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
  │ 1024×395×256×320     │ Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x32_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_    │ 
  │                      │ DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB2_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA4096_LBSPPB128_LBSPPM0_     │ 
  │                      │ LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_  │ 
  │                      │ SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_  │ 
  │                      │ WG64_4_1                                                                                                                 │ 
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
  │ 1024×40×256×1956     │ Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT256x48x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_     │ 
  │                      │ DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA4096_LBSPPB128_LBSPPM0_     │ 
  │                      │ LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_3_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_   │ 
  │                      │ SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_  │ 
  │                      │ WG64_4_1                                                                                                                 │ 
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
  │ 1024×262×256×294     │ Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT256x144x32_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_    │ 
  │                      │ DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB2_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA4096_LBSPPB128_LBSPPM0_     │ 
  │                      │ LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_9_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_   │ 
  │                      │ SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_  │ 
  │                      │ WG64_4_1                                                                                                                 │ 
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ 
  │ 1024×1195×256×2309   │ Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x32_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_    │ 
  │                      │ DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB2_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA4096_LBSPPB128_LBSPPM0_     │ 
  │                      │ LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_15_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_  │ 
  │                      │ SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_  │ 
  │                      │ WG64_4_1                                                                                                                 │ 
  └──────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ 
```

- Collected the att trace of hipblast and triton kernel `_bmm_register` and hipblasLT for two shapes `1024×262×256×294`, here are the difference:
```
┌──────────────────────┬─────────────┬─────────────┐
│      Parameter       │   Triton    │  hipBLASLt  │
├──────────────────────┼─────────────┼─────────────┤
│ BLOCK_M              │ 128         │ 144         │
├──────────────────────┼─────────────┼─────────────┤
│ BLOCK_N              │ 256         │ 256         │
├──────────────────────┼─────────────┼─────────────┤
│ BLOCK_K              │ 32          │ 32          │
├──────────────────────┼─────────────┼─────────────┤
│                      │             │             │
├──────────────────────┼─────────────┼─────────────┤
│ tile_size_A          │ 4096        │ 4608        │
├──────────────────────┼─────────────┼─────────────┤
│ tile_size_B          │ 8192        │ 8192        │
├──────────────────────┼─────────────┼─────────────┤
│ num_warps            │ 8           │ 4           │
├──────────────────────┼─────────────┼─────────────┤
│                      │             │             │
├──────────────────────┼─────────────┼─────────────┤
│ A load instruction   │ buf_ushort  │ buf_dword   │
├──────────────────────┼─────────────┼─────────────┤
│ A load instruction # │ 8           │ 9           │
├──────────────────────┼─────────────┼─────────────┤
│ B load instruction   │ buf_dwordx4 │ buf_dwordx4 │
├──────────────────────┼─────────────┼─────────────┤
│ B load instruction # │ 2           │ 4           │
├──────────────────────┼─────────────┼─────────────┤
│                      │             │             │
├──────────────────────┼─────────────┼─────────────┤
│ matrix_instr_nonkdim │ 32          │ 16          │
├──────────────────────┼─────────────┼─────────────┤
│ # MFMA instructions  │ 8           │ 36          │
├──────────────────────┼─────────────┼─────────────┤
│                      │             │             │
├──────────────────────┼─────────────┼─────────────┤
│ # workgroups         │ 3072        │ 2048        │
├──────────────────────┼─────────────┼─────────────┤
│ total shape (M×N×K)  │ 384×256×294 │ 288×256×294 │
└──────────────────────┴─────────────┴─────────────┘
  ```
Some findings:
- Loading of A in Triton uses `buffer_load_ushort`, but it uses `buffer_load_dword` in hipblasLT
- mfma shape, Triton uses mfma32, hipblaslt uses mfma16
- hipBlasLT can use tile size `BLOCK_M=144` (not power of 2). Compared to the triton configuration, it does `34%` (`384/288 - 1 = 0.34`) less computation.

To do:
- Check if there is way to use `buffer_load_dword` to load A, this can reduce the overhead of loading A from global memory to VGPR and then write to LDS
- Since A matrix is shared along the batch dim, we can share the A tile among a
fixed number of b matrix. (This can reduce the overhead of loading A and this is 
where the buffer_load_ushort is used)

For the shape `1024×1195×256×2309`, triton uses the config `(BM, BN, BK) = (128, 256, 32)`, hipblasLT use the config: `(BM, BN, BK) = (240, 256, 32)`

- Implemented each A tile is used by 2 B matrix, but perf is not good for the initial implementation, need further investigation
```
M x N x K (B)         path          TLX   rocBLAS   ratio  ok
1024x256x256 (320)    direct        86u       76u   0.88x  OK
395x256x320 (1024)    direct       153u      123u   0.80x  OK
40x256x1956 (1024)    reg          229u      182u   0.80x  OK
262x256x294 (1024)    reg          155u       85u   0.55x  OK
1195x256x2309 (1024)  reg         2338u     1713u   0.73x  OK
```

# Aug 12 2026
- Changed the kernel to add explicit indication of divisibility of A matrix and K value, 
- Remove the masks from the async_load,
got the following perf numbers:
```
M x N x K (B)         path          TLX   rocBLAS   ratio  ok
1024x256x256 (320)    direct        87u       75u   0.86x  OK
395x256x320 (1024)    direct       144u      122u   0.85x  OK
40x256x1956 (1024)    reg          172u      175u   1.01x  OK
262x256x294 (1024)    reg          113u       83u   0.74x  OK
1195x256x2309 (1024)  reg         2064u     1717u   0.83x  OK
```

# Aug 13 2026
Implemented shared-A multiB optimization (one CTA processes NUM_B_MATRIX=2 consecutive
B matrices from a single A load), fixing a critical LDS overflow bug in the previous
attempt and profiling to characterize the VGPR bottleneck.

Key findings:
- **LDS overflow root cause**: original `_bmm_register_multiB` with NB=3, NUM_B_MATRIX=2,
  BM=128 used 120KB LDS (exceeds 96KB CDNA4 limit) → kernel silently ran OOB. Fix: NB_MULTI=2
  (LDS: 80KB for BM=128).
- **VGPR spill root cause**: at num_warps=4, two (128,256)fp32 accumulators cause 643 VGPR
  spills (ScratchSize=920B) → 8x slowdown. At num_warps=8: 232 VGPRs, 0 spills.
- **MultiB only helps for M≤64**: halving NT (CTA count) hurts GPU occupancy for large-M
  shapes (M=262, M=1195) where NT is already high. For M=40 (NT=512→256), multiB is neutral
  to slightly positive.
- **Direct path nw=8 is a net win**: switching from nw=8 (already correct) confirms the
  direct path is optimal. A small improvement on 1024x256x256 (0.86→0.87x).

Current perf (multiB enabled for M≤64 reg path, single-B nw=8 for direct path):
```
M x N x K (B)         path          TLX   rocBLAS   ratio  ok
1024x256x256 (320)    direct        86u       75u   0.87x  OK
395x256x320 (1024)    direct       143u      121u   0.84x  OK
40x256x1956 (1024)    reg          176u      177u   1.01x  OK
262x256x294 (1024)    reg          117u       83u   0.74x  OK
1195x256x2309 (1024)  reg         2040u     1697u   0.83x  OK
```

Remaining gap for reg-path shapes is the same root cause as before: `buffer_load_ushort`
(1 fp16/VGPR) vs rocBLAS's d16-packed loads (2 fp16/VGPR) — a codegen limitation, not
a tiling/pipelining issue.

