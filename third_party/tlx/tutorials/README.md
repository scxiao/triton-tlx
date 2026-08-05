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

/workspace/projects/triton-tlx/python/triton/runtime/jit.py:525: UserWarning: [Triton] TRITON_ENABLE_C_CACHE: C fast path bypassed for kernel '__main__._bmm_direct': unknown reason
  return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
1024x256x256 (320)    direct        88u       76u   0.87x  OK
395x256x320 (1024)    direct       146u      122u   0.83x  OK
40x256x1956 (1024)    reg          193u      183u   0.95x  OK
262x256x294 (1024)    reg          125u       82u   0.66x  OK
1195x256x2309 (1024)  reg         2195u     1717u   0.78x  OK

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