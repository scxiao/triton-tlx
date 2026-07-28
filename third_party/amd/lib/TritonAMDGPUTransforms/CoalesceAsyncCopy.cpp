#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-coalesce-async-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCOALESCEASYNCCOPY
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Fallback for a direct-to-LDS async copy that cannot be lowered on this
// target. On CDNA the per-thread vector width can collapse below a bitwidth
// supported for direct-to-LDS (only 32- or 128-bit are legal), e.g. a masked
// partial-K fp16 load (vec=1 -> 16-bit) or a non-16-element-aligned global row
// stride (vec collapses toward 1-2). `canLoadDirectToLDS` then returns false
// and the op has no legal lowering: it later fails to legalize in
// ConvertTritonAMDGPUToLLVM (`unrealized_conversion_cast` / "failed to legalize
// async_copy_global_to_local"). Rather than leave such an op, rewrite it into a
// synchronous tt.load + ttg.local_store. This mirrors the non-async
// load->local_store pipeline; the Membar pass inserts the LDS barrier before
// the consuming local_load, so it is correct. It only loses the direct
// GMEM->LDS overlap for this copy -- the tt.load is still turned into a
// vectorized (alignment-tolerant) buffer_load/global_load into registers by the
// later convert-to-buffer-ops pass.
static LogicalResult
decomposeAsyncCopyToSync(ttg::AsyncCopyGlobalToLocalOp copyOp,
                         PatternRewriter &rewriter) {
  Location loc = copyOp.getLoc();
  Value src = copyOp.getSrc();
  Value dst = copyOp.getResult();
  Value mask = copyOp.getMask();
  Value other = copyOp.getOther();

  rewriter.setInsertionPoint(copyOp);

  // Synchronous global load into registers.
  Value loaded;
  if (mask)
    loaded = triton::LoadOp::create(rewriter, loc, src, mask, copyOp.getCache(),
                                    copyOp.getEvict(), copyOp.getIsVolatile());
  else
    loaded = triton::LoadOp::create(rewriter, loc, src, copyOp.getCache(),
                                    copyOp.getEvict(), copyOp.getIsVolatile());

  // Preserve `other` (the fill value for masked-out lanes); a masked tt.load
  // leaves those lanes undefined, so select them explicitly.
  if (other && mask)
    loaded = arith::SelectOp::create(rewriter, loc, mask, loaded, other);

  // Store into the shared-memory buffer. Membar adds the barrier before the
  // consuming local_load, exactly as for the non-async pipeline.
  ttg::LocalStoreOp::create(rewriter, loc, loaded, dst);

  // The synchronous copy is not part of any async group: drop its token from
  // every async_commit_group / async_wait that consumed it (both accept zero
  // tokens).
  Value token = copyOp.getToken();
  SmallVector<Operation *> users(token.getUsers().begin(),
                                 token.getUsers().end());
  for (Operation *user : users) {
    SmallVector<Value> kept;
    for (Value t : user->getOperands())
      if (t != token)
        kept.push_back(t);
    rewriter.setInsertionPoint(user);
    if (auto commit = dyn_cast<ttg::AsyncCommitGroupOp>(user)) {
      auto n = ttg::AsyncCommitGroupOp::create(rewriter, commit.getLoc(), kept);
      rewriter.replaceOp(commit, n.getResult());
    } else if (auto wait = dyn_cast<ttg::AsyncWaitOp>(user)) {
      auto n = ttg::AsyncWaitOp::create(rewriter, wait.getLoc(), kept,
                                        wait.getNum());
      rewriter.replaceOp(wait, n.getResult());
    }
  }

  rewriter.eraseOp(copyOp);
  return success();
}

// On gfx9 global and buffer loads directly to shared memory need to write
// coalesced. This pattern converts the layout of the src, mask and other to
// ensure the owned data per thread is contiguous and does not exceed the
// supported load vector size.
//
// Works for both ttg::AsyncCopyGlobalToLocalOp and
// triton::amdgpu::BufferLoadToLocalOp via template specialisation of a few
// op-specific helpers gathered in OpTraits<OpTy>.
//
// Key differences between the two op types:
//   AsyncCopyGlobalToLocalOp  — source is a tensor-of-pointers (srcTy encodes
//     both pointer element type and the distributed layout).  Contiguity is
//     bounded by both the axis-info result AND the consecutive-in-out property
//     of the reg→shared mapping.  canLoadDirectToLDS is called with srcTy
//     directly.  The swizzled branch asserts sizePerThread >= loadContig.
//
//   BufferLoadToLocalOp       — source is a scalar pointer + offset tensor
//     (i32). The true contiguity comes from pointer/offset divisibility and
//     must NOT be additionally capped by the current reg→shared consecutive-
//     in-out (which would clamp it to 1 when sizePerThread=[1]).
//     canLoadDirectToLDS must be called with a reconstructed effective
//     pointer tensor type.  The swizzled branch does not assert srcElemContig.

// ---------------------------------------------------------------------------
// Op-specific traits (specialised below)
// ---------------------------------------------------------------------------
template <typename OpTy> struct OpTraits;

template <> struct OpTraits<ttg::AsyncCopyGlobalToLocalOp> {
  static Value getSourceTensor(ttg::AsyncCopyGlobalToLocalOp op) {
    return op.getSrc();
  }

  static Value getDestValue(ttg::AsyncCopyGlobalToLocalOp op) {
    return op.getResult();
  }

  static constexpr bool capByRegToShared = true;

  static RankedTensorType getEffectivePtrType(ttg::AsyncCopyGlobalToLocalOp op,
                                              RankedTensorType srcTy) {
    return srcTy;
  }

  static constexpr bool assertSrcElemContig = true;

  static void assignSource(ttg::AsyncCopyGlobalToLocalOp op,
                           PatternRewriter &rewriter, Value newSrc) {
    op.getSrcMutable().assign(newSrc);
  }
};

template <> struct OpTraits<triton::amdgpu::BufferLoadToLocalOp> {
  static Value getSourceTensor(triton::amdgpu::BufferLoadToLocalOp op) {
    return op.getOffsets();
  }

  static Value getDestValue(triton::amdgpu::BufferLoadToLocalOp op) {
    return op.getDest();
  }

  static constexpr bool capByRegToShared = false;

  static RankedTensorType
  getEffectivePtrType(triton::amdgpu::BufferLoadToLocalOp op,
                      RankedTensorType /*offsetTy*/) {
    return cast<RankedTensorType>(
        LLVM::AMD::getPointerTypeWithShape(op.getPtr(), op.getOffsets()));
  }

  static constexpr bool assertSrcElemContig = false;

  static void assignSource(triton::amdgpu::BufferLoadToLocalOp op,
                           PatternRewriter &rewriter, Value newOffsets) {
    op.getOffsetsMutable().assign(newOffsets);
  }
};

// ---------------------------------------------------------------------------
// Combined rewrite pattern
// ---------------------------------------------------------------------------
template <typename OpTy>
struct CoalesceAsyncCopyToLocal : public OpRewritePattern<OpTy> {
  using Traits = OpTraits<OpTy>;

  CoalesceAsyncCopyToLocal(const triton::AMD::TargetInfo &targetInfo,
                           const DenseMap<OpTy, unsigned> &contiguityMap,
                           MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx), targetInfo{targetInfo},
        contiguityMap{contiguityMap} {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value srcTensor = Traits::getSourceTensor(op);
    Value dst = Traits::getDestValue(op);
    Value mask = op.getMask();
    Value other = op.getOther();

    auto srcTy = cast<RankedTensorType>(srcTensor.getType());
    auto dstTy = cast<ttg::MemDescType>(dst.getType());

    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    if (!blockedEnc)
      return rewriter.notifyMatchFailure(op, "src encoding must be #blocked");

    if (!isa<ttg::SwizzledSharedEncodingAttr, ttg::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      return rewriter.notifyMatchFailure(
          op, "dst encoding must be #swizzled or #padded");
    }

    auto it = contiguityMap.find(op);
    if (it == contiguityMap.end())
      return op->emitError() << "No contiguity information about the copy op";
    unsigned loadContig = it->second;
    assert(loadContig > 0);

    LinearLayout regLayout = triton::gpu::toLinearLayout(srcTy);
    LinearLayout sharedLayout;
    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(dstTy.getEncoding());
    if (paddedEnc) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(dstTy);
    }
    auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);

    if (Traits::capByRegToShared)
      loadContig = std::min<unsigned>(
          loadContig, regToSharedLayout.getNumConsecutiveInOut());

    auto elemBitWidth = dstTy.getElementTypeBitWidth();
    loadContig =
        fitToValidDirectToLdsVecSize(loadContig, elemBitWidth, targetInfo);

    if (loadContig == 0) {
      if constexpr (std::is_same_v<OpTy, ttg::AsyncCopyGlobalToLocalOp>)
        return decomposeAsyncCopyToSync(op, rewriter);
      return rewriter.notifyMatchFailure(
          op, "could not find layout config to create coalesced writes");
    }

    auto mod = op->template getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::lookupNumWarps(op);
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);

    RankedTensorType effectivePtrTy = Traits::getEffectivePtrType(op, srcTy);
    if (LLVM::AMD::canLoadDirectToLDS(targetInfo, effectivePtrTy,
                                      dstTy.getEncoding(),
                                      dstTy.getAllocShape(), loadContig))
      return rewriter.notifyMatchFailure(op, "already writes coalesced");

    if (!targetInfo.supportsDirectToLdsLoadBitWidth(loadContig * elemBitWidth))
      return rewriter.notifyMatchFailure(op,
                                         "unable to find supported vector size "
                                         "based on src and dst encodings");

    ttg::DistributedEncodingTrait newDistEnc;

    if (isa<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding())) {
      auto contigPerThread = ttg::getContigPerThread(srcTy);
      if (Traits::assertSrcElemContig) {
        auto srcElemContig = contigPerThread[blockedEnc.getOrder()[0]];
        assert(srcElemContig >= loadContig);
      }
      contigPerThread[blockedEnc.getOrder()[0]] = loadContig;
      newDistEnc = BlockedEncodingAttr::get(
          op.getContext(), srcTy.getShape(), contigPerThread,
          blockedEnc.getOrder(), numWarps, threadsPerWarp,
          blockedEnc.getCGALayout());
    } else if (paddedEnc) {
      auto *ctx = srcTy.getContext();
      StringAttr kOffset = StringAttr::get(ctx, "offset");
      auto rank = srcTy.getRank();
      auto offsetBases = sharedLayout.getBases().lookup(kOffset);

      int log2LoadContig = llvm::Log2_32(loadContig);
      int log2ThreadsPerWarp = llvm::Log2_32(threadsPerWarp);
      int log2NumWarps = llvm::Log2_32(numWarps);

      if ((int)offsetBases.size() < log2LoadContig + log2ThreadsPerWarp)
        return rewriter.notifyMatchFailure(
            op, "dst shape is too small. We require at least loadContig * "
                "threadsPerWarp elements");

      auto remainingBases = ArrayRef(offsetBases);
      auto takeN = [&remainingBases](size_t n) {
        auto take = std::min(remainingBases.size(), n);
        auto v = remainingBases.take_front(take).vec();
        remainingBases = remainingBases.drop_front(take);
        return v;
      };

      auto regBases = takeN(log2LoadContig);
      auto laneBases = takeN(log2ThreadsPerWarp);
      auto warpBases = takeN(log2NumWarps);
      warpBases.resize(log2NumWarps, std::vector<int32_t>(rank, 0));
      append_range(regBases, remainingBases);

      triton::LinearLayout newRegLayout(
          {
              {StringAttr::get(ctx, "register"), regBases},
              {StringAttr::get(ctx, "lane"), laneBases},
              {StringAttr::get(ctx, "warp"), warpBases},
          },
          triton::standardOutDimNames(ctx, rank));

      newRegLayout = triton::gpu::combineCtaCgaWithShape(
          newRegLayout, blockedEnc.getCGALayout(), srcTy.getShape());

      auto newRegToShared = newRegLayout.invertAndCompose(sharedLayout);
      if (newRegToShared.getNumConsecutiveInOut() < loadContig)
        return rewriter.notifyMatchFailure(
            op, "could not coalesce global addresses based on the linear "
                "component of the padded encoding");

      newDistEnc = ttg::LinearEncodingAttr::get(ctx, std::move(newRegLayout));
    } else {
      assert(false && "Unsupported layout");
    }

    if (newDistEnc == srcTy.getEncoding())
      return rewriter.notifyMatchFailure(
          op, "Unable to find a new src layout to coalesce writes to LDS");

    auto convertLayout = [&rewriter](auto loc, Value old, auto newEnc) {
      auto oldTy = cast<RankedTensorType>(old.getType());
      RankedTensorType newTy = oldTy.cloneWithEncoding(newEnc);
      return ttg::ConvertLayoutOp::create(rewriter, loc, newTy, old);
    };

    auto loc = op->getLoc();
    Value newSrc = convertLayout(loc, srcTensor, newDistEnc);
    if (mask)
      mask = convertLayout(loc, mask, newDistEnc);
    if (other)
      other = convertLayout(loc, other, newDistEnc);

    rewriter.modifyOpInPlace(op, [&]() {
      Traits::assignSource(op, rewriter, newSrc);
      if (mask)
        op.getMaskMutable().assign(mask);
      if (other)
        op.getOtherMutable().assign(other);
      op.setContiguity(loadContig);
    });
    return success();
  }

private:
  const triton::AMD::TargetInfo &targetInfo;
  const DenseMap<OpTy, unsigned> &contiguityMap;
};

// Convenience aliases.
using CoalesceAsyncCopyWrites =
    CoalesceAsyncCopyToLocal<ttg::AsyncCopyGlobalToLocalOp>;
using CoalesceBufferLoadToLocal =
    CoalesceAsyncCopyToLocal<triton::amdgpu::BufferLoadToLocalOp>;

} // anonymous namespace

class TritonAMDGPUCoalesceAsyncCopyPass
    : public impl::TritonAMDGPUCoalesceAsyncCopyBase<
          TritonAMDGPUCoalesceAsyncCopyPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = &getContext();

    triton::AMD::TargetInfo targetInfo(gfxArch);

    mlir::RewritePatternSet patterns(context);

    if (!llvm::is_contained({AMD::ISAFamily::CDNA3, AMD::ISAFamily::CDNA4},
                            targetInfo.getISAFamily()))
      return; // This pass is CDNA3 and CDNA4 specific.

    // Precompute the contiguity of all async-copy ops before any IR changes to
    // avoid rebuilding ModuleAxisInfoAnalysis on every pattern application.
    AMD::ModuleAxisInfoAnalysis axisAnalysis(m);
    DenseMap<ttg::AsyncCopyGlobalToLocalOp, unsigned> asyncCopyContiguity;
    DenseMap<triton::amdgpu::BufferLoadToLocalOp, unsigned>
        bufferOpsToLocalContiguity;

    m->walk([&](Operation *op) {
      if (auto globalCopyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
        unsigned contiguity =
            mlir::LLVM::AMD::getContiguity(globalCopyOp.getSrc(), axisAnalysis);
        if (auto mask = globalCopyOp.getMask())
          contiguity = std::min<unsigned>(contiguity,
                                          axisAnalysis.getMaskAlignment(mask));
        asyncCopyContiguity.insert({globalCopyOp, contiguity});
      }

      if (auto bufferCopyOp =
              dyn_cast<triton::amdgpu::BufferLoadToLocalOp>(op)) {
        Value ptr = bufferCopyOp.getPtr();
        Value offsets = bufferCopyOp.getOffsets();

        unsigned elemBitWidth = triton::getPointeeBitWidth(ptr.getType());
        unsigned elemNumBytes = std::max(elemBitWidth / 8, 1u);

        // Alignment from the scalar base pointer divisibility.
        unsigned ptrAlign = 1;
        if (auto *ptrInfo = axisAnalysis.getAxisInfo(ptr)) {
          unsigned ptrDivisibility = ptrInfo->getDivisibility(0);
          ptrAlign = std::max(ptrDivisibility / elemNumBytes, 1u);
        }

        // Alignment from the offset tensor's innermost (fast-varying)
        // dimension, derived from axis-info divisibility — NOT capped by
        // sizePerThread.
        unsigned offsetAlign = 1;
        if (auto *offsetInfo = axisAnalysis.getAxisInfo(offsets)) {
          auto contiguityVec = offsetInfo->getContiguity();
          SmallVector<unsigned> offsetOrder =
              getOrderFromContiguity(contiguityVec);
          unsigned innerDim = offsetOrder[0];
          unsigned divisibility = offsetInfo->getDivisibility(innerDim);
          offsetAlign = std::max(divisibility / elemNumBytes, 1u);
        }

        // Cap to the widest vectorised load (128 bits).
        unsigned maxVec = 128 / elemBitWidth;

        // Cap to the number of elements each thread can access in the offsets
        // tensor. Vectorizing beyond what a thread owns is not possible.
        auto offsetsTy = cast<RankedTensorType>(offsets.getType());
        unsigned elemsPerThread =
            triton::gpu::getTotalElemsPerThread(offsetsTy);
        unsigned contiguity =
            std::min({ptrAlign, offsetAlign, maxVec, elemsPerThread});

        // NOTE: We intentionally do NOT cap by getMaskAlignment(mask). The
        // mask for buffer_load_to_local may have a different (smaller) shape
        // than the offsets tensor (e.g., mask_n[:, None] is [BLOCK_N, 1] while
        // offsets is [BLOCK_N, BLOCK_D_Q]). getMaskAlignment would return 1
        // for such a 1-wide mask, incorrectly capping contiguity along the
        // fast (column) dimension.
        bufferOpsToLocalContiguity.insert({bufferCopyOp, contiguity});
      }
    });

    patterns.add<CoalesceAsyncCopyWrites>(targetInfo, asyncCopyContiguity,
                                          context);
    patterns.add<CoalesceBufferLoadToLocal>(
        targetInfo, bufferOpsToLocalContiguity, context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
