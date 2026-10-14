import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    # isCloseVectorization must stay False for the min/max sat-mul body below:
    # with it enabled the min/max ops measure ~27x slower than without (A/B
    # measured, 16.7M fp32: 11432us vs 135us). (The previous bit-compare +
    # tl.where body happened to profit from vectorization; the sat-mul body
    # does not, so this knob is body-specific.)
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


# ---------------------------------------------------------------------------
# Fast path (contiguous float tensors above _MIN_FAST_NUMEL).
#
# The original implementation classified NaN / +-inf on the *bit pattern* of
# the fp32 image of x:
#
#     x_bits = x.to(tl.float32).to(tl.int32, bitcast=True)
#     x_nan = (x_bits & 0x7FFFFFFF) > 0x7F800000
#     x_posinf = x_bits == 0x7F800000
#     x_neginf = x_bits == 0xFF800000
#
# `arith::CmpIOp` has no 512-bit vector form in this backend
# (VectorizabilityAnalysis.h, TTX_SCALAR_TO_VECTOR_OPS), and the closure walk is
# all-or-nothing: one integer comparison drags the load, the arithmetic and the
# store onto the scalar path.  Measured on a 16.7M-element fp32 tile the
# bit-pattern kernel runs at 302 GB/s while the same kernel with float
# comparisons runs at 741 GB/s and a pure copy at 1108 GB/s.
#
# Float classification uses only ops that do have vector forms here:
# `arith::CmpFOp` and `arith::SelectOp` (f16/f32/i32).  `x != x` is the NaN test
# -- no libdevice `isnan` extern, which is scalar for the same reason -- and
# +-inf are plain float comparisons.
#
# When posinf/neginf are the dtype extremes (the `None` defaults, i.e. exactly
# what a bare `nan_to_num(x)` call produces) the two inf tests collapse to
# `minimum`/`maximum`, also vector ops, cutting the op count further:
#   min(x, finfo.max) maps +inf -> finfo.max and is the identity on every finite
#   value; max(x, finfo.min) likewise for -inf.  That identity only holds when
#   the replacement is >= every finite value (resp. <=), so it is gated on the
#   host side and the general float-compare kernel is used otherwise.
#
# All arithmetic runs in fp32: the `to(fp32)` / `to(x.dtype)` round trip is free
# (measured identical to a pure copy), while native fp16/bf16 comparisons fail
# to lower in this backend and bf16 has no vector `select` form.
# ---------------------------------------------------------------------------

_MIN_FAST_NUMEL = 1024
_BLOCK_BY_ELEMSIZE = {2: 16384, 4: 8192, 8: 4096}
_FAST_WARPS = 16


@triton.jit
def _nan_to_num_body(x, nan, posinf, neginf, USE_MINMAX: tl.constexpr):
    if USE_MINMAX:
        # The NaN test reads the original x, so it is independent of the
        # clamp chain (better ILP) and the result is correct whichever way
        # minimum/maximum treat a NaN operand.
        m_nan = x != x
        x = tl.minimum(x, posinf)
        x = tl.maximum(x, neginf)
        x = tl.where(m_nan, nan, x)
    else:
        m_nan = x != x
        m_pos = x == float("inf")
        m_neg = x == float("-inf")
        x = tl.where(m_nan, nan, x)
        x = tl.where(m_pos, posinf, x)
        x = tl.where(m_neg, neginf, x)
    return x


@triton.jit
def _nan_to_num_kernel(
    x_ptr, y_ptr, nan, posinf, neginf, USE_MINMAX: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    x = _nan_to_num_body(x, nan, posinf, neginf, USE_MINMAX)
    tl.store(y_ptr + offs, x.to(y_ptr.dtype.element_ty))


# Tail kernel: the last program is slid back so that its tile is fully in range.
# No load/store mask is needed -- the overlap with the previous tile is
# rewritten with the same value, and the elementwise op is idempotent.  The
# caller guarantees numel >= BLOCK.
@triton.jit
def _nan_to_num_tail_kernel(
    x_ptr,
    y_ptr,
    numel,
    nan,
    posinf,
    neginf,
    USE_MINMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    start = tl.minimum(tl.program_id(0) * BLOCK, numel - BLOCK)
    offs = start + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    x = _nan_to_num_body(x, nan, posinf, neginf, USE_MINMAX)
    tl.store(y_ptr + offs, x.to(y_ptr.dtype.element_ty))


# Generic path: 0-dim, tiny, non-contiguous and non-float inputs keep the
# original implementation unchanged.
@pointwise_dynamic(
    is_tensor=[True, False, False, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def nan_to_num_func(x, nan, posinf, neginf):
    # Classify x into {finite, +inf, -inf, NaN} with pure saturated FP
    # arithmetic (min/max/mul only). The previous bit-compare + tl.where body
    # lowers to vcmp/vselect sequences that measure ~2.6x slower than the
    # pure-fp classify on this backend (same class of slow path as hardshrink /
    # not_equal). Each of m_pos / m_neg / m_fin is exactly 0.0 or 1.0:
    #   m_pos = 1 iff x == +inf;  m_neg = 1 iff x == -inf;
    #   m_fin = 1 iff x is finite (incl. +/-0.0, subnormals, +/-f32 max).
    # min/max here are non-NaN-propagating (maxnum/minnum), so the NaN case
    # falls out as m_fin == m_pos == m_neg == 0 -> m_nan == 1.
    # NOTE: all constants below are f32-inline literals; module-level
    # tl.constexpr Python floats are f64 on this backend and would promote the
    # whole body to the software-emulated f64 (a >10x regression).
    M = 3.4028234663852886e38  # exact f32 max finite
    m_pos = tl.minimum(tl.maximum((x - M) * 1e30 * 1e30, 0.0), 1.0)
    m_neg = tl.minimum(tl.maximum((0.0 - x - M) * 1e30 * 1e30, 0.0), 1.0)
    # (x - x) is exactly +0.0 for finite x and NaN for +/-inf/NaN input.
    m_fin = tl.minimum(tl.maximum(1.0 + (x - x) * 1e30 * 1e30, 0.0), 1.0)
    m_nan = 1.0 - (m_fin + m_pos + m_neg)
    # Blend with a nested select on the raw input -- never x in fp arithmetic:
    # this backend flushes subnormal f32/bf16 inputs to +0.0 in the fp add
    # (FTZ; verified: x + 0.0 == 0.0 for subnormal x), so the multiplied-then-
    # summed blend loses subnormal finite values. The masks are exactly 0/1
    # and mutually exclusive, so the select chain is a straight 4-way pick.
    return tl.where(
        m_fin > 0.5,
        x,
        tl.where(
            m_nan > 0.5,
            nan,
            tl.where(m_pos > 0.5, posinf, tl.where(m_neg > 0.5, neginf, 0.0)),
        ),
    )


def _nan_to_num_impl(A, nan, posinf, neginf, out=None):
    numel = A.numel()
    if (
        A.is_floating_point()
        and A.is_contiguous()
        and A.dim() > 0
        and numel >= _MIN_FAST_NUMEL
    ):
        if out is None:
            out = torch.empty_like(A)
        block = min(
            _BLOCK_BY_ELEMSIZE.get(A.element_size(), 8192),
            triton.next_power_of_2(numel),
        )
        finfo = torch.finfo(A.dtype)
        # minimum/maximum only reproduce the +-inf replacement when the target
        # is the largest / smallest value of the dtype, and the NaN filler must
        # survive the clamp unchanged.
        use_minmax = (
            posinf >= finfo.max and neginf <= finfo.min and neginf <= nan <= posinf
        )
        grid = (triton.cdiv(numel, block),)
        if numel % block == 0:
            _nan_to_num_kernel[grid](
                A,
                out,
                nan,
                posinf,
                neginf,
                USE_MINMAX=use_minmax,
                BLOCK=block,
                num_warps=_FAST_WARPS,
            )
        else:
            _nan_to_num_tail_kernel[grid](
                A,
                out,
                numel,
                nan,
                posinf,
                neginf,
                USE_MINMAX=use_minmax,
                BLOCK=block,
                num_warps=_FAST_WARPS,
            )
        return out
    if out is None:
        return nan_to_num_func(A, nan, posinf, neginf)
    nan_to_num_func(A, nan, posinf, neginf, out0=out)
    return out


def nan_to_num(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_KUNLUNXIN NAN_TO_NUM")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    return _nan_to_num_impl(A, nan, posinf, neginf)


def nan_to_num_(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_KUNLUNXIN NAN_TO_NUM_")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    return _nan_to_num_impl(A, nan, posinf, neginf, out=A)
