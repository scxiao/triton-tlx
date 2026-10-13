"""Standalone dependencies for bench_tlx_kda.py.

Extracted from sglang so the benchmark can run with only PYTHONPATH pointing at
the sglang kernel sources, without a full sglang pip install.
"""

from __future__ import annotations

import builtins
import contextlib
import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, NamedTuple, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# From sglang.kernels.jit.utils.common: cache_once
# ---------------------------------------------------------------------------

_TYPE_LIST = (bool, int, float, str, torch.dtype, torch.device, None.__class__)


# ---------------------------------------------------------------------------
# From sglang.kernels.jit.benchmark.marker: do_bench and helpers
# ---------------------------------------------------------------------------


class BenchResult(NamedTuple):
    metrics: Tuple[Any, ...]
    times: List[float]
    memory_footprint: Optional[int]
    flops: Optional[float] = None


@functools.lru_cache(maxsize=None)
def _get_benchmark_stream(device_id: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device_id)


def _clone_recursive(in_: Any) -> Any:
    if isinstance(in_, torch.Tensor):
        return in_.clone()
    elif isinstance(in_, (list, tuple)):
        return type(in_)(_clone_recursive(x) for x in in_)
    elif isinstance(in_, dict):
        return {k: _clone_recursive(v) for k, v in in_.items()}
    elif isinstance(in_, _TYPE_LIST):
        return in_
    raise ValueError(f"unsupported type: {type(in_)}")


def _get_nbytes_recursive(in_: Any) -> int:
    if isinstance(in_, torch.Tensor):
        return in_.nbytes
    elif isinstance(in_, (list, tuple)):
        return sum(_get_nbytes_recursive(x) for x in in_)
    elif isinstance(in_, dict):
        return sum(_get_nbytes_recursive(v) for v in in_.values())
    elif isinstance(in_, _TYPE_LIST):
        return 0
    raise ValueError(f"unsupported type: {type(in_)}")


def _process_metrics(times: list[float], metrics: tuple) -> list[float]:
    results: list[float] = []
    times = sorted(x / 1000 for x in times)
    for metric in metrics:
        if metric == "avg":
            results.append(sum(times) / len(times))
        else:
            assert 0 <= metric <= 1, f"invalid metric: {metric}"
            which = min(int(len(times) * metric), len(times) - 1)
            results.append(times[which])
    return results


@functools.lru_cache(maxsize=None)
def _get_l2_cache_size() -> int:
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.L2_cache_size


_L2_SAFE_RATIO = 5


def _get_flush_l2_buffer() -> torch.Tensor:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    l2_size = _get_l2_cache_size()
    safe_size = int(l2_size * _L2_SAFE_RATIO)
    return torch.empty(safe_size, device=device, dtype=torch.uint8)


def _calculate_rotation_count(nbytes: int, min_rotations: int = 2) -> int:
    l2_size = _get_l2_cache_size()
    safe_cache_threshold = l2_size * _L2_SAFE_RATIO
    if nbytes <= 0 or nbytes >= safe_cache_threshold:
        return 1
    num_rotations = math.ceil(safe_cache_threshold / nbytes) + 1
    return max(min_rotations, num_rotations)


def _do_bench_internal_graph(
    fn: Callable,
    replay_iters: int,
    input_args: Tuple[Any, ...],
    input_kwargs: Dict[str, Any],
    graph_clone_args: Iterable[int],
    graph_clone_kwargs: Iterable[str],
    graph_context,
    sync_multigpu_fn: Callable[[], Any],
) -> List[float]:
    result: List[float] = []
    stream = torch.cuda.current_stream()
    empty_tensor = _get_flush_l2_buffer()
    nbytes = sum(_get_nbytes_recursive(input_args[i]) for i in graph_clone_args)
    nbytes += sum(_get_nbytes_recursive(input_kwargs[k]) for k in graph_clone_kwargs)
    rotate_count = min(_calculate_rotation_count(nbytes), 100)
    loop_count = math.ceil(100 / rotate_count) * rotate_count
    input_args_list = [input_args] * rotate_count
    input_kwargs_list = [input_kwargs] * rotate_count
    graph_clone_args = set(graph_clone_args)
    graph_clone_kwargs = set(graph_clone_kwargs)

    graph = torch.cuda.CUDAGraph()
    for i in builtins.range(1, rotate_count):
        input_args_list[i] = tuple(
            (
                _clone_recursive(input_args[j])
                if j in graph_clone_args
                else input_args[j]
            )
            for j in builtins.range(len(input_args))
        )
        input_kwargs_list[i] = dict(
            (k, (_clone_recursive(v) if k in graph_clone_kwargs else v))
            for k, v in input_kwargs.items()
        )
    with graph_context:
        with torch.cuda.graph(graph, stream=stream):
            for i in builtins.range(loop_count):
                args = input_args_list[i % rotate_count]
                kwargs = input_kwargs_list[i % rotate_count]
                fn(*args, **kwargs)

    graph.replay()
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    for _ in builtins.range(max(replay_iters // loop_count, 10)):
        empty_tensor.zero_()
        sync_multigpu_fn()
        tic.record(stream)
        graph.replay()
        toc.record(stream)
        stream.synchronize()
        result.append(tic.elapsed_time(toc) / loop_count)
    return result


def do_bench(
    fn: Callable,
    *,
    input_args: Tuple[Any, ...] = (),
    input_kwargs: Dict[str, Any] = {},
    use_cuda_graph: bool = True,
    warmup_iters: int = 20,
    replay_iters: int = 1000,
    metrics: Tuple = (0.5, "avg"),
    stream: torch.cuda.Stream | None = None,
    graph_clone_args: Iterable[int] | Literal["all"] | None = "all",
    graph_clone_kwargs: Iterable[str] | Literal["all"] | None = "all",
    disable_log_bandwidth: bool = False,
    memory_args: Iterable[Any] | Literal["all"] | None = "all",
    memory_output: Iterable[Any] | Literal["out"] | None = "out",
    extra_memory_args: Iterable[Any] | None = None,
    extra_memory_footprint: int = 0,
    flops: Optional[float] = None,
    graph_context_fn: Optional[Callable] = None,
    sync_multigpu_fn: Optional[Callable[[], Any]] = None,
    estimated_time_ms: Optional[float] = 1.0,
) -> BenchResult:
    device_id = torch.cuda.current_device()
    if stream is None:
        stream = _get_benchmark_stream(device_id)
    old_current_stream = torch.cuda.current_stream(device_id)
    result: List[float] = []
    sync_multigpu_fn = sync_multigpu_fn or (lambda: None)
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        stream.wait_stream(old_current_stream)
        sync_multigpu_fn()
        for _ in builtins.range(warmup_iters):
            fn(*input_args, **input_kwargs)
        if use_cuda_graph:
            if graph_clone_args == "all":
                graph_clone_args = builtins.range(len(input_args))
            elif graph_clone_args is None:
                graph_clone_args = []
            if graph_clone_kwargs == "all":
                graph_clone_kwargs = input_kwargs.keys()
            elif graph_clone_kwargs is None:
                graph_clone_kwargs = []
            graph_context = (
                graph_context_fn()
                if graph_context_fn is not None
                else contextlib.nullcontext()
            )
            result = _do_bench_internal_graph(
                fn,
                replay_iters,
                input_args,
                input_kwargs,
                graph_clone_args,
                graph_clone_kwargs,
                graph_context,
                sync_multigpu_fn,
            )
        else:
            tic = torch.cuda.Event(enable_timing=True)
            toc = torch.cuda.Event(enable_timing=True)
            empty_tensor = _get_flush_l2_buffer()
            if estimated_time_ms is not None:
                empty_tensor.zero_()
                sync_multigpu_fn()
                tic.record(stream)
                fn(*input_args, **input_kwargs)
                toc.record(stream)
                stream.synchronize()
                duration_ms = tic.elapsed_time(toc)
                estimted_iters = int(estimated_time_ms / duration_ms)
                replay_iters = min(replay_iters, estimted_iters)

            for _ in builtins.range(max(replay_iters, 10)):
                empty_tensor.zero_()
                sync_multigpu_fn()
                tic.record(stream)
                fn(*input_args, **input_kwargs)
                toc.record(stream)
                stream.synchronize()
                result.append(tic.elapsed_time(toc))

    stream.synchronize()
    result = _process_metrics(result, metrics)
    memory_footprint = None
    if not disable_log_bandwidth:
        if memory_args == "all":
            memory_args = input_args + tuple(input_kwargs.values())
        if memory_output == "out":
            memory_output = fn(*input_args, **input_kwargs)
        memory_footprint = extra_memory_footprint
        memory_footprint += _get_nbytes_recursive(extra_memory_args)
        memory_footprint += _get_nbytes_recursive(memory_args)
        memory_footprint += _get_nbytes_recursive(memory_output)

    return BenchResult(metrics, result, memory_footprint, flops)


# ---------------------------------------------------------------------------
# From test_kimi_k3_kda_tlx.py: test data construction
# ---------------------------------------------------------------------------

_HEADS = 12
_DIM = 128

_DEVICE = torch.device("cuda")


def _split_qkv(packed: torch.Tensor):
    q, k, v = packed.split(_HEADS * _DIM, dim=-1)
    return tuple(t.unflatten(-1, (_HEADS, _DIM)).unsqueeze(0) for t in (q, k, v))


@dataclass
class Case:
    qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    pool: torch.Tensor
    cache_indices: torch.Tensor
    query_start_loc: torch.Tensor


def _make_case(
    seq_lens: list[int],
    *,
    decode: bool,
    beta_is_raw: bool = True,
    a_log_mean: float = -4.0,
    seed: int = 20260929,
) -> Case:
    gen = torch.Generator(device=_DEVICE).manual_seed(seed + sum(seq_lens))
    tokens = sum(seq_lens)
    width = _HEADS * _DIM

    def randn(*shape, dtype=torch.bfloat16, scale=1.0):
        return scale * torch.randn(shape, dtype=dtype, device=_DEVICE, generator=gen)

    if decode:
        a = randn(tokens, width)
        b = randn(1, tokens, _HEADS)
    else:
        a = randn(1, tokens, _HEADS, _DIM)
        b = randn(1, tokens, _HEADS, dtype=torch.float32)
        if not beta_is_raw:
            b = b.sigmoid()

    slots = len(seq_lens) + 3
    cache_indices = (
        torch.randperm(slots - 1, device=_DEVICE, generator=gen)[: len(seq_lens)] + 1
    ).to(torch.int32)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=_DEVICE)
    query_start_loc = torch.nn.functional.pad(lens.cumsum(0, dtype=torch.int32), (1, 0))

    return Case(
        qkv=randn(tokens, 3 * width),
        a=a,
        b=b,
        A_log=a_log_mean + randn(_HEADS, dtype=torch.float32, scale=0.1),
        dt_bias=randn(width, dtype=torch.float32, scale=0.1),
        pool=randn(slots, _HEADS, _DIM, _DIM, dtype=torch.float32, scale=0.02),
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )


# ---------------------------------------------------------------------------
# Standalone TlxKDAKernel (no TritonKDAKernel inheritance)
# ---------------------------------------------------------------------------


class TlxKDAKernel:
    """Minimal standalone KDA kernel that calls TLX directly.

    Unlike the full sglang TlxKDAKernel (which inherits TritonKDAKernel and
    falls back to FLA), this only implements the TLX-accelerated paths that
    the benchmark exercises.
    """

    def __init__(self):
        from sglang.kernels.ops.kimi_k3 import tlx as tlx_kernels

        if not tlx_kernels.is_available():
            raise ValueError("KDA TLX backend needs a TLX-capable triton")
        self._tlx = tlx_kernels
        self._prefill_ok = tlx_kernels.is_prefill_available()

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        from sglang.kernels.ops.kimi_k3.tlx.kimi_k3_kda_prepare import (
            prepare_kda_inputs,
        )

        HV, V, K = ssm_states.shape[-3:]
        q_n, k_n, g, beta = prepare_kda_inputs(
            q, k, a, b, A_log, dt_bias, num_heads=HV, head_dim=K,
            lower_bound=lower_bound, sigmoid_beta=True,
        )
        return self._tlx.kda_recurrent_decode(
            q_n, k_n, v, g, beta,
            scale=K**-0.5,
            state_pool=ssm_states,
            read_indices=cache_indices,
            write_indices=cache_indices,
            cu_seqlens=query_start_loc,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        beta_is_raw: bool = False,
        **kwargs,
    ):
        from sglang.kernels.ops.kimi_k3.tlx.kimi_k3_kda_prepare import (
            prepare_kda_inputs,
        )

        HV, V, K = ssm_states.shape[-3:]
        q_n, k_n, log_g, beta_act = prepare_kda_inputs(
            q, k, g, beta, A_log, dt_bias, num_heads=HV, head_dim=K,
            lower_bound=lower_bound, sigmoid_beta=beta_is_raw,
        )
        slots = cache_indices.long()
        initial_state = ssm_states[slots]
        out, final_state = self._tlx.kda_paged_prefill(
            q_n, k_n, v, log_g, beta_act,
            scale=K**-0.5,
            initial_state=initial_state,
            cu_seqlens=query_start_loc,
        )
        ssm_states[slots] = final_state
        return out


# Expose do_bench and _get_l2_cache_size as a marker-like namespace for the
# bench script's monkey-patching of the L2 size override.
class _Marker:
    do_bench = staticmethod(do_bench)
    _get_l2_cache_size = staticmethod(_get_l2_cache_size)


marker = _Marker()
