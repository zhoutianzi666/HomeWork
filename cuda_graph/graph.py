import paddle
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
import paddle.device.cuda.graphs as graphs
from typing import Callable
import types
import inspect

def cuda_graph_cached(arg_converters, cache_spec_getter=None):
    def decorator(cls_or_fn):
        return _experimental_cuda_graph(cls_or_fn, arg_converters, cache_spec_getter)
    return decorator

def _experimental_cuda_graph(cls_or_fn, arg_converters, cache_spec_getter):
    if inspect.isclass(cls_or_fn):
        return _experimental_cuda_graph_class(cls_or_fn, arg_converters, cache_spec_getter)
    else:
        return _experimental_cuda_graph_func(cls_or_fn, arg_converters, cache_spec_getter)

def _experimental_cuda_graph_class(cls, arg_converters, cache_spec_getter):
    assert hasattr(cls, 'forward')
    if hasattr(cls.forward, '__experimental_cuda_graph_func_flag__'):
        return cls
    return type(cls.__name__, (cls,), {
        'forward': _experimental_cuda_graph_func(cls.forward, arg_converters, cache_spec_getter)
    })

def _experimental_cuda_graph_func(fn, arg_converters, cache_spec_getter):
    def wrapper(*args, **kwargs):
        return _experimental_cuda_graph_impl(fn, args, kwargs, arg_converters, cache_spec_getter)
    wrapper.__original_func__ = fn
    wrapper.__experimental_cuda_graph_func_flag__ = True
    return wrapper

def _experimental_cuda_graph_impl(fn, args, kwargs, arg_converters, cache_spec_getter):
    cache_spec_by_arg_converters = _get_inputs_cache_spec_by_arg_converters(
        args,
        kwargs,
        arg_converters,
        cache_spec_getter
    )
    if not cache_spec_by_arg_converters.enable_cuda_graph:
        return fn(*args, **kwargs)
    fn_key = _get_cache_key(fn, args, cache_spec_by_arg_converters.cache_key)
    cuda_graph_ctx = _get_or_create_cuda_graph_ctx(fn_key)
    cuda_graph_ctx.iter_no += 1

    kWarmupLimits = 3
    if cuda_graph_ctx.iter_no < kWarmupLimits:
        return fn(*args, **kwargs)

    return _capture_then_replay_cuda_graph(cuda_graph_ctx, fn, args, kwargs, arg_converters)

def _get_inputs_cache_spec_by_arg_converters(args, kwargs, arg_converters, cache_spec_getter):
    opt_self, args = _unpack_opt_self_and_args(args)
    def default_cache_spec_getter_impl(*args, **kwargs):
        args_cache_spec = _get_args_cache_spec_by_arg_converters(args, arg_converters)
        kwargs_cache_spec = _get_kwargs_cache_spec_by_arg_converters(kwargs, arg_converters)
        return args_cache_spec * kwargs_cache_spec
    assert len(opt_self) <= 1
    if len(opt_self) == 0:
        def default_cache_spec_getter(*args, **kwargs):
            return default_cache_spec_getter_impl(*args, **kwargs)
    else:
        def default_cache_spec_getter(self, *args, **kwargs):
            return default_cache_spec_getter_impl(*args, **kwargs)

    if cache_spec_getter is None:
        return default_cache_spec_getter(*opt_self, *args, **kwargs)
    else:
        return cache_spec_getter(default_cache_spec_getter, *opt_self, *args, **kwargs)

def _get_args_cache_spec_by_arg_converters(args, arg_converters):
    cache_specs = [
        _get_cache_spec_by_arg_converters(arg, arg_converters)
        for arg in args
    ]
    return CUDAGraphCacheSpec.reduce(cache_specs)

def _get_kwargs_cache_spec_by_arg_converters(kwargs, arg_converters):
    cache_specs = {
        k:_get_cache_spec_by_arg_converters(v, arg_converters)
        for k, v in kwargs.items()
    }
    return CUDAGraphCacheSpec(
        enable_cuda_graph=all(x.enable_cuda_graph for _, x in cache_specs.items()),
        cache_key=tuple((k, v.cache_key) for k, v in cache_specs.items())
    )

def _get_cache_spec_by_arg_converters(arg, arg_converters):
    if type(arg) in arg_converters:
        return _get_cache_spec_by_arg_converter(arg, arg_converters[type(arg)])
    assert "_" in arg_converters,f"{type(arg)=}"
    return _get_cache_spec_by_arg_converter(arg, arg_converters["_"])

def _get_cache_spec_by_arg_converter(arg, arg_converter):
    return arg_converter.get_cuda_graph_cache_spec(arg)

def _get_cache_key(fn, args, cache_keys_by_arg_converters):
    if len(args) < 1:
        return (fn, cache_keys_by_arg_converters)
    self_obj = args[0]
    cls = type(self_obj)
    name = fn.__name__
    if hasattr(cls, name) and fn == getattr(cls, name).__original_func__:
        return (id(self_obj), fn, cache_keys_by_arg_converters)
    return (fn, cache_keys_by_arg_converters)


@dataclass
class CudaGraphCtx:
    iter_no: int = 0
    cuda_graph: graphs.CUDAGraph | None = None
    args_buffer: list[paddle.Tensor] = field(default_factory=list)
    kwargs_buffer: dict[str, paddle.Tensor] = field(default_factory=dict)
    output_buffer: paddle.Tensor | list[paddle.Tensor] | None = None

# 全局上下文表
_g_fn_key2cuda_graph_ctx : dict[Callable, CudaGraphCtx] = {}

# 上下文创建或获取
def _get_or_create_cuda_graph_ctx(fn_key):
    global _g_fn_key2cuda_graph_ctx
    if fn_key not in _g_fn_key2cuda_graph_ctx:
        cuda_graph_ctx = CudaGraphCtx()
        _g_fn_key2cuda_graph_ctx[fn_key] = cuda_graph_ctx
    return _g_fn_key2cuda_graph_ctx[fn_key]

# 捕获和重放图执行
def _capture_then_replay_cuda_graph(cuda_graph_ctx, fn, args, kwargs, arg_converters):
    # print("type(args)", [type(x) for x in args])
    opt_self, args = _unpack_opt_self_and_args(args)
    # print("type(opt_self)", [type(x) for x in opt_self])
    # print("type(args)", [type(x) for x in args])
    _try_recapture(cuda_graph_ctx, fn, opt_self, args, kwargs, arg_converters)
    _feed_inputs(cuda_graph_ctx, fn, args, kwargs, arg_converters)
    cuda_graph_ctx.cuda_graph.replay()
    print(f"-------[CUDA GRAPH] cuda graph replay")
    return _clone(cuda_graph_ctx.output_buffer)

def _unpack_opt_self_and_args(args):
    if len(args) < 1:
        return (), args
    _self = args[0]
    if not hasattr(_self, "forward"):
        return (), args
    
    if not hasattr(_self.forward, '__experimental_cuda_graph_func_flag__'):
        return (), args
    return args[0:1], args[1:]

def _feed_inputs(cuda_graph_ctx, fn, args, kwargs, arg_converters):
    for i, arg in enumerate(args):
        _select_then_feed_input(arg, cuda_graph_ctx.args_buffer[i], arg_converters)
    for k, v in kwargs.items():
        _select_then_feed_input(v, cuda_graph_ctx.kwargs_buffer[k], arg_converters)

def _select_then_feed_input(arg, buffered_arg, arg_converters):
    if type(arg) in arg_converters:
        return _feed_input(arg, buffered_arg, arg_converters[type(arg)])
    assert "_" in arg_converters
    return _feed_input(arg, buffered_arg, arg_converters["_"])

def _feed_input(arg, buffered_arg, arg_converter):
    return arg_converter.feed_as_arg_in_cuda_graph(arg, buffered_arg)

def _try_recapture(cuda_graph_ctx, fn, opt_self, args, kwargs, arg_converters):
    ctx = cuda_graph_ctx
    if not _need_recapture(ctx, fn, args, kwargs, arg_converters):
        return

    ctx.cuda_graph = graphs.CUDAGraph(pool_id=_get_global_poold_id())
    # ctx.cuda_graph = graphs.CUDAGraph()
    print(f"-------[CUDA GRAPH] cuda graph capture start")
    _reallocate_inputs(ctx, fn, args, kwargs, arg_converters)
    _feed_inputs(ctx, fn, args, kwargs, arg_converters)
    ctx.cuda_graph.capture_begin()
    ctx.output_buffer = fn(*opt_self, *ctx.args_buffer, **ctx.kwargs_buffer)
    # ctx.output_buffer = fn(*opt_self, *args, **kwargs)
    ctx.cuda_graph.capture_end()
    print(f"-------[CUDA GRAPH] cuda graph capture end")

def _reallocate_inputs(cuda_graph_ctx, fn, args, kwargs, arg_converters):
    cuda_graph_ctx.args_buffer = [
        _select_conveter_then_reallocate_arg(arg, arg_converters)
        for arg in args
    ]
    cuda_graph_ctx.kwargs_buffer = {
        k:_select_conveter_then_reallocate_arg(v, arg_converters)
        for k, v in kwargs.items()
    }

def _select_conveter_then_reallocate_arg(arg, arg_converters):
    if type(arg) in arg_converters:
        return _reallocate_arg(arg, arg_converters[type(arg)])
    assert "_" in arg_converters,f"{type(arg)=}"
    return _reallocate_arg(arg, arg_converters["_"])

def _reallocate_arg(arg, arg_converter):
    return arg_converter.reallocate_as_arg_in_cuda_graph(arg)

def _need_recapture(cuda_graph_ctx, fn, args, kwargs, arg_converters):
    if cuda_graph_ctx.cuda_graph is None:
        print("-------[INFO] cuda_graph_ctx.cuda_graph is None, need recapture")
        return True
    if len(cuda_graph_ctx.args_buffer) != len(args):
        print("-------[INFO] len(cuda_graph_ctx.args_buffer) != len(args), need recapture")
        return True
    if len(cuda_graph_ctx.kwargs_buffer) != len(kwargs):
        print("-------[INFO] len(cuda_graph_ctx.kwargs_buffer) != len(kwargs), need recapture")
        return True
    for i, arg in enumerate(args):
        if _need_reallocate(arg, cuda_graph_ctx.args_buffer[i], arg_converters):
            print("-------[INFO] _need_reallocate(arg, cuda_graph_ctx.args_buffer[i], arg_converters), need recapture")
            return True
    for k, v in kwargs.items():
        if k not in cuda_graph_ctx.kwargs_buffer:
            print("-------[INFO] k not in cuda_graph_ctx.kwargs_buffer, need recapture")
            return True
        if _need_reallocate(v, cuda_graph_ctx.kwargs_buffer[k], arg_converters):
            print("-------[INFO] _need_reallocate(v, cuda_graph_ctx.kwargs_buffer[k], arg_converters), need recapture")
            return True
    return False

def _need_reallocate(arg, buffered_arg, arg_converters):
    if type(arg) in arg_converters:
        return _is_arg_need_reallocate(arg, buffered_arg, arg_converters[type(arg)])
    assert "_" in arg_converters
    return _is_arg_need_reallocate(arg, buffered_arg, arg_converters["_"])

def _is_arg_need_reallocate(arg, buffered_arg, arg_converter):
    # print("-------[INFO] arg, " , arg)
    # print("-------[INFO] arg.type, " , arg.type)
    # print("-------[INFO] buffered_arg, " , buffered_arg )
    return arg_converter.need_reallocate_as_arg_in_cuda_graph(arg, buffered_arg)

def _clone(result):
    if isinstance(result, (list, tuple)):
        return type(result)(_clone(x) for x in result)
    else:
        return result.clone() if result._is_initialized() else result

_g_global_pool_id = None


def _get_global_poold_id():
    global _g_global_pool_id
    if _g_global_pool_id is None:
        from paddle.base.core import CUDAGraph as CoreCUDAGraph
        _g_global_pool_id = CoreCUDAGraph.gen_new_memory_pool_id()
    return _g_global_pool_id

@dataclass
class CUDAGraphCacheSpec:
    enable_cuda_graph: bool = True
    cache_key: tuple = ()

    def __mul__(self, other):
        return CUDAGraphCacheSpec(
            enable_cuda_graph=self.enable_cuda_graph and other.enable_cuda_graph,
            cache_key=(*self.cache_key, *other.cache_key)
        )

    @staticmethod
    def reduce(cache_specs):
        if len(cache_specs) == 0:
            return CUDAGraphCacheSpec()
        enable_cuda_graph = all(x.enable_cuda_graph for x in cache_specs)
        cache_key = tuple(x.cache_key for x in cache_specs)
        return CUDAGraphCacheSpec(
            enable_cuda_graph=enable_cuda_graph,
            cache_key=cache_key,
        )

class PaddleTensorConverter:
    def feed_as_arg_in_cuda_graph(self, arg, buffered_arg) -> None:
        paddle.assign(arg, output=buffered_arg)

    def need_reallocate_as_arg_in_cuda_graph(self, arg, buffered_arg) -> bool:
        shape_not_equal = (arg.shape != buffered_arg.shape)
        print(f"{arg.shape=}, {buffered_arg.shape=}, {shape_not_equal=}")
        return shape_not_equal

    def reallocate_as_arg_in_cuda_graph(self, arg) -> paddle.Tensor:
        return paddle.empty_like(arg)

    def get_cuda_graph_cache_spec(self, arg) -> CUDAGraphCacheSpec:
        return CUDAGraphCacheSpec(
            enable_cuda_graph=True,
            cache_key=tuple(arg.shape)
        )

class NopeConverter:

    def feed_as_arg_in_cuda_graph(self, arg, buffered_arg):
        pass

    def need_reallocate_as_arg_in_cuda_graph(self, arg, buffered_arg):
        return False

    def reallocate_as_arg_in_cuda_graph(self, arg):
        return arg

    def get_cuda_graph_cache_spec(self, arg) -> CUDAGraphCacheSpec:
        return CUDAGraphCacheSpec(
            enable_cuda_graph=True,
            cache_key=(arg,)
        )

BaseArgConverters = {
    paddle.Tensor: PaddleTensorConverter(),
    types.NoneType: NopeConverter(),
    bool: NopeConverter(),
    int: NopeConverter(),
    float: NopeConverter(),
    str: NopeConverter(),
}

def test():

    def cache_spec_getter(default_cache_spec_getter, self, *args, **kwargs):
        return default_cache_spec_getter(self, *args, **kwargs)

    @cuda_graph_cached(BaseArgConverters, cache_spec_getter=cache_spec_getter)
    class Bar:
        def forward(self, x):
            return x + 1

    def TestExperimentalCudaGraph(bar):
        logits = paddle.empty([1, 64])
        for i in range(10):
            print(bar.forward(logits))

    bar0 = Bar()
    TestExperimentalCudaGraph(bar0)
    bar1 = Bar()
    TestExperimentalCudaGraph(bar1)    

ArgConverters = BaseArgConverters

if __name__ == '__main__':
    test()