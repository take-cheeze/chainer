import ctypes
import cupy
from cupy.core._scalar import get_typename as _get_typename
from cupy.cuda import compiler
import distutils.spawn
import hashlib
import numpy as np
import os
import string
import subprocess


def _get_kernel_params(
        params,  # type: tuple
        args_info,  # type: tuple
):  # type: str
    p = None  # type: ParameterInfo
    ret = []
    for i in range(len(params)):
        p = params[i]
        type, dtype, ndim = args_info[i]
        is_array = type is np.ndarray
        if type is cupy.Indexer:
            t = 'CIndexer<%d>' % ndim
        else:
            t = _get_typename(dtype)
            if is_array:
                t = 'CArray<%s, %d>' % (t, ndim)
        ret.append('{}{} {}{}'.format(
            'const ' if p.is_const else '',
            t,
            '_raw_' if is_array and not p.raw else '',
            p.name))
    return ', '.join(ret)


def get_cache_dir():
    return os.path.join(compiler.get_cache_dir(), 'native')


_empty_file_preprocess_cache = {}
_cached_compiler_executable = None


def _compiler_executable():
    if _cached_compiler_executable is not None:
        return _cached_compiler_executable

    env = os.getenv('CXX', None)
    if env is not None:
        return env

    ret = distutils.spawn.find_executable('gcc') or \
        distutils.spawn.find_executable('clang')
    _cached_compiler_executable = ret
    return ret


def compiler_available():
    return _compiler_executable() is not None


class Function(object):
    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.ptr = getattr(module.dll, name)

    def _launch(self, grid0, grid1, grid2, block0, block1, block2,
                args, shared_mem, stream):
        pargs = []
        kargs = []
        cp = None
        kargs.reserve(len(args))
        for a in args:
            cp = _pointer(a)
            pargs.append(cp)
            kargs.push_back(cp.ptr)

        self.ptr(grid0, grid1, grid2, block0, block1, block2,
                 shared_mem, stream, kargs, 0)

    def __call__(self, grid, block, args, shared_mem=0, stream=None):
        grid = (grid + (1, 1))[:3]
        block = (block + (1, 1))[:3]
        s = _get_stream(stream)
        self._launch(
            max(1, grid[0]), max(1, grid[1]), max(1, grid[2]),
            max(1, block[0]), max(1, block[1]), max(1, block[2]),
            args, shared_mem, s)

    def linear_launch(self, size, args, shared_mem=0,
                      block_max_size=128, stream=None):
        # TODO(beam2d): Tune it
        gridx = min(
            0x7fffffff, (size + block_max_size - 1) // block_max_size)
        blockx = min(block_max_size, size)
        s = _get_stream(stream)
        self._launch(gridx, 1, 1, blockx, 1, 1, args, shared_mem, s)


class Module(object):
    def __init__(self, path):
        self.path = path
        self.dll = ctypes.cdll.LoadLibrary(path)

    def get_function(self, name):
        return Function(name, self)


def compile_with_cache(source, options=[], arch=None, cache_dir=None,
                       extra_source=None):
    # NVRTC does not use extra_source. extra_source is used for cache key.
    global _empty_file_preprocess_cache
    if cache_dir is None:
        cache_dir = get_cache_dir()

    if compiler._get_bool_env_variable('CUPY_CUDA_COMPILE_WITH_DEBUG', False):
        options += ('-g3')

    env = (options,)
    base = _empty_file_preprocess_cache.get(env, None)
    if base is None:
        # This is checking of NVRTC compiler internal version
        proc = subprocess.run(
            [_compiler_executable(), '-E', '-xc++', '-', '-o', '-'] + options,
            input=source, capture_output=True,)
        base = proc.stdout
        _empty_file_preprocess_cache[env] = base
    key_src = '%s %s %s %s' % (env, base, source, extra_source)

    key_src = key_src.encode('utf-8')
    name = '%s_2.cubin' % hashlib.md5(key_src).hexdigest()

    if not os.path.isdir(cache_dir):
        try:
            os.makedirs(cache_dir)
        except OSError:
            if not os.path.isdir(cache_dir):
                raise

    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        subprocess.run([_compiler_executable(), '-xc++', '-', '-shared',
                        '-o', path])

        # Save .cu source file along with .cubin
        if compiler._get_bool_env_variable(
                'CUPY_CACHE_SAVE_CUDA_SOURCE', False):
            with open(path + '.cu', 'w') as f:
                f.write(source)

    return Module(path)


def _get_simple_elementwise_kernel(
        params, operation, name, preamble,
        loop_prep='', after_loop='', options=()
):  # type: function.Function
    module_code = string.Template('''
    ${preamble}
    extern "C" __global__ void ${name}(${params}) {
      ${loop_prep};
      for (size_t i = 0; i < _ind.size(); ++i) {
        _ind.set(i);
        ${operation};
      }
      ${after_loop};
    }
    ''').substitute(
        params=params,
        operation=operation,
        name=name,
        preamble=preamble,
        loop_prep=loop_prep,
        after_loop=after_loop)
    module = compile_with_cache(module_code, options)
    return module.get_function(name)


def _get_elementwise_kernel(
        args_info,  # type: tuple
        types,  # type: tuple
        params,  # type: tuple
        operation, name,
        preamble,
        kwargs  # type: dict
):  # type: function.Function
    kernel_params = _get_kernel_params(params, args_info)
    types_preamble = '\n'.join(
        'typedef %s %s;' % (_get_typename(v), k) for k, v in types)
    preamble = types_preamble + '\n' + preamble

    op = []
    for p, a in zip(params, args_info):
        if not p.raw and a[0] == np.ndarray:
            if p.is_const:
                fmt = 'const {t} &{n} = _raw_{n}[_ind.get()];'
            else:
                fmt = '{t} &{n} = _raw_{n}[_ind.get()];'
            op.append(fmt.format(t=p.ctype, n=p.name))
    op.append(operation)
    operation = '\n'.join(op)
    return _get_simple_elementwise_kernel(
        kernel_params, operation, name,
        preamble, **kwargs)


class ElementwiseKernel(cupy.ElementwiseKernel):
    def _get_elementwise_kernel(self, dev_id, args_info, types):
        key = (dev_id, args_info, types)
        kern = self._kernel_memo.get(key, None)
        if kern is not None:
            return kern
        kern = _get_elementwise_kernel(
            args_info, types, self.params, self.operation,
            self.name, self.preamble, self.kwargs)
        self._kernel_memo[key] = kern
        return kern


def elementwise(in_params, out_params, operation, name, **kwargs):
    if _compiler_executable() is None:
        return None
    return ElementwiseKernel(in_params, out_params, operation, name, **kwargs)
