"""Backend of the heavy steps: 'gpu' (CuPy: the cross-convolution and SFFT's GPU engine) or 'cpu' (NumPy, SciPy and numba). The resampling
(drizzle) runs on the CPU with either backend. The CPU backend does not need CuPy; block_cupy() makes CuPy unimportable, so that a run does
what a CPU-only machine does."""
import os
import sys

import numpy as np


def block_cupy():
    """Make CuPy unimportable for this process; call it before anything imports CuPy."""
    for name in ("cupy", "cupyx"):
        if sys.modules.get(name) is not None and name in sys.modules:
            raise RuntimeError(f"{name} is already imported; block_cupy() must run before it")
        sys.modules[name] = None
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def gpu_problem():
    """None when the GPU engines can work here, else the reason. A small FFT and a reduction are run: a driver without a CUDA toolkit fails
    only there (CuPy compiles its own kernels at run time, so no nvcc is needed)."""
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return "CuPy finds no CUDA device"
        a = cupy.fft.rfft2(cupy.ones((16, 16), dtype=cupy.float64))
        float(cupy.sum(cupy.abs(a)))
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:200]}"
    return None


def gpu_available():
    """Whether gpu_problem() finds nothing wrong."""
    return gpu_problem() is None


def resolve(backend):
    """'auto' -> 'gpu' when the GPU works here, else 'cpu'; 'gpu' raises RuntimeError with the reason when it does not work."""
    b = str(backend).lower()
    if b == "auto":
        return "gpu" if gpu_available() else "cpu"
    if b not in ("gpu", "cpu"):
        raise ValueError("backend must be 'gpu', 'cpu' or 'auto'")
    if b == "gpu":
        why = gpu_problem()
        if why:
            raise RuntimeError(f"backend 'gpu' requested, but: {why}. Use 'cpu' or 'auto' to run on the CPU instead.")
    return b


def use_device(backend, device):
    """Make one GPU the default for everything in this process."""
    if backend == "gpu":
        import cupy
        cupy.cuda.Device(int(device)).use()


def xp(backend):
    """The array module of a backend: cupy for 'gpu', numpy otherwise."""
    if backend == "gpu":
        import cupy
        return cupy
    return np


def to_numpy(a):
    """A cupy or numpy array as a numpy array."""
    if hasattr(a, "get"):
        return a.get()
    return np.asarray(a)


def free(backend):
    """Release CuPy's memory pool (backend 'gpu')."""
    if backend == "gpu":
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()


def sfft_backend(backend):
    """The SFFT solver's name of a backend."""
    return "gpu" if backend == "gpu" else "cpu"


def describe():
    """One line on the CuPy version and the CUDA devices this process can use."""
    try:
        import cupy
        n = cupy.cuda.runtime.getDeviceCount()
        names = [cupy.cuda.runtime.getDeviceProperties(i)["name"].decode() for i in range(n)]
        why = gpu_problem()
        return f"CuPy {cupy.__version__}, {n} CUDA device(s): {', '.join(names)}" + (f"; not usable: {why}" if why else "")
    except Exception as e:
        return f"no usable CuPy ({type(e).__name__}): CPU backend only"


def configure_cpu(threads=8, log=print):
    """Settings of the CPU backend: numba's threads (at most numba's pool, fixed at its first import) and the FFT workers of the
    cross-convolution. Returns the number of threads."""
    threads = max(1, int(threads))
    if "numba" not in sys.modules:
        # numba sizes its pool from NUMBA_NUM_THREADS at its first import. Once numba is imported the variable must stay as it is: numba
        # reads it again whenever it compiles a function and raises when it differs from the pool it has launched (a GPU run that
        # imported numba, then a CPU run, then any function compiled afresh, such as the legacy packets with NUMBA_CACHE=False)
        os.environ.setdefault("NUMBA_NUM_THREADS", str(threads))
    import numba
    limit = int(numba.config.NUMBA_NUM_THREADS)
    if threads > limit:
        log(f"WARNING: {threads} threads asked for, but numba's pool has {limit} in this process (set at its first import; restart the kernel or set NUMBA_NUM_THREADS to change it): using {limit}")
        threads = limit
    # the legacy CPU configuration of sfft.legacy.sfftcore assigns a number to numba.set_num_threads, replacing the function for the whole
    # process; numba.np.ufunc.parallel keeps the function itself, so a flow run after a legacy CPU subtraction still sets its threads
    from numba.np.ufunc.parallel import set_num_threads
    set_num_threads(threads)
    from . import crossconv
    crossconv.WORKERS = threads
    log(f"CPU backend: {threads} threads")
    return threads
