"""SFFT's solver: a frozen specification of the linear system (spec), spatial bases as data (basis), the CPU engine (engine_cpu) and its
CuPy twin (engine_gpu, imported only on GPU machines), and the readers of a solution (solution). sfft.spaceflow.subtract.solve solves with it on
both backends."""
from .spec import (SFFTSpec, KernelSpec, BackgroundSpec, KernelRegularization,
                   PolynomialBasis, BSplineBasis)
from .engine_cpu import SFFTEngine, SFFTSolution

from .solution import (save_solution, load_solution, kernel_at,
                       flux_scaling_at, background_at, kernel_grid)

__all__ = ['SFFTSpec', 'KernelSpec', 'BackgroundSpec', 'KernelRegularization',
           'PolynomialBasis', 'BSplineBasis', 'SFFTEngine', 'SFFTSolution',
           'save_solution', 'load_solution', 'kernel_at', 'flux_scaling_at',
           'background_at', 'kernel_grid']
