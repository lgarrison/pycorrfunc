from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import _corrfunc, _corrfuncf

# import Corrfunc
# import pycorrfunc.compat as Corrfunc

# from Corrfunc.theory.DD import DD
# import pycorrfunc.compat.theory.DD import DD

# TODO: should this be an astropy or pandas table instead?
# TODO: where should we compute xi?
@dataclass
class DDResult:
    npairs: np.ndarray
    ravg: np.ndarray
    weighted_pairs: np.ndarray
    N1: int
    N2: int
    autocorr: bool
    weight_type: Optional[str]
    boxsize: Optional[float]


ISA_LOOKUP = {
    "fastest": -1,
    "fallback": 0,
    "sse": 1,
    "sse2": 2,
    "sse3": 3,
    "ssse3": 4,
    "sse4": 5,
    "sse42": 6,
    "avx": 7,
    "avx2": 8,
    "avx512f": 9,
    "arm64": 10,
}

def lookup_isa(isa):
    isa = isa.lower()
    if isa not in ISA_LOOKUP:
        raise ValueError(f'ISA "{isa}" not recognized')
    return ISA_LOOKUP[isa]

def DD(
    X1,
    Y1,
    Z1,
    bins,
    X2=None,
    Y2=None,
    Z2=None,
    num_threads=None,
    boxsize=None,
    Rmax=None,
    weight_type=None,
    verbose=False,
    isa="fastest",
):
    dtype = X1.dtype
    X1 = np.asarray(X1, dtype=dtype)
    Y1 = np.asarray(Y1, dtype=dtype)
    Z1 = np.asarray(Z1, dtype=dtype)
    X2 = np.asarray(X2, dtype=dtype)
    Y2 = np.asarray(Y2, dtype=dtype)
    Z2 = np.asarray(Z2, dtype=dtype)

    if type(bins) is int:
        if Rmax is None:
            raise ValueError("Rmax should be provided when bins is an integer")
        bin_edges = np.linspace(0, Rmax, bins + 1, dtype=dtype)
    else:
        bin_edges = np.asarray(bins, dtype=dtype)
        if Rmax is not None:
            raise ValueError("Rmax should not be provided when bins is an array")

    npairs = np.zeros(len(bin_edges) - 1, dtype=np.uint64)
    ravg = np.zeros(len(bin_edges) - 1, dtype=dtype)
    weighted_pairs = np.zeros(len(bin_edges) - 1, dtype=dtype)

    module = {np.float32: _corrfuncf, np.float64: _corrfunc}[dtype.type]

    if num_threads is None:
        num_threads = 0

    isa = lookup_isa(isa)

    module.countpairs(
        X1,
        Y1,
        Z1,
        X2,
        Y2,
        Z2,
        bin_edges,
        npairs,
        ravg,
        weighted_pairs,
        num_threads,
        boxsize,
        weight_type,
        verbose,
        isa
    )

    res = DDResult(
        npairs = npairs,
        ravg = ravg,
        weighted_pairs = weighted_pairs,
        N1 = len(X1),
        N2 = len(X2),
        autocorr = X2 is None,
        weight_type = weight_type,
        boxsize = boxsize,
    )

    return res
