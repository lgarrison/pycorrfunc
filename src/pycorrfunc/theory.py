from dataclasses import dataclass

import numpy as np

from . import _corrfunc, _corrfuncf


@dataclass
class DDResult:
    npairs: np.ndarray
    ravg: np.ndarray
    weighted_pairs: np.ndarray


def DD(
    X1,
    Y1,
    Z1,
    bins,
    X2 = None,
    Y2 = None,
    Z2 = None,
    num_threads = None,
    boxsize = None,
    Rmax = None,
):
    dtype = X1.dtype
    X1 = np.asarray(X1, dtype=dtype)
    Y1 = np.asarray(Y1, dtype=dtype)
    Z1 = np.asarray(Z1, dtype=dtype)
    X2 = np.asarray(X2, dtype=dtype)
    Y2 = np.asarray(Y2, dtype=dtype)
    Z2 = np.asarray(Z2, dtype=dtype)

    if type(bins) is int:
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
    )

    return DDResult(npairs, ravg, weighted_pairs)
