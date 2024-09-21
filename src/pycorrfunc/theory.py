from dataclasses import dataclass
from typing import Optional

import numpy as np
import astropy.table

from . import _pycorrfunc, _pycorrfuncf

from .isa import lookup_isa

# import Corrfunc
# import pycorrfunc.compat as Corrfunc

# from Corrfunc.theory.DD import DD
# import pycorrfunc.compat.theory.DD import DD


def DD(
    X1,
    Y1,
    Z1,
    bins,
    *,
    W1=None,
    X2=None,
    Y2=None,
    Z2=None,
    W2=None,
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

    # TODO: support multiple weights per particle,
    # possibly of different types (e.g., DOUBLE and int for bit flags)
    if W1 is not None:
        W1 = np.asarray(W1, dtype=dtype)
    if W2 is not None:
        W2 = np.asarray(W2, dtype=dtype)

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

    module = {np.float32: _pycorrfuncf, np.float64: _pycorrfunc}[dtype.type]

    if num_threads is None:
        num_threads = 0

    isa = lookup_isa(isa)

    module.countpairs(
        X1=X1,
        Y1=Y1,
        Z1=Z1,
        W1=W1,
        X2=X2,
        Y2=Y2,
        Z2=Z2,
        W2=W2,
        bin_edges=bin_edges,
        npairs=npairs,
        ravg=ravg,
        weighted_pairs=weighted_pairs,
        num_threads=num_threads,
        boxsize=boxsize,
        weight_type=weight_type,
        verbose=verbose,
        isa=isa,
    )

    res = astropy.table.Table(
        {
            "npairs": npairs,
            "ravg": ravg,
            "weighted_pairs": weighted_pairs,
        },
        meta={
            "N1": len(X1),
            "N2": len(X2),
            "autocorr": X2 is None,
            "weight_type": weight_type,
            "boxsize": boxsize,
        },
    )

    return res
