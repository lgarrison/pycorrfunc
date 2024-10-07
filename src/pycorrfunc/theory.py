import astropy.table
import numpy as np

from .isa import _lookup_isa

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
    weight_method=None,
    verbose=False,
    isa='fastest',
    do_ravg=False,
    dtype=np.float64,
    **kwargs,
):
    grid_refine = kwargs.pop('grid_refine', None)
    max_cells = kwargs.pop('max_cells', None)
    accum_dtype = kwargs.pop('accum_dtype', None)

    if kwargs:
        raise TypeError(f'Unknown keyword arguments: {list(kwargs)}')
    
    if max_cells is None:
        max_cells = 500

    dtype = np.dtype(dtype)
    if dtype.type == np.float64:
        from . import _pycorrfunc
        module = _pycorrfunc
    elif dtype.type == np.float32:
        from . import _pycorrfuncf
        module = _pycorrfuncf
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')

    if accum_dtype is None:
        accum_dtype = np.float64 if module.sizeof_DoubleAccum == 8 else np.float32

    X1 = np.ascontiguousarray(X1, dtype=dtype)
    Y1 = np.ascontiguousarray(Y1, dtype=dtype)
    Z1 = np.ascontiguousarray(Z1, dtype=dtype)

    if X2 is not None:
        X2 = np.ascontiguousarray(X2, dtype=dtype)
    if Y2 is not None:
        Y2 = np.ascontiguousarray(Y2, dtype=dtype)
    if Z2 is not None:
        Z2 = np.ascontiguousarray(Z2, dtype=dtype)

    # TODO: support multiple weights per particle,
    # possibly of different types (e.g., DOUBLE and int for bit flags)
    if W1 is not None:
        W1 = np.ascontiguousarray(W1, dtype=dtype)
    if W2 is not None:
        W2 = np.ascontiguousarray(W2, dtype=dtype)

    if boxsize is not None:
        boxsize = np.broadcast_to(np.asarray(boxsize, dtype=dtype), (3,))

    if type(bins) is int:
        if Rmax is None:
            raise ValueError('Rmax should be provided when bins is an integer')
        bin_edges = np.linspace(0, Rmax, bins + 1, dtype=dtype)
    else:
        bin_edges = np.ascontiguousarray(bins, dtype=dtype)
        if Rmax is not None:
            raise ValueError('Rmax should not be provided when bins is an array')

    npairs = np.zeros(len(bin_edges) - 1, dtype=np.uint64)
    if do_ravg:
        ravg = np.zeros(len(bin_edges) - 1, dtype=accum_dtype)
    else:
        ravg = None
    if W1 is not None:
        wavg = np.zeros(len(bin_edges) - 1, dtype=accum_dtype)
    else:
        wavg = None

    if num_threads is None:
        num_threads = 0

    isa = _lookup_isa(isa, module)

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
        wavg=wavg,
        num_threads=num_threads,
        boxsize=boxsize,
        weight_method=weight_method,
        verbose=verbose,
        isa=isa,
        grid_refine=grid_refine,
        max_cells=max_cells,
    )

    res = astropy.table.Table(
        {
            'npairs': npairs,
        },
        meta={
            'N1': len(X1),
            'N2': len(X2) if X2 is not None else None,
            'autocorr': X2 is None,
            'weight_method': weight_method,
            'boxsize': boxsize,
            'isa': isa,
            'num_threads': num_threads,
            'dtype': dtype,
            'accum_dtype': accum_dtype
        },
        copy=False,
    )

    if do_ravg:
        res['ravg'] = ravg
    if W1 is not None:
        res['wavg'] = wavg

    return res
