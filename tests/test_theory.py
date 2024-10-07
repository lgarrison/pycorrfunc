import numpy as np
import numpy.testing as npt
import pytest

import pycorrfunc.theory as theory

DEFAULT_BOXSIZE = 123.0


def brute_2pcf(pos, w, bin_edges, boxsize=None):
    """
    Brute-force calculation of 2-point correlation function.
    """
    pdiff = np.abs(pos[:, :, np.newaxis] - pos[:, np.newaxis])
    if boxsize is not None:
        boxsize = np.atleast_2d(boxsize).T
        mask = pdiff >= boxsize[..., np.newaxis] / 2
        pdiff -= mask * boxsize[..., np.newaxis]

    r2 = (pdiff**2).sum(axis=0).reshape(-1)
    brutecounts, _ = np.histogram(r2, bins=bin_edges**2)
    mask = brutecounts > 0

    # and compute ravg of pairs in each bin
    ibin = np.digitize(r2, bins=bin_edges**2) - 1
    r = np.sqrt(r2)
    ravg = np.zeros(len(bin_edges), dtype=np.float64)
    np.add.at(ravg, ibin, r)
    ravg = ravg[:-1]  # oversized by 1 for overflow
    ravg[mask] /= brutecounts[mask]

    # and compute wavg of pairs in each bin
    pairw = (w[:, np.newaxis] * w).reshape(-1)
    wavg = np.zeros(len(bin_edges), dtype=np.float64)
    np.add.at(wavg, ibin, pairw)
    wavg = wavg[:-1]
    wavg[mask] /= brutecounts[mask]

    return brutecounts, ravg, wavg


@pytest.mark.parametrize('autocorr', [False, True], ids=['cross', 'auto'])
@pytest.mark.parametrize(
    'gridref', [None, 3, [1, 2, 3]], ids=['refdef', 'ref3', 'ref123']
)
@pytest.mark.parametrize(
    'maxcells', [None, 1, 2, 3], ids=['maxdef', 'max1', 'max2', 'max3']
)
@pytest.mark.parametrize(
    'boxsize', [123.0, (51.0, 75.0, 123.0), None], ids=['iso', 'aniso', 'open']
)
# @pytest.mark.parametrize("funcname", ["DD", "DDrppi", "DDsmu"])
@pytest.mark.parametrize('funcname', ['DD'])
def test_gridding(autocorr, gridref, maxcells, boxsize, funcname, isa='fastest', dtype='f4'):
    check_brute(autocorr, gridref, maxcells, boxsize, funcname, isa, dtype)


@pytest.mark.parametrize('autocorr', [False, True], ids=['cross', 'auto'])
@pytest.mark.parametrize('funcname', ['DD'])
@pytest.mark.parametrize('isa', ['avx512', 'avx', 'sse42', 'fallback'])
@pytest.mark.parametrize('dtype', ['f4', 'f8'])
def test_kernels(
    funcname, isa, dtype, autocorr, gridref=None, maxcells=None, boxsize=DEFAULT_BOXSIZE
):
    check_brute(autocorr, gridref, maxcells, boxsize, funcname, isa, dtype)


def check_brute(
    autocorr,
    gridref,
    maxcells,
    boxsize,
    funcname,
    isa,
    dtype,
    N=1000,
):
    """Generate random points in a box and compare to brute-force results.

    In the periodic case, use close to the max allowable Rmax,
    0.49*min(boxsize).

    Tests both isotropic and anisotropic boxes.
    """

    rng = np.random.default_rng(1237)
    if boxsize is not None:
        boxsize_arr = np.atleast_2d(boxsize).T
        bin_edges = np.linspace(0.01, 0.49 * boxsize_arr.min(), 20)
    else:
        # non-periodic has no upper limit on Rmax
        boxsize_arr = np.atleast_2d(DEFAULT_BOXSIZE).T
        bin_edges = np.linspace(0.01, 1.1 * DEFAULT_BOXSIZE, 20)
    # pimax = np.floor(0.49 * boxsize.min())
    # mu_max = 0.5
    # nmu_bins = 10
    func = getattr(theory, funcname)

    pos = rng.random(size=(3, N), dtype=dtype) * boxsize_arr

    w = np.ones(N, dtype=dtype)
    # w = rng.random(size=N, dtype=dtype)

    kwargs = dict(
        X1=pos[0],
        Y1=pos[1],
        Z1=pos[2],
        W1=w,
        weight_method='pair_product',
        bins=bin_edges,
        isa=isa,
        boxsize=boxsize,
        max_cells=maxcells,
        dtype=dtype,
        verbose=True,
        grid_refine=gridref,
        do_ravg=True,
    )

    if not autocorr:
        kwargs.update(
            dict(
                X2=pos[0],
                Y2=pos[1],
                Z2=pos[2],
                W2=w,
            )
        )

    if funcname == 'DDrppi':
        # # Compute rp^2 = dx^2 + dy^2, and pi = abs(dz)
        # args.insert(2, pimax)
        # sqr_rpdiff = (pdiff[:, :, :2] ** 2).sum(axis=-1).reshape(-1)
        # pidiff = np.abs(pdiff[:, :, 2]).reshape(-1)
        # pibins = np.linspace(0.0, pimax, int(pimax) + 1)
        # brutecounts, _, _ = np.histogram2d(sqr_rpdiff, pidiff, bins=(bins**2, pibins))
        # brutecounts = brutecounts.reshape(-1)  # corrfunc flattened convention
        pass
    elif funcname == 'DDsmu':
        # # Compute s^2 = dx^2 + dy^2 + dz^2, mu = |dz| / s
        # args[3:3] = (mu_max, nmu_bins)
        # sdiff = np.sqrt((pdiff**2).sum(axis=-1).reshape(-1))
        # sdiff[sdiff == 0.0] = np.inf  # don't divide by 0
        # mu = np.abs(pdiff[:, :, 2]).reshape(-1) / sdiff
        # mubins = np.linspace(0, mu_max, nmu_bins + 1)
        # brutecounts, _, _ = np.histogram2d(sdiff, mu, bins=(bins, mubins))
        # brutecounts = brutecounts.reshape(-1)  # corrfunc flattened convention
        pass
    elif funcname == 'DD':
        brute_counts, brute_ravg, brute_wavg = brute_2pcf(
            pos, w, bin_edges, boxsize=boxsize
        )
    else:
        raise NotImplementedError(funcname)

    # spot-check that we have non-zero counts
    assert np.any(brute_counts > 0)

    results = func(**kwargs)

    # FUTURE We might prefer that the tolerance scale with sqrt(brute_counts) or similar
    if dtype == 'f4':
        rtol = 1e-4
        atol = 0.0
    else:
        rtol = 1e-13
        atol = 0.0

    npt.assert_equal(results['npairs'], brute_counts)
    npt.assert_allclose(results['ravg'], brute_ravg, rtol=rtol, atol=atol)
    npt.assert_equal(results['wavg'], brute_wavg)


def test_isa_error():
    with pytest.raises(ValueError, match='ISA'):
        theory.DD(
            X1=np.array([0.0]),
            Y1=np.array([0.0]),
            Z1=np.array([0.0]),
            W1=np.array([1.0]),
            bins=np.array([0.0, 1.0]),
            isa='not_an_isa',
        )


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_one_dev(dtype):
    # make a particle at (0, 0, 0) and (1, 1, 1)
    X = np.array([0, 1], dtype=dtype)
    Y = np.array([0, 1], dtype=dtype)
    Z = np.array([0, 1], dtype=dtype)
    X2 = X.copy()
    Y2 = Y.copy()
    Z2 = Z.copy()
    bin_edges = np.array([0.0, 2.0, 4.0])
    box = 10.0

    result = theory.DD(X, Y, Z, bin_edges, X2=X2, Y2=Y2, Z2=Z2, boxsize=box)
    expected_npairs = np.array([4, 0])
    assert np.allclose(
        result['npairs'], expected_npairs
    ), f"Expected {expected_npairs}, but got {result['npairs']}"
