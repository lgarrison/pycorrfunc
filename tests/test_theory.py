import numpy as np
import pytest

import pycorrfunc.theory as theory


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_DD(dtype):
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
        result.npairs, expected_npairs
    ), f"Expected {expected_npairs}, but got {result.npairs}"
