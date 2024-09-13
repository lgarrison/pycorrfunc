import numpy as np
import pycorrfunc.theory as theory
import pytest

def test_dd_double():
    # make a particle at (0, 0, 0) and (1, 1, 1)
    X = np.array([0, 1], dtype=np.float64)
    Y = np.array([0, 1], dtype=np.float64)
    Z = np.array([0, 1], dtype=np.float64)
    X2 = X.copy()
    Y2 = Y.copy()
    Z2 = Z.copy()
    bin_edges = np.array([0., 2., 4.], dtype=np.float64)
    box = 10.

    result = theory.DD(X, Y, Z, X2, Y2, Z2, bin_edges, 1, box)
    # result.npairs should be [4, 0]
    expected_npairs = np.array([4, 0], dtype=np.float64)
    assert np.allclose(result.npairs, expected_npairs), f"Expected {expected_npairs}, but got {result.npairs}"




def test_dd_float():
    # make a particle at (0, 0, 0) and (0, 0, 1)
    X = np.array([0, 1], dtype=np.float32)
    Y = np.array([0, 1], dtype=np.float32)
    Z = np.array([0, 1], dtype=np.float32)
    X2 = X.copy()
    Y2 = Y.copy()
    Z2 = Z.copy()
    bin_edges = np.array([0., 2., 4.], dtype=np.float32)
    box = 10.
    
    result = theory.DD(X, Y, Z, X2, Y2, Z2, bin_edges, 1, box)
    # result.npairs should be [4, 0]
    expected_npairs = np.array([4, 0], dtype=np.float32)
    assert np.allclose(result.npairs, expected_npairs), f"Expected {expected_npairs}, but got {result.npairs}"

if __name__ == "__main__":
    pytest.main(["-s"])
