"""Test the allclose fixture."""

import numpy as np
import pytest


def eye_vector(n, k, dtype=bool):
    return np.eye(1, n, k=k, dtype=dtype)[0]


def add_close_noise(x, atol, rtol, rng):
    m = rng.uniform(-rtol, rtol, size=x.shape)
    n = rng.uniform(-atol, atol, size=x.shape)
    return x + m*np.abs(x) + n


def get_vector_pairs(atol, rtol, rng):
    x = rng.uniform(-1, 1, size=100)
    y = add_close_noise(x, atol=atol, rtol=rtol, rng=rng)

    m = eye_vector(x.size, 0)
    x0 = x * (~m)
    y0 = x0 + 1.1*atol*m

    i = np.argmax(np.abs(x))
    assert np.abs(x[i])*rtol > atol
    x1 = x
    y1 = x1 + 3*rtol*x1[i]*eye_vector(x.size, i)

    return [
        (True, x, y),
        (False, x0, y0),
        (False, x1, y1),
    ]


def test_allclose(allclose):
    atol = 1e-5
    rtol = 1e-3

    pairs = get_vector_pairs(atol, rtol, np.random.RandomState(3))
    for close, x, y in pairs:
        assert allclose(y, x, atol=atol, rtol=rtol) is close


def test_tolerances_big(allclose):
    """Make sure tolerances in setup.cfg are properly applied"""
    # atol=0.01 and rtol=0.2 has been set in setup.cfg.
    atol = 0.01
    rtol = 0.2

    pairs = get_vector_pairs(atol, rtol, np.random.RandomState(4))
    for close, x, y in pairs:
        assert allclose(y, x) is close


def test_tolerances_small(allclose):
    """Make sure tolerances in setup.cfg are properly applied"""
    # atol=0.001 and rtol=0.005 has been set in setup.cfg.
    atol = 0.001
    rtol = 0.005

    pairs = get_vector_pairs(atol, rtol, np.random.RandomState(5))
    for close, x, y in pairs:
        assert allclose(y, x) is close


@pytest.mark.parametrize('big_tols', [False, True])
def test_parametrized(big_tols, allclose):
    if big_tols:
        atol, rtol = 0.1, 0.2
    else:
        atol, rtol = 0.001, 0.002

    pairs = get_vector_pairs(atol, rtol, np.random.RandomState(6))
    for close, x, y in pairs:
        assert allclose(y, x) is close


def test_multiple_tolerances(allclose):
    # setup.cfg specifies separate first, second, and third tolerances
    rng = np.random.RandomState(7)

    atol0, rtol0 = 0.001, 0.004
    close0, x0, y0 = get_vector_pairs(atol0, rtol0, rng)[0]
    assert close0 and allclose(y0, x0)

    atol1, rtol1 = 0.01, 0.05
    close1, x1, y1 = get_vector_pairs(atol1, rtol1, rng)[0]
    assert close1 and allclose(y1, x1)

    # go back to smaller tols, to test we're not just using last tols
    atol2, rtol2 = 0.002, 0.005
    close2, x2, y2 = get_vector_pairs(atol2, rtol2, rng)[0]
    assert close2 and allclose(y2, x2)
