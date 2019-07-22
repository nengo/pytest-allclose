"""Test the allclose fixture."""

import inspect

import numpy as np
import pytest


def eye_vector(n, k, dtype=bool):
    return np.eye(1, n, k=k, dtype=dtype)[0]


def add_close_noise(x, atol, rtol, rng):
    scale = rng.uniform(-rtol, rtol, size=x.shape)
    offset = rng.uniform(-atol, atol, size=x.shape)
    return x + scale * np.abs(x) + offset


def get_vector_pairs(atol, rtol, rng):
    x = rng.uniform(-1, 1, size=100)

    # augment the whole vector by less than the tolerances (should pass)
    y = add_close_noise(x, atol=atol, rtol=rtol, rng=rng)

    # augment a random element by more then `atol` (should fail)
    i = rng.randint(x.size)
    mask = eye_vector(x.size, i)
    x0 = x * (~mask)
    y0 = x0 + 1.1 * atol * mask

    # augment the largest magnitude element by more than rtol (should fail)
    i = np.argmax(np.abs(x))
    assert np.abs(x[i]) * rtol > atol
    x1 = x
    y1 = x1 + 3 * rtol * x1[i] * eye_vector(x.size, i)

    # return pairs of x (reference) and y (augmented) values,
    # and whether they are close according to the tolerances
    return [(x, y, True), (x0, y0, False), (x1, y1, False)]


def test_allclose(allclose):
    rng = np.random.RandomState(3)
    atol = 1e-5
    rtol = 1e-3

    pairs = get_vector_pairs(atol, rtol, rng)
    for x, y, close in pairs:
        assert allclose(y, x, atol=atol, rtol=rtol) == close


def test_tolerances_big(allclose):
    """Make sure tolerances in setup.cfg are properly applied"""
    rng = np.random.RandomState(4)

    # atol=0.01 and rtol=0.2 has been set in setup.cfg.
    pairs = get_vector_pairs(atol=0.01, rtol=0.2, rng=rng)
    for x, y, close in pairs:
        assert allclose(y, x) == close


def test_tolerances_small(allclose):
    """Make sure tolerances in setup.cfg are properly applied"""
    rng = np.random.RandomState(5)

    # atol=0.001 and rtol=0.005 has been set in setup.cfg.
    pairs = get_vector_pairs(atol=0.001, rtol=0.005, rng=rng)
    for x, y, close in pairs:
        assert allclose(y, x) == close


@pytest.mark.parametrize("big_tols", [False, True])
def test_parametrized(big_tols, allclose):
    rng = np.random.RandomState(6)
    atol, rtol = (0.1, 0.2) if big_tols else (0.001, 0.002)

    pairs = get_vector_pairs(atol, rtol, rng)
    for x, y, close in pairs:
        assert allclose(y, x) == close


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_precedence(order, allclose):
    closure_vars = inspect.getclosurevars(allclose)
    overrides = closure_vars.nonlocals["overrides"]
    assert len(overrides) == 1

    pairs = get_vector_pairs(order, order * 2, np.random.RandomState(6))
    if order in (1, 2):
        assert overrides[0] == dict(atol=order, rtol=order * 2)
        for x, y, close in pairs:
            assert allclose(y, x) == close
    else:
        assert overrides[0] == dict(atol=2, rtol=4)
        x, y, _ = pairs[0]
        assert not allclose(y, x)


def test_multiple_tolerances(allclose):
    # setup.cfg specifies separate first, second, and third tolerances
    rng = np.random.RandomState(7)

    x0, y0, close0 = get_vector_pairs(atol=0.001, rtol=0.004, rng=rng)[0]
    assert close0 and allclose(y0, x0)

    x1, y1, close1 = get_vector_pairs(atol=0.01, rtol=0.05, rng=rng)[0]
    assert close1 and allclose(y1, x1)

    # go back to smaller tols, to test we're not just using last tols
    x2, y2, close2 = get_vector_pairs(atol=0.002, rtol=0.005, rng=rng)[0]
    assert close2 and allclose(y2, x2)


def test_xtol(allclose):
    x = np.linspace(-1, 1)
    dx = x[1] - x[0]
    y = [x + i * dx for i in range(4)]

    assert allclose(y[1], x, atol=1e-8, rtol=1e-8, xtol=1)
    assert not allclose(y[2], x, atol=1e-8, rtol=1e-8, xtol=1)
    assert allclose(y[2], x, atol=1e-8, rtol=1e-8, xtol=2)
    assert not allclose(y[3], x, atol=1e-8, rtol=1e-8, xtol=2)


def test_docstring(allclose):
    assert allclose.__doc__.startswith(
        "Checks if two arrays are close, mimicking `numpy.allclose`."
    )

    assert "Parameters\n    ----------" in allclose.__doc__
    assert "Returns\n    -------" in allclose.__doc__
