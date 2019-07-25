import re

import numpy as np
import pytest


pytest_plugins = ["pytester"]  # adds the `testdir` fixture


def assert_all_passed(result):
    """Assert that all outcomes are 0 except for 'passed'.

    Also returns the number of passed tests.
    """
    outcomes = result.parseoutcomes()
    for outcome in outcomes:
        if outcome not in ("passed", "seconds"):
            assert outcomes[outcome] == 0
    return outcomes.get("passed", 0)


@pytest.mark.parametrize('offsets', [(0.001,), (0.001, 0.002)])
@pytest.mark.parametrize('relative', [False, True])
def test_rmse_output(offsets, relative, testdir):
    testdir.makeconftest("""
from pytest_allclose import report_rmses

def pytest_terminal_summary(terminalreporter):
    report_rmses(terminalreporter, relative=%(relative)s)
    """ % dict(relative=relative))

    test = """
import numpy as np
import pytest

@pytest.mark.parametrize('offset', [%(offsets)s])
def test_rmse(offset, allclose):
    x = np.linspace(-1, 1)
    y = x + offset
    assert allclose(y, x, atol=offset + 1e-8)
""" % dict(offsets=', '.join(str(x) for x in offsets))
    testdir.makefile('.py', *test.split('\n'))

    result = testdir.runpytest("-v")
    n_passed = assert_all_passed(result)
    assert n_passed > 0

    tag = 'mean %sRMSE: ' % ("relative " if relative else "")
    lines = [s[len(tag):] for s in result.outlines if s.startswith(tag)]
    assert len(lines) == 1
    line = lines[0]
    parts = line.split()
    assert len(parts) == 4 and parts[1] == '+/-' and parts[3] == '(std)'
    mean, std = float(parts[0]), float(parts[2])

    x = np.linspace(-1, 1)
    x_rms = np.sqrt(np.mean(x**2))
    rmses = [offset / x_rms for offset in offsets] if relative else offsets
    assert np.allclose(mean, np.mean(rmses), atol=1e-4)
    assert np.allclose(std, np.std(rmses), atol=1e-4)


@pytest.mark.parametrize('rel_error, print_fail',
                         [(1.00101, 5),
                          (1.0011, 6)])
def test_print_fail_output(rel_error, print_fail, testdir):
    test = """
import numpy as np

def test_print_fail(allclose):
    t = np.linspace(0, 1)
    x = np.sin(2*np.pi*t)
    y = x * %(rel_error)s
    assert allclose(y, x, atol=0.001, rtol=0, print_fail=%(print_fail)d)
""" % dict(rel_error=rel_error, print_fail=print_fail)
    testdir.makefile('.py', *test.split('\n'))

    result = testdir.runpytest("-v")
    outcomes = result.parseoutcomes()
    assert outcomes.get('passed', 0) == 0 and outcomes.get('failed', 0) == 1

    # reference values
    ref_inds = (12, 13, 36, 37)
    t = np.linspace(0, 1)
    x = np.sin(2*np.pi*t)
    y = x * rel_error
    ref_inds = (~np.isclose(y, x, atol=0.001, rtol=0)).nonzero()[0]

    # parse output lines
    pattern = "allclose first ([0-9]+) failures"
    line_inds = [i for i, s in enumerate(result.outlines)
                 if re.match(pattern, s)]
    assert len(line_inds) == 1
    line_ind = line_inds[0]
    n_lines = int(re.match(pattern, result.outlines[line_ind]).groups()[0])
    lines = result.outlines[line_ind+1:line_ind+n_lines+1]
    assert n_lines == min(print_fail, len(ref_inds))

    pattern = r'[ ]*\(([0-9, ]+)\): (\S+) (\S+)'
    parsed = []
    for line in lines:
        match = re.match(pattern, line)
        assert match
        groups = match.groups()
        parsed.append((
            tuple(int(s) for s in groups[0].split(',') if len(s) > 0),
            float(groups[1]),
            float(groups[2]),
        ))

    # check output lines against reference values
    for ref_ind, parts in zip(ref_inds, parsed):
        assert parts[0] == (ref_ind,)
        assert np.allclose(parts[1], y[ref_ind])
        assert np.allclose(parts[2], x[ref_ind])
