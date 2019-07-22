"""The ``allclose`` fixture definition"""

from fnmatch import fnmatch

import numpy as np
import pytest


@pytest.fixture
def allclose(request):
    """A functional replacement for np.allclose.

    This allows backends to specify different tolerances by adding

    .. code-block:: ini

        nengo_test_tolerances =
            path/to/test/file:testname atol=x rtol=y
            path/to/other/file:testname2 atol=z  # this is a comment
            ...

    to their pytest config file.

    If the test is parametrized then ``testname[param0-param1]`` will match
    specific parameter settings, and ``testname*`` will match all parameter
    settings.

    If a test contains multiple ``allclose`` checks then multiple entries
    with the same testname can be added to the config, and will be applied
    in the given order. If there are fewer config entries than tests, the
    last entry will be applied to remaining tests with the same testname.
    """

    nodename = request.node.nodeid

    call_count = [0]  # need to access the closure this way for 2.7 compat
    override = []

    tol_cfg = request.config.inicfg.get("allclose_test_tolerances", "")
    for line in (x for x in tol_cfg.split('\n') if len(x) > 0):
        # each line contains a testname and list of kwargs (e.g. `atol=0.1`)
        split_line = line.split()
        testname = "*" + split_line[0]
        kwargs = {}
        for entry in split_line[1:]:
            if entry.startswith("#"):
                break
            k, v = entry.split("=")
            kwargs[k] = float(v)

        # escape special characters in testname
        escaped = "[]?"
        testname = list(testname)
        for i, c in enumerate(testname):
            if c in escaped:
                testname[i] = "[%s]" % c
        testname = "".join(testname)

        if fnmatch(nodename, testname):
            override.append(kwargs)

    def _allclose(a, b, **kwargs):
        # set tolerances
        if override:
            kwargs.update(override[min(call_count[0], len(override) - 1)])
            call_count[0] += 1

        return np.allclose(a, b, **kwargs)

    return _allclose
