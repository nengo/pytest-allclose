"""The ``allclose`` fixture definition"""

from fnmatch import fnmatch

import numpy as np
import pytest


def _add_common_docs(func):
    func.__doc__ += """
    Parameters
    ----------
    a : np.ndarray
        First array to be compared.
    b : np.ndarray
        Second array to be compared.
    rtol : float, optional
        Relative tolerance between a and b (relative to b).
    atol : float, optional
        Absolute tolerance between a and b.
    xtol : int, optional
        Allow signals to be right or left shifted by up to ``xtol``
        indices along the first axis
    equal_nan : bool, optional
        If True, nans will be considered equal to nans.
    print_fail : int, optional
        If > 0, print out the first ``print_fail`` entries failing
        the allclose check along the first axis.
    record_rmse : bool, optional
        Whether to record the RMSE value for this comparison. Defaults to True.
        Set to False whenever ``a`` and ``b`` should be far apart
        (when ensuring two signals are sufficiently different, for example).

    Returns
    -------
    bool
        True if the two arrays are considered close according to the tolerances.
    """
    return func


@pytest.fixture
@_add_common_docs
def allclose(request):
    """Returns a function checking if two arrays are close, mimicking `numpy.allclose`.

    .. currentmodule:: allclose

    .. function:: _allclose(a, b, rtol=1e-5, atol=1e-8, xtol=0, equal_nan=False, \
                            print_fail=5, record_rmse=True)
       :noindex:
    """

    overrides = _get_allclose_overrides(request)
    call_count = [0]

    @_add_common_docs
    def _allclose(
        a,
        b,
        rtol=1e-5,
        atol=1e-8,
        xtol=0,
        equal_nan=False,
        print_fail=5,
        record_rmse=True,
    ):
        """Checks if two arrays are close, mimicking `numpy.allclose`.
        """

        if len(overrides) > 0:
            override_args = overrides[min(call_count[0], len(overrides) - 1)]
            atol = override_args.get("atol", atol)
            rtol = override_args.get("rtol", rtol)
            xtol = override_args.get("xtol", xtol)
            equal_nan = override_args.get("equal_nan", equal_nan)
            print_fail = override_args.get("print_fail", print_fail)
            record_rmse = override_args.get("record_rmse", record_rmse)
            call_count[0] += 1

        a = np.atleast_1d(a)
        b = np.atleast_1d(b)

        rmse = _safe_rms(a - b)
        if record_rmse and not np.any(np.isnan(rmse)):
            request.node.user_properties.append(("rmse", rmse))

            ab_rms = _safe_rms(a) + _safe_rms(b)
            rmse_relative = (2 * rmse / ab_rms) if ab_rms > 0 else np.nan
            if not np.any(np.isnan(rmse_relative)):
                request.node.user_properties.append(("rmse_relative", rmse_relative))

        close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

        # if xtol > 0, check that number of adjacent positions. If they are
        # close, then we consider things close.
        for i in range(1, xtol + 1):
            close[i:] |= np.isclose(
                a[i:], b[:-i], rtol=rtol, atol=atol, equal_nan=equal_nan
            )
            close[:-i] |= np.isclose(
                a[:-i], b[i:], rtol=rtol, atol=atol, equal_nan=equal_nan
            )

            # we assume that the beginning and end of the array are close
            # (since we're comparing to entries outside the bounds of
            # the other array)
            close[[i - 1, -i]] = True

        result = np.all(close)

        if print_fail > 0 and not result:
            diffs = []
            # broadcast a and b to have same shape as close for indexing
            broadcast_a = a + np.zeros(b.shape, dtype=a.dtype)
            broadcast_b = b + np.zeros(a.shape, dtype=b.dtype)
            for k, ind in enumerate(zip(*(~close).nonzero())):
                if k >= print_fail:
                    break
                diffs.append("%s: %s %s" % (ind, broadcast_a[ind], broadcast_b[ind]))

            print(
                "allclose first %d failures:\n  %s" % (len(diffs), "\n  ".join(diffs))
            )

        return result

    return _allclose


_allclose_arg_types = dict(
    atol=float, rtol=float, xtol=int, equal_nan=bool, print_fail=int, record_rmse=bool
)


def _rms(x, axis=None, keepdims=False):
    return np.sqrt(np.mean(x ** 2, axis=axis, keepdims=keepdims))


def _safe_rms(x):
    x = np.asarray(x)
    return _rms(x).item() if x.size > 0 else np.nan


def _get_allclose_overrides(request):
    nodename = request.node.nodeid
    tol_cfg = request.config.inicfg.get("allclose_tolerances", "")

    # multiple overrides will match subsequent ``allclose`` calls
    overrides = []

    # we only match one pattern, the first one encountered
    matched = None

    for line in (x for x in tol_cfg.split("\n") if len(x) > 0):
        # each line contains a pattern and list of kwargs (e.g. `atol=0.1`)
        split_line = line.split()
        pattern = "*" + split_line[0]

        # escape special characters in pattern
        replace = {"[": "[[]", "]": "[]]", "?": "[?]"}
        pattern = "".join(replace.get(s, s) for s in pattern)

        if matched is None and fnmatch(nodename, pattern) or pattern == matched:
            matched = pattern
            kwargs = {}
            for entry in split_line[1:]:
                if entry.startswith("#"):
                    break

                k, v = entry.split("=")
                if k not in _allclose_arg_types:
                    raise ValueError("Unrecognized argument %r" % k)

                kwargs[k] = _allclose_arg_types[k](v)

            overrides.append(kwargs)

    return overrides


def report_rmses(terminalreporter, relative=True):
    """Report RMSEs recorded by the allclose fixture in the Pytest terminal.

    This function helps with reporting recorded root mean squared errors
    (RMSEs). These RMSEs offer a measure of performance for each test
    by quantifying how close their outputs are to the target values.
    While this metric has some value on its own, it is most useful as a
    relative metric, to evaluate if change offers an improvement to tests,
    and if so, how much.

    When using RMSEs, it is important to set ``record_rmse`` to False on any
    ``allclose`` call where closer values correspond to a drop in performance
    (e.g. when using ``allclose`` to ensure values are different).

    Parameters
    ----------
    terminalreporter : _pytest.terminal.TerminalReporter
        The terminal reporter object provided by ``pytest_terminal_summary``.
    relative : bool, optional
        Whether to print relative (default) or absolute RMSEs. Relative RMSEs
        are normalized by the mean RMS of ``a`` and ``b`` in ``allclose``.
        Since different tests often compare against values of different
        magnitudes, relative RMSEs provide a better metric across tests by
        ensuring all tests contribute proportionally to the average RMSE.
        One exception is when comparing to a signal that is all zeros, since
        the relative RMSE will always be 2 no matter how close the values are.

    Examples
    --------
    See `RMSE error reporting`_ for an example.
    """

    rmse_name = "rmse_relative" if relative else "rmse"

    tr = terminalreporter
    all_rmses = []
    for passed_test in tr.stats.get("passed", []):
        for name, val in passed_test.user_properties:
            if name == rmse_name:
                all_rmses.append(val)

    if len(all_rmses) > 0:
        relstr = "relative " if relative else ""
        tr.write_sep("=", "%sroot mean squared error for allclose checks" % relstr)
        tr.write_line(
            "mean %sRMSE: %.5f +/- %.4f (std)"
            % (relstr, np.mean(all_rmses), np.std(all_rmses))
        )
