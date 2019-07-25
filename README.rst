***************
pytest-allclose
***************

``pytest-allclose`` provides the ``allclose`` Pytest fixture,
extending Numpy's ``allclose`` function with test-specific features.

A core feature of the ``allclose`` fixture is that the tolerances for tests
can be configured externally.
This allows different repositories to share the same tests,
but use different tolerances.
See the "Configuration" section below for details.

Installation
============

To use this fixture, install with

.. code-block:: bash

   pip install pytest-allclose

Usage
=====

The ``allclose`` fixture is used just like Numpy's ``allclose`` function.

.. code-block:: python

   import numpy as np

   def test_close(allclose):
       x = np.linspace(-1, 1)
       y = x + 0.001
       assert allclose(y, x, atol=0.002)
       assert not allclose(y, x, atol=0.0005)
       assert not allclose(y, x, rtol=0.002)

The ``allclose`` fixture stores root-mean-square error values,
which can be reported in the pytest terminal summary.
To do so, put the following in your ``conftest.py`` file.

..code-block:: python

    from pytest_allclose import report_rmses

    def pytest_terminal_summary(terminalreporter):
        report_rmses(terminalreporter)

The ``allclose`` fixture has a number of arguments
that are not part of Numpy's ``allclose``.
One such argument is ``xtol``,
which allows arrays that have been shifted along their first axis
by a certain number of steps to be considered close.

.. code-block:: python

   import numpy as np

   def test_close(allclose):
       x = np.linspace(-1, 1)

       assert allclose(x[1:], x[:-1], xtol=1)
       assert allclose(x[3:], x[:-3], xtol=3)
       assert not allclose(x[3:], x[:-3], xtol=1)

Configuration
=============

The following configuration option exists.

allclose_test_tolerances
------------------------

``allclose_test_tolerances`` accepts a list of test name patterns,
followed by values for any of the ``allclose`` parameters.
These values will override any values provided within the test function itself,
allowing multiple repositories to use the same test suite,
but with different tolerances.

.. code-block:: ini

   allclose_test_tolerances =
       test_file.py:test_function atol=0.3  # set atol for specific test
       test_file.py:test_func* rtol=0.2  # set rtol for tests matching wildcard
       test_file.py:* atol=0.1 rtol=0.3  # set both tols for all tests in file
       test_*tion rtol=0.2  # set rtol for all matching tests in any file
       test_function[True] atol=0.1  # set atol only for one parametrization

Matching is done using ``fnmatch``,
but the following characters are escaped from matching: ``[]?``.
This is to allow easy matching of particular parametrizations of tests.
Thus the wildcard character ``*`` is the only character
that is interpreted specially.

If a test has multiple ``allclose`` calls,
you can use multiple tolerance lines that match the same test
to set different values for the first, second, third, etc. calls.
If there are more ``allclose`` calls than tolerance lines,
the last tolerance line will be used for all remaining ``allclose`` calls.

.. code-block:: python

   def test_close(allclose):
       x = np.linspace(-1, 1)
       y = x + 0.001
       assert allclose(y, x)
       assert not allclose(y, x)

.. code-block:: ini

   allclose_test_tolerances =
       test_close atol=0.002  # set atol for first allclose call
       test_close atol=0.0005  # set atol for second allclose call

.. note:: Different tolerance lines correspond to *calls* of the function,
          not lines of code. If you have e.g. a ``for`` loop that calls
          ``allclose`` 3 times, then each of these calls corresponds to a
          new tolerance line. If you have a fourth ``allclose`` call that
          you want to have different tolerances, you would need
          three tolerance lines for the first three calls in the ``for`` loop,
          then a fourth line for the last call.

.. note:: Multiple different patterns can match the same test,
          in which case each matching pattern will be interpreted as providing
          different tolerances for a subsequent ``allclose`` call in the test.
          This precludes using a general pattern to set tolerances for all
          tests in a file, then using a more specific pattern to set different
          tolerances for a few specific tests, for example.

See the full
`documentation <https://www.nengo.ai/pytest-allclose>`__
for the API reference.
