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

Usage
=====

The ``allclose`` fixture is used just like Numpy's ``allclose`` function.

.. code-block:: python

   import numpy

   def test_close(allclose):
       x = np.linspace(-1, 1)
       y = x + 0.001
       assert allclose(y, x, atol=0.002)
       assert not allclose(y, x, atol=0.0005)
       assert not allclose(y, x, rtol=0.002)

To use these fixtures, install with

.. code-block:: bash

   pip install pytest-allclose

Configuration
=============

The following configuration option exists.

allclose_test_tolerances
------------------------

``allclose_test_tolerances`` accepts a list of test name patterns,
followed by values for ``atol``, ``rtol``, or both.

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

See the full
`documentation <https://www.nengo.ai/pytest-allclose>`__
for the API reference.
