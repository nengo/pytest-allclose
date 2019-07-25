"""
pytest_allclose
===============

Pytest fixture extending Numpy's allclose function.
"""

from .version import version as __version__

from .plugin import report_rmses

__copyright__ = "2019-2019 pytest_plt contributors"
__license__ = "MIT license"
