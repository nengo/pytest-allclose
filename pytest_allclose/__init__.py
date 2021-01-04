"""
pytest_allclose
===============

Pytest fixture extending Numpy's allclose function.
"""

from .plugin import report_rmses
from .version import version as __version__

__copyright__ = "2019-2021 pytest_plt contributors"
__license__ = "MIT license"
