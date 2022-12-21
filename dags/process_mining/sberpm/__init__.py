from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

from process_mining.sberpm._holder import DataHolder
from sberpm._version import __version__

__all__ = [
    "autoinsights",
    "ml",
    "DataHolder",
    "metrics",
    "__version__"
]
