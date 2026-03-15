"""dfcontext — Generate optimal LLM context from pandas DataFrames."""

from dfcontext._version import __version__
from dfcontext.config import ContextConfig
from dfcontext.core import to_context

__all__ = ["__version__", "ContextConfig", "to_context"]
