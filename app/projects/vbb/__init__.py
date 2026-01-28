"""
VBB (Value-Based Bidding) Package
=================================

ML model for value-based bidding optimization.

Usage:
    from projects.vbb import get_session, DatasetLoader
    
    session = get_session()
    loader = DatasetLoader(session)
    df = loader.load()

Exports:
    Session:    get_session
    Config:     Configuration constants and helpers
    Data:       DatasetLoader
"""

from typing import TYPE_CHECKING
from snowflake.snowpark import Session

# Type hints for lazy imports (enables IDE autocomplete)
if TYPE_CHECKING:
    from projects.vbb.data.loader import DatasetLoader as DatasetLoader

from projects import Project

# Config exports
from projects.vbb.config import (
    CACHE_PATH,
    TIMESTAMP_COL,
)


def get_session() -> Session:
    """Get a Snowflake session configured for the VBB project."""
    from src.connection.session import get_session as get_snowflake_session
    return get_snowflake_session(Project.VBB)


# ============================================================================
# Lazy imports for data functions
# ============================================================================
_lazy_imports = {
    "DatasetLoader": ("projects.vbb.data.loader", "DatasetLoader"),
}


def __getattr__(name: str):
    """Lazy import handler for data components."""
    if name in _lazy_imports:
        module_path, attr_name = _lazy_imports[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Session
    "get_session",
    # Config
    "CACHE_PATH",
    "TIMESTAMP_COL",
    # Data (lazy loaded)
    "DatasetLoader",
]
