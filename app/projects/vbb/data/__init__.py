"""
VBB Data Module
===============

Dataset loading and processing for VBB.

Modules:
    loader:   DatasetLoader with caching support
    queries/: SQL query definitions
"""

from projects.vbb.data.loader import DatasetLoader

__all__ = ["DatasetLoader"]
