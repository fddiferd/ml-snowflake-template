"""
Stored Procedure Utilities
==========================

Provides reusable components for creating Snowflake stored procedures.

Usage:
    from src.sproc import create_sproc_handler
    
    # In your project's sproc.py:
    run_sproc = create_sproc_handler("PROJECT_NAME", "projects.myproject.main")
"""

from src.sproc.base import create_sproc_handler

__all__ = ["create_sproc_handler"]
