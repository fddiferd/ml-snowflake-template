"""
Environment Configuration
=========================

Lazy-loaded environment settings. Values are read from env vars
on first access, not at import time (needed for Snowflake stored procedures).

Usage:
    from src.environment import environment
    
    if environment.target.is_dev:
        # DEV-specific logic
        pass
"""

import os
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class Target(Enum):
    DEV = "DEV"
    STAGING = "STAGING"
    PROD = "PROD"

    @property
    def is_dev(self) -> bool:
        return self == Target.DEV


class Environment:
    """Lazy-loaded environment. Reads env vars on first access."""
    
    _loaded = False
    _target: Target
    _developer: str | None
    _use_cache: bool
    
    def _load(self) -> None:
        """Load env vars once on first access."""
        if Environment._loaded:
            return
        
        # Read TARGET (required)
        target_str = os.getenv("TARGET")
        if not target_str:
            raise ValueError("TARGET is not set")
        
        Environment._target = Target(target_str.upper())
        Environment._developer = os.getenv("DEVELOPER")
        Environment._use_cache = os.getenv("USE_CACHE", "").upper() == "TRUE"
        
        # Validate: DEV requires DEVELOPER
        if Environment._target == Target.DEV and not Environment._developer:
            raise ValueError("DEVELOPER is not set for DEV target")
        
        logger.info(f"Environment: target={Environment._target.value}, use_cache={Environment._use_cache}")
        Environment._loaded = True
    
    @property
    def target(self) -> Target:
        self._load()
        return Environment._target
    
    @property
    def use_cache(self) -> bool:
        self._load()
        return Environment._use_cache
    
    @property
    def schema_name(self) -> str:
        self._load()
        if Environment._target == Target.DEV and Environment._developer:
            return f"DEV_{Environment._developer.upper()}"
        return Environment._target.value


# Global instance
environment = Environment()
