"""
VBB Configuration
=================

Single source of truth for VBB project configuration: constants, column names, etc.

Usage:
    from projects.vbb.config import CACHE_PATH, TIMESTAMP_COL
"""

# =============================================================================
# Cache and Storage
# =============================================================================

CACHE_PATH = "app/projects/vbb/data/cache"

# =============================================================================
# Column Constants
# =============================================================================

TIMESTAMP_COL = "EVENT_DATE"

# =============================================================================
# Table Names (for Snowflake output)
# =============================================================================

TABLE_RAW_DATA = "RAW_DATA"
TABLE_RESULTS = "RESULTS"
TABLE_METADATA = "MODEL_METADATA"
