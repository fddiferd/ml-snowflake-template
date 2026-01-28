"""
Shared Configuration for ML Layer
==================================

Single source of truth for Snowflake infrastructure settings and common packages.
This module centralizes configuration that's shared across all projects.

Usage:
    from src.shared_config import ROLE_NAME, WAREHOUSE_NAME, BASE_PACKAGES
"""

# =============================================================================
# Snowflake Infrastructure Settings
# =============================================================================

ROLE_NAME = "ML_LAYER_ROLE"
WAREHOUSE_NAME = "ML_LAYER_WH"
STAGE_NAME = "ML_LAYER_STAGE"

# =============================================================================
# Python Packages for Stored Procedures
# =============================================================================

# Base packages required by all projects
BASE_PACKAGES = [
    'snowflake-snowpark-python',
    'pandas',
    'scikit-learn',
    'numpy',
    'pydantic',
    'toml',
    'joblib',
]

# ML-specific packages (opt-in per project)
ML_PACKAGES = [
    'xgboost',
    'shap',
]

# All packages (base + ML)
ALL_PACKAGES = BASE_PACKAGES + ML_PACKAGES


def get_packages_sql_list(packages: list[str] | None = None) -> str:
    """Generate SQL-formatted package list for stored procedure definition.
    
    Args:
        packages: List of package names. If None, uses ALL_PACKAGES.
    
    Returns:
        SQL-formatted string like "'pkg1', 'pkg2', 'pkg3'"
    
    Example:
        >>> get_packages_sql_list(['pandas', 'numpy'])
        "'pandas', 'numpy'"
    """
    if packages is None:
        packages = ALL_PACKAGES
    return ", ".join(f"'{pkg}'" for pkg in packages)
