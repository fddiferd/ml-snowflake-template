"""
Stored Procedure Entry Point for VBB
=====================================

This module is designed to be deployed as a Snowflake stored procedure.
Uses the shared sproc factory to minimize boilerplate.

The stored procedure is scheduled via Snowflake Tasks (weekly).
"""

from src.sproc import create_sproc_handler

# Create the stored procedure handler using the factory
# This eliminates ~50 lines of boilerplate code
run_sproc = create_sproc_handler(
    project_name="VBB",
    main_module_path="projects.vbb.main",
    target="PROD"
)
