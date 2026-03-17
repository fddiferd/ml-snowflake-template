"""
Stored Procedure Entry Point for Adwords GCLID Upload
=====================================================

This module is designed to be deployed as a Snowflake stored procedure.
Uses the shared sproc factory to minimize boilerplate.

The stored procedure is scheduled via Snowflake Tasks (daily at 3pm PST).
"""

from src.sproc import create_sproc_handler

run_sproc = create_sproc_handler(
    project_name="ADWORDS_GCLID_UPLOAD",
    main_module_path="projects.adwords_gclid_upload.main",
    target="PROD"
)
