"""
Base Stored Procedure Factory
=============================

Provides a factory function to create standardized stored procedure handlers
with consistent error handling, logging, and result formatting.

This eliminates ~40 lines of boilerplate from each project's sproc.py file.

Usage:
    from src.sproc.base import create_sproc_handler
    
    # Create a handler for your project
    run_sproc = create_sproc_handler("PLTV", "projects.pltv.main")
"""

from typing import Callable


def create_sproc_handler(
    project_name: str,
    main_module_path: str,
    target: str = "PROD"
) -> Callable:
    """Factory to create stored procedure handlers with standard error handling.
    
    Creates a `run_sproc` function that:
    - Sets up the TARGET environment variable
    - Configures logging
    - Tracks execution timing
    - Handles errors gracefully
    - Returns a standardized JSON result
    
    Args:
        project_name: Name of the project (e.g., "PLTV", "VBB")
        main_module_path: Import path to the main module (e.g., "projects.pltv.main")
        target: Target environment (DEV, STAGING, PROD). Default: "PROD"
    
    Returns:
        A `run_sproc(session) -> str` function suitable for Snowflake stored procedures.
    
    Example:
        # In projects/vbb/sproc.py:
        from src.sproc.base import create_sproc_handler
        run_sproc = create_sproc_handler("VBB", "projects.vbb.main")
    """
    
    def run_sproc(session) -> str:
        """Stored procedure entry point for Snowflake Tasks.
        
        This function is called by Snowflake when the task executes.
        The session is automatically provided by Snowflake's runtime.
        
        Args:
            session: Snowflake Session (automatically provided)
        
        Returns:
            str: JSON string with status, timing, and error info
        """
        # All imports deferred to runtime to avoid validation errors
        import os
        import json
        import time
        import logging
        import importlib
        from datetime import datetime
        
        # Set TARGET environment variable before importing modules that depend on it
        os.environ["TARGET"] = target
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        start_time = time.time()
        result = {
            "status": "SUCCESS",
            "project": project_name,
            "target": target,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "duration_seconds": None,
            "error": None
        }
        
        try:
            # Dynamic import of the main module
            logger.info(f"Starting {project_name} stored procedure...")
            main_module = importlib.import_module(main_module_path)
            
            # Import WriterType for the main function call
            from src.writers import WriterType
            
            # Run the main pipeline with Snowflake writer
            # Pass session directly to avoid creating a new session from config.toml
            main_module.main(session=session, writer_type=WriterType.SNOWFLAKE, reset_schema=False)
            
            result["completed_at"] = datetime.now().isoformat()
            result["duration_seconds"] = round(time.time() - start_time, 2)
            logger.info(f"{project_name} sproc completed successfully in {result['duration_seconds']}s")
            return json.dumps(result)
            
        except Exception as e:
            result["status"] = "FAILED"
            result["error"] = str(e)
            result["completed_at"] = datetime.now().isoformat()
            result["duration_seconds"] = round(time.time() - start_time, 2)
            logger.error(f"{project_name} sproc failed: {e}")
            return json.dumps(result)
    
    return run_sproc
