"""
Stored Procedure Entry Point for PLTV
=====================================

This module is designed to be deployed as a Snowflake stored procedure.
All imports are deferred inside the function to avoid validation errors.

The stored procedure is scheduled via Snowflake Tasks (weekly).
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
    from datetime import datetime
    
    # Set TARGET environment variable before importing modules that depend on it
    os.environ["TARGET"] = "PROD"
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    result = {
        "status": "SUCCESS",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "duration_seconds": None,
        "error": None
    }
    
    try:
        # Import the main module at runtime
        from projects.pltv.main import main
        from src.writers import WriterType
        
        # Run the main pipeline with Snowflake writer
        main(writer_type=WriterType.SNOWFLAKE, reset_schema=False)
        
        result["completed_at"] = datetime.now().isoformat()
        result["duration_seconds"] = round(time.time() - start_time, 2)
        logger.info(f"PLTV sproc completed successfully in {result['duration_seconds']}s")
        return json.dumps(result)
        
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        result["completed_at"] = datetime.now().isoformat()
        result["duration_seconds"] = round(time.time() - start_time, 2)
        logger.error(f"PLTV sproc failed: {e}")
        return json.dumps(result)
