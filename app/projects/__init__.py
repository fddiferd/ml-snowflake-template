"""
Projects Package for ML Layer Workspace
========================================

Central registry of all ML Layer projects. Each project maps to:
- A Snowflake database (ML_LAYER_{PROJECT}_DB)
- A stored procedure ({PROJECT}_RUN)
- A scheduled task ({PROJECT}_WEEKLY_TASK)

To add a new project:
1. Add enum member here
2. Create app/projects/{project}/ directory structure
3. Create setup/{project}/tasks.sql
4. Create .github/workflows/deploy-{project}.yml
"""

from enum import Enum

# Export for convenience
__all__ = ["Project"]


class Project(Enum):
    """Enumeration of all ML Layer projects."""
    
    CORE = "CORE"  # General core database (ML_LAYER_DB)
    
    # Specific ML projects
    PLTV = "PLTV"  # Predicted Lifetime Value
    VBB = "VBB"    # VBB Model

    @property
    def database_name(self) -> str:
        """Get the Snowflake database name for this project."""
        if self == Project.CORE:
            return "ML_LAYER_DB"
        return f"ML_LAYER_{self.value.upper()}_DB"

    @property
    def procedure_name(self) -> str:
        """Get the stored procedure name for this project."""
        return f"{self.value.upper()}_RUN"

    @property
    def task_name(self) -> str:
        """Get the scheduled task name for this project."""
        return f"{self.value.upper()}_WEEKLY_TASK"

    @property
    def stage_path(self) -> str:
        """Get the stage path for uploading artifacts."""
        return f"{self.value.lower()}/ml_layer.zip"

    @property
    def handler_path(self) -> str:
        """Get the Python handler path for the stored procedure."""
        return f"projects.{self.value.lower()}.sproc.run_sproc"

    @property
    def main_module_path(self) -> str:
        """Get the Python module path for the main entry point."""
        return f"projects.{self.value.lower()}.main"


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Project Enum Members and Configs:")
    for project in Project:
        logger.info(f"Project: {project.name}")
        logger.info(f"  - Database: {project.database_name}")
        logger.info(f"  - Procedure: {project.procedure_name}")
        logger.info(f"  - Task: {project.task_name}")
        logger.info(f"  - Handler: {project.handler_path}")


