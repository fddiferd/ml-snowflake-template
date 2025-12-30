"""Projects package for ML Layer workspace"""

from enum import Enum

# Export for convenience
__all__ = ["Project"]


class Project(Enum):
    CORE = "CORE" # general core database

    # specific projects
    PLTV = "PLTV"

    @property
    def database_name(self) -> str:
        if self == Project.CORE:
            return "ML_LAYER_DB"
        return f"ML_LAYER_{self.value.upper()}_DB"

    @property
    def name(self) -> str:
        return self.value.upper()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Project Enum Members and Configs:")
    for project in Project:
        logger.info(f"Project: {project.name} - database name: {project.database_name}")


# Notebook func
from src.environment import environment

def set_project_root_for_notebook(master_project_path: str = environment.master_project_path):
    """Ensure that the current working directory is the folder below 'master_project_path' (default: 'GitHub')"""
    import os
    import logging
    logger = logging.getLogger(__name__)
    
    cwd = os.getcwd()
    # Keep going up until the parent folder is the master_project_path
    while os.path.basename(os.path.dirname(cwd)) != master_project_path:
        parent = os.path.dirname(cwd)
        if parent == cwd:
            # Reached filesystem root without finding master_project_path
            raise ValueError(f"Could not find parent folder '{master_project_path}' in path hierarchy")
        cwd = parent
    
    os.chdir(cwd)
    logger.info(f"Current working directory: {os.getcwd()}")