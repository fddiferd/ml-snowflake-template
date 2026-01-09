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