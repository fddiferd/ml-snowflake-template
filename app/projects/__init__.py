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


