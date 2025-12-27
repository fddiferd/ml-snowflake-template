"""Projects package for ML Layer workspace"""

from enum import Enum

# Export for convenience
__all__ = ["Project"]


class Project(Enum):
    PLTV = "PLTV"

    @property
    def database_name(self) -> str:
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
        logger.info(f"\nProject: {project.name}")
        logger.info(f"  database_name: {project.database_name}")
