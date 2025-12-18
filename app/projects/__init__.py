"""Projects package for ML Layer workspace"""

from enum import Enum
from typing import Protocol

from projects.pltv import config as pltv_config

# Export for convenience
__all__ = ["Project"]


class Project(Enum):
    PLTV = "PLTV"

    @property
    def schema_name(self) -> str:
        return self.value.upper()

    @property
    def config(self) -> 'ProjectConfig':
        match self:
            case Project.PLTV:
                return pltv_config


class ProjectConfig(Protocol):
    version_number: int
    timestamp_col: str | None = None
    partition_col: str | None = None

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    from pprint import pformat

    print("Project Enum Members and Configs:")
    for project in Project:
        print(f"\nProject: {project.name}")
        print(f"  schema_name: {project.schema_name}")
        print(f"  config:\n{pformat(project.config.__dict__)}")
