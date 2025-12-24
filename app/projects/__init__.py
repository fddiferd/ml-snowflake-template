"""Projects package for ML Layer workspace"""

from enum import Enum

# Export for convenience
__all__ = ["Project"]


class Project(Enum):
    PLTV = "PLTV"

    @property
    def schema_name(self) -> str:
        return self.value.upper()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    from pprint import pformat

    print("Project Enum Members and Configs:")
    for project in Project:
        print(f"\nProject: {project.name}")
        print(f"  schema_name: {project.schema_name}")
